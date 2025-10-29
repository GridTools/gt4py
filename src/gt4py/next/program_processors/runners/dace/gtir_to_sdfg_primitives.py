# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Iterable, Optional, Protocol

import dace
from dace import subsets as dace_subsets

from gt4py.eve.extended_typing import MaybeNestedInTuple
from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.program_processors.runners.dace import (
    gtir_dataflow,
    gtir_domain,
    gtir_python_codegen,
    gtir_to_sdfg,
    gtir_to_sdfg_types,
    gtir_to_sdfg_utils,
    utils as gtx_dace_utils,
)
from gt4py.next.program_processors.runners.dace.gtir_to_sdfg_concat_where import (
    translate_concat_where,
)
from gt4py.next.program_processors.runners.dace.gtir_to_sdfg_scan import translate_scan
from gt4py.next.type_system import type_info as ti, type_specifications as ts


if TYPE_CHECKING:
    from gt4py.next.program_processors.runners.dace import gtir_to_sdfg


class PrimitiveTranslator(Protocol):
    @abc.abstractmethod
    def __call__(
        self,
        node: gtir.Node,
        ctx: gtir_to_sdfg.SubgraphContext,
        sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    ) -> gtir_to_sdfg_types.FieldopResult:
        """Creates the dataflow subgraph representing a GTIR primitive function.

        This method is used by derived classes to build a specialized subgraph
        for a specific GTIR primitive function.

        Args:
            node: The GTIR node describing the primitive to be lowered.
            ctx: The SDFG context in which to lower the primitive subgraph.
            sdfg_builder: The object responsible for visiting child nodes of the primitive node.

        Returns:
            A list of data access nodes and the associated GT4Py data type, which provide
            access to the result of the primitive subgraph. The GT4Py data type is useful
            in the case the returned data is an array, because the type provdes the domain
            information (e.g. order of dimensions, dimension types).
        """


def _parse_fieldop_arg(
    node: gtir.Expr,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    domain: gtir_domain.FieldopDomain,
) -> MaybeNestedInTuple[gtir_dataflow.IteratorExpr | gtir_dataflow.MemletExpr]:
    """
    Helper method to visit an expression passed as argument to a field operator
    and create the local view for the field argument.
    """
    arg = sdfg_builder.visit(node, ctx=ctx)

    if not isinstance(arg, gtir_to_sdfg_types.FieldopData):
        raise ValueError("Expected a field, found a tuple of fields.")
    return arg.get_local_view(domain, ctx.sdfg)


def _create_field_operator_impl(
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    field_domain: gtir_domain.FieldopDomain,
    output_edge: gtir_dataflow.DataflowOutputEdge,
    output_type: ts.FieldType,
    map_exit: dace.nodes.MapExit,
) -> gtir_to_sdfg_types.FieldopData:
    """
    Helper method to allocate a temporary array that stores one field computed
    by a field operator.

    This method is called by `_create_field_operator()`.

    Args:
        ctx: The SDFG context in which to lower the field operator.
        sdfg_builder: The object used to build the map scope in the provided SDFG.
        domain: The domain of the field operator that computes the field.
        output_edge: The dataflow write edge representing the output data.
        output_type: The GT4Py field type descriptor.
        map_exit: The `MapExit` node of the field operator map scope.

    Returns:
        The field data descriptor, which includes the field access node in the
        given `state` and the field domain offset.
    """
    dataflow_output_desc = output_edge.result.dc_node.desc(ctx.sdfg)

    # the memory layout of the output field follows the field operator compute domain
    field_dims, field_origin, field_shape = gtir_domain.get_field_layout(field_domain)
    if len(field_domain) == 0:
        # The field operator computes a zero-dimensional field, and the data subset
        # is set later depending on the element type (`ts.ListType` or `ts.ScalarType`)
        field_subset = dace_subsets.Range([])
    else:
        field_indices = gtir_domain.get_domain_indices(field_dims, field_origin)
        field_subset = dace_subsets.Range.from_indices(field_indices)

    if isinstance(output_edge.result.gt_dtype, ts.ScalarType):
        if output_edge.result.gt_dtype != output_type.dtype:
            raise TypeError(
                f"Type mismatch, expected {output_type.dtype} got {output_edge.result.gt_dtype}."
            )
        assert isinstance(dataflow_output_desc, dace.data.Scalar)
    else:
        assert isinstance(output_type.dtype, ts.ListType)
        assert isinstance(output_edge.result.gt_dtype.element_type, ts.ScalarType)
        if output_edge.result.gt_dtype.element_type != output_type.dtype.element_type:
            raise TypeError(
                f"Type mismatch, expected {output_type.dtype.element_type} got {output_edge.result.gt_dtype.element_type}."
            )
        assert isinstance(dataflow_output_desc, dace.data.Array)
        assert len(dataflow_output_desc.shape) == 1
        # extend the array with the local dimensions added by the field operator (e.g. `neighbors`)
        assert all(dim.kind != gtx_common.DimensionKind.LOCAL for dim in field_dims)
        local_dim: gtx_common.Dimension = output_edge.result.gt_dtype.offset_type  # type: ignore[assignment]  # checked in ValueExpr.__post_init__
        # construct the full subset according to the canonical field domain
        extended_dims = gtx_common.order_dimensions(
            [*field_dims, output_edge.result.gt_dtype.offset_type]  # type: ignore[list-item]  # checked in ValueExpr.__post_init__
        )
        local_idx = extended_dims.index(local_dim)

        field_shape.insert(local_idx, dataflow_output_desc.shape[0])
        field_subset = (
            dace_subsets.Range(field_subset[:local_idx])
            + dace_subsets.Range.from_array(dataflow_output_desc)
            + dace_subsets.Range(field_subset[local_idx:])
        )

    # allocate local temporary storage
    if len(field_shape) == 0:  # zero-dimensional field
        field_name, _ = sdfg_builder.add_temp_scalar(ctx.sdfg, dataflow_output_desc.dtype)
        field_subset = dace_subsets.Range.from_string("0")
    else:
        field_name, _ = sdfg_builder.add_temp_array(
            ctx.sdfg, field_shape, dataflow_output_desc.dtype
        )
    field_node = ctx.state.add_access(field_name)

    # and here the edge writing the dataflow result data through the map exit node
    output_edge.connect(map_exit, field_node, field_subset)

    return gtir_to_sdfg_types.FieldopData(
        field_node, ts.FieldType(field_dims, output_edge.result.gt_dtype), tuple(field_origin)
    )


def _create_field_operator(
    ctx: gtir_to_sdfg.SubgraphContext,
    domain: gtir_domain.FieldopDomain,
    node_type: ts.FieldType | ts.TupleType,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    input_edges: Iterable[gtir_dataflow.DataflowInputEdge],
    output_tree: tuple[gtir_dataflow.DataflowOutputEdge | tuple[Any, ...], ...],
) -> gtir_to_sdfg_types.FieldopResult:
    """
    Helper method to build the output of a field operator, which can consist of
    a single field or a tuple of fields.

    A tuple of fields is returned when one stencil computes a grid point on multiple
    fields: for each field, this method will call `_create_field_operator_impl()`.

    Args:
        ctx: The SDFG context in which to lower the field operator.
        domain: The domain of the field operator that computes the field.
        node_type: The GT4Py type of the IR node that produces this field.
        sdfg_builder: The object used to build the map scope in the provided SDFG.
        input_edges: List of edges to pass input data into the dataflow.
        output_tree: A tree representation of the dataflow output data.

    Returns:
        The descriptor of the field operator result, which can be either a single
        field or a tuple fields.
    """

    if len(domain) == 0:
        # create a trivial map for zero-dimensional fields
        map_range = {
            "__gt4py_zerodim": "0",
        }
    else:
        # create map range corresponding to the field operator domain
        map_range = {
            gtir_to_sdfg_utils.get_map_variable(
                domain_range.dim
            ): f"{domain_range.start}:{domain_range.stop}"
            for domain_range in domain
        }
    map_entry, map_exit = sdfg_builder.add_map("fieldop", ctx.state, map_range)

    # here we setup the edges passing through the map entry node
    for edge in input_edges:
        edge.connect(map_entry)

    if isinstance(node_type, ts.FieldType):
        assert len(output_tree) == 1 and isinstance(
            output_tree[0], gtir_dataflow.DataflowOutputEdge
        )
        output_edge = output_tree[0]
        return _create_field_operator_impl(
            ctx, sdfg_builder, domain, output_edge, node_type, map_exit
        )
    else:
        # handle tuples of fields
        output_symbol_tree = gtir_to_sdfg_utils.make_symbol_tree("x", node_type)
        return gtx_utils.tree_map(
            lambda edge, sym, ctx_=ctx: _create_field_operator_impl(
                ctx_, sdfg_builder, domain, edge, sym.type, map_exit
            )
        )(output_tree, output_symbol_tree)


def translate_as_fieldop(
    node: gtir.Node,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopResult:
    """
    Generates the dataflow subgraph for the `as_fieldop` builtin function.

    Expects a `FunCall` node with two arguments:
    1. a lambda function representing the stencil, which is lowered to a dataflow subgraph
    2. the domain of the field operator, which is used as map range

    The dataflow can be as simple as a single tasklet, or implement a local computation
    as a composition of tasklets and even include a map to range on local dimensions (e.g.
    neighbors and map builtins).
    The stencil dataflow is instantiated inside a map scope, which applies the stencil
    over the field domain.
    """
    assert isinstance(node, gtir.FunCall)
    assert cpm.is_call_to(node.fun, "as_fieldop")
    assert isinstance(node.type, (ts.FieldType, ts.TupleType))

    fun_node = node.fun
    assert len(fun_node.args) == 2
    fieldop_expr, fieldop_domain_expr = fun_node.args

    if cpm.is_call_to(fieldop_expr, "scan"):
        return translate_scan(node, ctx, sdfg_builder)

    if cpm.is_ref_to(fieldop_expr, "deref"):
        # Special usage of 'deref' as argument to fieldop expression, to pass a scalar
        # value to 'as_fieldop' function. It results in broadcasting the scalar value
        # over the field domain.
        assert isinstance(node.type, ts.FieldType)
        stencil_expr = im.lambda_("a")(im.deref("a"))
        stencil_expr.expr.type = node.type.dtype
    elif isinstance(fieldop_expr, gtir.Lambda):
        # Default case, handled below: the argument expression is a lambda function
        # representing the stencil operation to be computed over the field domain.
        stencil_expr = fieldop_expr
    else:
        raise NotImplementedError(
            f"Expression type '{type(fieldop_expr)}' not supported as argument to 'as_fieldop' node."
        )

    # parse the domain of the field operator
    assert isinstance(fieldop_domain_expr.type, ts.DomainType)
    field_domain = gtir_domain.get_field_domain(
        domain_utils.SymbolicDomain.from_expr(fieldop_domain_expr)
    )

    # visit the list of arguments to be passed to the lambda expression
    fieldop_args = [_parse_fieldop_arg(arg, ctx, sdfg_builder, field_domain) for arg in node.args]

    # represent the field operator as a mapped tasklet graph, which will range over the field domain
    input_edges, output_edges = gtir_dataflow.translate_lambda_to_dataflow(
        ctx.sdfg, ctx.state, sdfg_builder, stencil_expr, fieldop_args
    )

    return _create_field_operator(
        ctx, field_domain, node.type, sdfg_builder, input_edges, output_edges
    )


def _construct_if_branch_output(
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    field_domain: gtir_domain.FieldopDomain,
    true_br: gtir_to_sdfg_types.FieldopData,
    false_br: gtir_to_sdfg_types.FieldopData,
) -> gtir_to_sdfg_types.FieldopData:
    """
    Helper function called by `translate_if()` to allocate a temporary field to store
    the result of an if expression.
    """
    assert true_br.gt_type == false_br.gt_type
    out_type = true_br.gt_type

    if isinstance(out_type, ts.ScalarType):
        dtype = gtx_dace_utils.as_dace_type(out_type)
        out, _ = sdfg_builder.add_temp_scalar(ctx.sdfg, dtype)
        out_node = ctx.state.add_access(out)
        return gtir_to_sdfg_types.FieldopData(out_node, out_type, origin=())

    assert isinstance(out_type, ts.FieldType)
    dims, origin, shape = gtir_domain.get_field_layout(field_domain)
    assert dims == out_type.dims

    if isinstance(out_type.dtype, ts.ScalarType):
        dtype = gtx_dace_utils.as_dace_type(out_type.dtype)
    else:
        assert isinstance(out_type.dtype, ts.ListType)
        assert out_type.dtype.offset_type is not None
        assert isinstance(out_type.dtype.element_type, ts.ScalarType)
        dtype = gtx_dace_utils.as_dace_type(out_type.dtype.element_type)
        offset_provider_type = sdfg_builder.get_offset_provider_type(
            out_type.dtype.offset_type.value
        )
        assert isinstance(offset_provider_type, gtx_common.NeighborConnectivityType)
        shape = [*shape, offset_provider_type.max_neighbors]

    out, _ = sdfg_builder.add_temp_array(ctx.sdfg, shape, dtype)
    out_node = ctx.state.add_access(out)

    return gtir_to_sdfg_types.FieldopData(out_node, out_type, tuple(origin))


def _write_if_branch_output(
    ctx: gtir_to_sdfg.SubgraphContext,
    src: gtir_to_sdfg_types.FieldopData,
    dst: gtir_to_sdfg_types.FieldopData,
) -> None:
    """
    Helper function called by `translate_if()` to write the result of an if-branch,
    here `src` field, to the output 'dst' field. The data subset is based on the
    domain of the `dst` field. Therefore, the full shape of `dst` array is written.
    """
    if src.gt_type != dst.gt_type:
        raise ValueError(
            f"Source and destination type mismatch, '{dst.gt_type}' vs '{src.gt_type}'."
        )
    dst_node = ctx.state.add_access(dst.dc_node.data)
    dst_shape = dst_node.desc(ctx.sdfg).shape

    if isinstance(src.gt_type, ts.ScalarType):
        ctx.state.add_nedge(
            src.dc_node,
            dst_node,
            dace.Memlet(data=src.dc_node.data, subset="0"),
        )
    else:
        if isinstance(src.gt_type.dtype, ts.ListType):
            src_origin = [*src.origin, 0]
            dst_origin = [*dst.origin, 0]
        else:
            src_origin = [*src.origin]
            dst_origin = [*dst.origin]

        data_subset = dace_subsets.Range(
            (
                f"{dst_start - src_start}",
                f"{dst_start - src_start + size - 1}",  # subtract 1 because the range boundaries are included
                1,
            )
            for src_start, dst_start, size in zip(src_origin, dst_origin, dst_shape, strict=True)
        )

        ctx.state.add_nedge(
            src.dc_node,
            dst_node,
            dace.Memlet(
                data=src.dc_node.data,
                subset=data_subset,
                other_subset=dace_subsets.Range.from_array(dst_node.desc(ctx.sdfg)),
            ),
        )


def translate_if(
    node: gtir.Node,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopResult:
    """Generates the dataflow subgraph for the `if_` builtin function."""
    assert cpm.is_call_to(node, "if_")
    assert len(node.args) == 3
    cond_expr, true_expr, false_expr = node.args

    # expect condition as first argument
    if_stmt = gtir_python_codegen.get_source(cond_expr)

    # evaluate the if-condition in a new entry state and use the current head state
    # to join the true/false branch states as follows:
    #
    #               ------------
    #           === |   cond   | ===
    #          ||   ------------   ||
    #          \/                  \/
    #     ------------       -------------
    #     |   true   |       |   false   |
    #     ------------       -------------
    #          ||                  ||
    #          ||   ------------   ||
    #           ==> |   head   | <==
    #               ------------
    #
    cond_state = ctx.sdfg.add_state_before(ctx.state, ctx.state.label + "_cond")
    ctx.sdfg.remove_edge(ctx.sdfg.out_edges(cond_state)[0])

    # expect true branch as second argument
    true_state = ctx.sdfg.add_state(ctx.state.label + "_true_branch")
    tbranch_ctx = gtir_to_sdfg.SubgraphContext(ctx.sdfg, true_state)
    ctx.sdfg.add_edge(cond_state, true_state, dace.InterstateEdge(condition=if_stmt))
    ctx.sdfg.add_edge(true_state, ctx.state, dace.InterstateEdge())

    # and false branch as third argument
    false_state = ctx.sdfg.add_state(ctx.state.label + "_false_branch")
    fbranch_ctx = gtir_to_sdfg.SubgraphContext(ctx.sdfg, false_state)
    ctx.sdfg.add_edge(cond_state, false_state, dace.InterstateEdge(condition=f"not({if_stmt})"))
    ctx.sdfg.add_edge(false_state, ctx.state, dace.InterstateEdge())

    true_br_result = sdfg_builder.visit(true_expr, ctx=tbranch_ctx)
    false_br_result = sdfg_builder.visit(false_expr, ctx=fbranch_ctx)

    node_output = gtx_utils.tree_map(
        lambda domain,
        true_br,
        false_br,
        _ctx=ctx,
        sdfg_builder=sdfg_builder: _construct_if_branch_output(
            ctx=_ctx,
            sdfg_builder=sdfg_builder,
            field_domain=gtir_domain.get_field_domain(domain),
            true_br=true_br,
            false_br=false_br,
        )
    )(
        node.annex.domain,
        true_br_result,
        false_br_result,
    )
    gtx_utils.tree_map(lambda src, dst, _ctx=tbranch_ctx: _write_if_branch_output(_ctx, src, dst))(
        true_br_result, node_output
    )
    gtx_utils.tree_map(lambda src, dst, _ctx=fbranch_ctx: _write_if_branch_output(_ctx, src, dst))(
        false_br_result, node_output
    )

    return node_output


def translate_index(
    node: gtir.Node,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopResult:
    """
    Lowers the `index` builtin function to a mapped tasklet that writes the dimension
    index values to a transient array. The extent of the index range is taken from
    the domain information that should be present in the node annex.
    """
    assert cpm.is_call_to(node, "index")
    assert isinstance(node.type, ts.FieldType)

    field_domain = gtir_domain.get_field_domain(node.annex.domain)
    assert len(field_domain) == 1
    dim_index = gtir_to_sdfg_utils.get_map_variable(field_domain[0].dim)

    index_data, _ = sdfg_builder.add_temp_scalar(ctx.sdfg, gtir_to_sdfg_types.INDEX_DTYPE)
    index_node = ctx.state.add_access(index_data)
    index_value = gtir_dataflow.ValueExpr(
        dc_node=index_node,
        gt_dtype=gtx_dace_utils.as_itir_type(gtir_to_sdfg_types.INDEX_DTYPE),
    )
    index_write_tasklet = sdfg_builder.add_tasklet(
        "index",
        ctx.state,
        inputs={},
        outputs={"__val"},
        code=f"__val = {dim_index}",
    )
    ctx.state.add_edge(
        index_write_tasklet,
        "__val",
        index_node,
        None,
        dace.Memlet(data=index_data, subset="0"),
    )

    input_edges = [
        gtir_dataflow.EmptyInputEdge(ctx.state, index_write_tasklet),
    ]
    output_edge = gtir_dataflow.DataflowOutputEdge(ctx.state, index_value)
    return _create_field_operator(
        ctx, field_domain, node.type, sdfg_builder, input_edges, (output_edge,)
    )


def _get_data_nodes(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    data_name: str,
    data_type: ts.DataType,
) -> gtir_to_sdfg_types.FieldopResult:
    if isinstance(data_type, ts.FieldType):
        data_node = state.add_access(data_name)
        return sdfg_builder.make_field(data_node, data_type)

    elif isinstance(data_type, ts.ScalarType):
        if data_name in sdfg.symbols:
            data_node = _get_symbolic_value(
                sdfg, state, sdfg_builder, data_name, data_type, temp_name=f"__{data_name}"
            )
        else:
            data_node = state.add_access(data_name)
        return gtir_to_sdfg_types.FieldopData(data_node, data_type, origin=())

    elif isinstance(data_type, ts.TupleType):
        symbol_tree = gtir_to_sdfg_utils.make_symbol_tree(data_name, data_type)
        return gtx_utils.tree_map(
            lambda sym: _get_data_nodes(sdfg, state, sdfg_builder, sym.id, sym.type)
        )(symbol_tree)

    else:
        raise NotImplementedError(f"Symbol type {type(data_type)} not supported.")


def _get_symbolic_value(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    symbolic_expr: dace.symbolic.SymExpr,
    scalar_type: ts.ScalarType,
    temp_name: Optional[str] = None,
) -> dace.nodes.AccessNode:
    tasklet_node = sdfg_builder.add_tasklet(
        "get_value",
        state,
        {},
        {"__out"},
        f"__out = {symbolic_expr}",
    )
    temp_name, _ = sdfg.add_scalar(
        temp_name or sdfg.temp_data_name(),
        gtx_dace_utils.as_dace_type(scalar_type),
        find_new_name=True,
        transient=True,
    )
    data_node = state.add_access(temp_name)
    state.add_edge(
        tasklet_node,
        "__out",
        data_node,
        None,
        dace.Memlet(data=temp_name, subset="0"),
    )
    return data_node


def translate_literal(
    node: gtir.Node,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopResult:
    """Generates the dataflow subgraph for a `ir.Literal` node."""
    assert isinstance(node, gtir.Literal)

    data_type = node.type
    data_node = _get_symbolic_value(ctx.sdfg, ctx.state, sdfg_builder, node.value, data_type)

    return gtir_to_sdfg_types.FieldopData(data_node, data_type, origin=())


def translate_make_tuple(
    node: gtir.Node,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopResult:
    assert cpm.is_call_to(node, "make_tuple")
    return tuple(sdfg_builder.visit(arg, ctx=ctx) for arg in node.args)


def translate_tuple_get(
    node: gtir.Node,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopResult:
    assert cpm.is_call_to(node, "tuple_get")
    assert len(node.args) == 2

    if not isinstance(node.args[0], gtir.Literal):
        raise ValueError("Tuple can only be subscripted with compile-time constants.")
    assert ti.is_integral(node.args[0].type)
    index = int(node.args[0].value)

    data_nodes = sdfg_builder.visit(node.args[1], ctx=ctx)
    if isinstance(data_nodes, gtir_to_sdfg_types.FieldopData):
        raise ValueError(f"Invalid tuple expression {node}")
    unused_arg_nodes: Iterable[gtir_to_sdfg_types.FieldopData] = gtx_utils.flatten_nested_tuple(
        tuple(arg for i, arg in enumerate(data_nodes) if arg is not None and i != index)
    )
    ctx.state.remove_nodes_from(
        [arg.dc_node for arg in unused_arg_nodes if ctx.state.degree(arg.dc_node) == 0]
    )
    return data_nodes[index]


def translate_scalar_expr(
    node: gtir.Node,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopResult:
    assert isinstance(node, gtir.FunCall)
    assert isinstance(node.type, ts.ScalarType)

    args = []
    connectors = []
    scalar_expr_args = []

    for i, arg_expr in enumerate(node.args):
        visit_expr = True
        if isinstance(arg_expr, gtir.SymRef):
            try:
                # check if symbol is defined in the GT4Py program, throws `KeyError` exception if undefined
                sdfg_builder.get_symbol_type(arg_expr.id)
            except KeyError:
                # all `SymRef` should refer to symbols defined in the program, except in case of non-variable argument,
                # e.g. the type name `float64` used in casting expressions like `cast_(variable, float64)`
                visit_expr = False

        if visit_expr:
            # we visit the argument expression and obtain the access node to
            # a scalar data container, which will be connected to the tasklet
            arg = sdfg_builder.visit(arg_expr, ctx=ctx)
            if not (
                isinstance(arg, gtir_to_sdfg_types.FieldopData)
                and isinstance(node.type, ts.ScalarType)
            ):
                raise ValueError(f"Invalid argument to scalar expression {arg_expr}.")
            param = f"__arg{i}"
            args.append(arg.dc_node)
            connectors.append(param)
            scalar_expr_args.append(gtir.SymRef(id=param))
        else:
            assert isinstance(arg_expr, gtir.SymRef)
            scalar_expr_args.append(arg_expr)

    # we visit the scalar expression replacing the input arguments with the corresponding data connectors
    scalar_node = gtir.FunCall(fun=node.fun, args=scalar_expr_args)
    python_code = gtir_python_codegen.get_source(scalar_node)
    tasklet_node = sdfg_builder.add_tasklet(
        name="scalar_expr",
        state=ctx.state,
        inputs=set(connectors),
        outputs={"__out"},
        code=f"__out = {python_code}",
    )
    # create edges for the input data connectors
    for arg_node, conn in zip(args, connectors, strict=True):
        ctx.state.add_edge(
            arg_node,
            None,
            tasklet_node,
            conn,
            dace.Memlet(data=arg_node.data, subset="0"),
        )
    # finally, create temporary for the result value
    temp_name, _ = sdfg_builder.add_temp_scalar(ctx.sdfg, gtx_dace_utils.as_dace_type(node.type))
    temp_node = ctx.state.add_access(temp_name)
    ctx.state.add_edge(
        tasklet_node,
        "__out",
        temp_node,
        None,
        dace.Memlet(data=temp_name, subset="0"),
    )

    return gtir_to_sdfg_types.FieldopData(temp_node, node.type, origin=())


def translate_symbol_ref(
    node: gtir.Node,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopResult:
    """Generates the dataflow subgraph for a `ir.SymRef` node."""
    assert isinstance(node, gtir.SymRef)

    symbol_name = str(node.id)
    # we retrieve the type of the symbol in the GT4Py prgram
    gt_symbol_type = sdfg_builder.get_symbol_type(symbol_name)

    # Create new access node in current state. It is possible that multiple
    # access nodes are created in one state for the same data container.
    # We rely on the dace simplify pass to remove duplicated access nodes.
    return _get_data_nodes(ctx.sdfg, ctx.state, sdfg_builder, symbol_name, gt_symbol_type)


if TYPE_CHECKING:
    # Use type-checking to assert that all translator functions implement the `PrimitiveTranslator` protocol
    __primitive_translators: list[PrimitiveTranslator] = [
        translate_as_fieldop,
        translate_concat_where,
        translate_if,
        translate_index,
        translate_literal,
        translate_make_tuple,
        translate_tuple_get,
        translate_scalar_expr,
        translate_scan,
        translate_symbol_ref,
    ]
