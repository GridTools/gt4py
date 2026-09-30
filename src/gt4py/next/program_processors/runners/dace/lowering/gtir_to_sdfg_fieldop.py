# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Construction of the map scope and the result fields of a field operator.

The helpers in this module are shared by the lowering of regular field operators
(`gtir_to_sdfg_primitives.translate_as_fieldop()`) and of scan field operators
(`gtir_to_sdfg_scan.translate_scan_fieldop()`). The two differ in which dimensions
are traversed by the map scope: a regular field operator maps over the full domain,
while a scan traverses the column (vertical) dimension inside the stencil, by
means of a `LoopRegion`, and therefore maps only over the horizontal dimensions.
This is expressed by the `inner_dims` argument, that is the dimensions which the
dataflow computes by itself: they are excluded from the map range and written in
full shape by the output memlet.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import dace
from dace import nodes as dace_nodes, subsets as dace_subsets

from gt4py.eve.xtyping import MaybeNestedInTuple
from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import domain_utils, ir_makers as im
from gt4py.next.iterator.transforms import infer_domain
from gt4py.next.program_processors.runners.dace.lowering import (
    gtir_domain,
    gtir_to_sdfg,
    gtir_to_sdfg_lambda,
    gtir_to_sdfg_types,
    gtir_to_sdfg_utils,
)
from gt4py.next.type_system import type_specifications as ts


def parse_fieldop_arg(
    node: gtir.Expr,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    domain: gtir_domain.FieldopDomain,
    by_value: bool = False,
) -> MaybeNestedInTuple[gtir_to_sdfg_lambda.IteratorExpr | gtir_to_sdfg_lambda.MemletExpr]:
    """
    Helper method to visit an expression passed as argument to a field operator
    and create the local view for the field argument.

    Args:
        node: The GTIR expression passed as argument to the field operator.
        ctx: The SDFG context in which to lower the field operator.
        sdfg_builder: The object used to build the map scope in the provided SDFG.
        domain: The domain of the field operator.
        by_value: When `True`, a field argument is passed by value, that is as a
            `MemletExpr` with the full field shape, rather than as an iterator.
            This is the case of a scan field operator, where the stencil accesses
            the full column of its arguments.

    Returns:
        The local view of the argument, in the form of a tuple in case of a tuple
        of fields.
    """

    def parse_arg(
        arg: gtir_to_sdfg_types.FieldopData,
    ) -> gtir_to_sdfg_lambda.IteratorExpr | gtir_to_sdfg_lambda.MemletExpr:
        arg_expr = arg.get_local_view(domain, ctx.sdfg)
        if not by_value or isinstance(arg_expr, gtir_to_sdfg_lambda.MemletExpr):
            return arg_expr
        field_type = ts.FieldType(
            dims=[dim for dim, _ in arg_expr.field_domain], dtype=arg_expr.gt_dtype
        )
        return gtir_to_sdfg_lambda.MemletExpr(
            arg_expr.field, field_type, arg_expr.get_memlet_subset(ctx.sdfg)
        )

    arg = sdfg_builder.visit(node, ctx=ctx)

    if isinstance(arg, gtir_to_sdfg_types.FieldopData):
        return parse_arg(arg)
    else:
        # handle tuples of fields
        return gtx_utils.tree_map(parse_arg)(arg)


def _create_field_operator_impl(
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    output_edge: gtir_to_sdfg_lambda.DataflowOutputEdge | None,
    output_domain: infer_domain.NonTupleDomainAccess,
    output_type: ts.FieldType,
    map_exit: dace_nodes.MapExit | None,
    inner_dims: Sequence[gtx_common.Dimension],
    is_zero_dim: bool,
) -> gtir_to_sdfg_types.FieldopData | None:
    """
    Helper method to allocate a temporary array that stores one field computed
    by a field operator.

    This method is called by `create_field_operator()`, once for each field of
    the result, which is a tuple of fields in case of tuple return. For tuples, it
    can happen that one of the nested fields is not used outside of the field
    operator, and therefore does not need to be computed: in that case the domain
    inferred by gt4py on this field is empty, the corresponding `output_edge`
    argument is `None`, and this function returns `None` as well.

    Args:
        ctx: The SDFG context in which to lower the field operator.
        sdfg_builder: The object used to build the map scope in the provided SDFG.
        output_edge: The dataflow write edge representing the output data.
        output_domain: The domain of this field, as inferred by gt4py.
        output_type: The GT4Py field type descriptor.
        map_exit: The `MapExit` node of the field operator map scope, `None` when
            the dataflow computes all dimensions of the domain by itself.
        inner_dims: The dimensions that the dataflow computes by itself, therefore
            written in full shape rather than element-wise.
        is_zero_dim: The field operator has an empty domain, thus it computes a
            zero-dimensional field regardless of `output_domain`.

    Returns:
        The field data descriptor, which includes the field access node in the
        given `state` and the field domain offset, or `None` if the field does
        not need to be computed.
    """
    if output_edge is None:
        # According to domain inference, this tuple field does not need to be computed.
        assert output_domain == infer_domain.DomainAccessDescriptor.NEVER
        return None
    assert isinstance(output_domain, domain_utils.SymbolicDomain)
    # A zero-dimensional field operator is lowered to a trivial map, therefore its
    #  result is a zero-dimensional field even when domain inference has assigned it
    #  a domain, e.g. `make_const_list` of a scalar value used as `concat_where` branch.
    field_domain = [] if is_zero_dim else gtir_domain.get_field_domain(output_domain)

    dataflow_output_desc = output_edge.result.dc_node.desc(ctx.sdfg)

    # the memory layout of the output field follows the field operator compute domain
    field_dims, field_origin, field_shape = gtir_domain.get_field_layout(field_domain)
    if len(field_domain) == 0:
        # The field operator computes a zero-dimensional field, and the data subset
        # is set later depending on the element type (`ts.ListType` or `ts.ScalarType`)
        field_subset = dace_subsets.Range([])
    else:
        field_subset = gtir_domain.get_element_subset(field_dims, field_origin)

    if isinstance(output_edge.result.gt_dtype, ts.ScalarType):
        if output_edge.result.gt_dtype != output_type.dtype:
            raise TypeError(
                f"Type mismatch, expected {output_type.dtype} got {output_edge.result.gt_dtype}."
            )
        if len(inner_dims) == 0:
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
        assert output_edge.result.gt_dtype.offset_type is not None
        local_dim = output_edge.result.gt_dtype.offset_type
        # construct the full subset according to the canonical field domain
        extended_dims = gtx_common.order_dimensions([*field_dims, local_dim])
        local_idx = extended_dims.index(local_dim)

        field_shape.insert(local_idx, dataflow_output_desc.shape[0])
        field_subset = (
            dace_subsets.Range(field_subset[:local_idx])
            + dace_subsets.Range.from_array(dataflow_output_desc)
            + dace_subsets.Range(field_subset[local_idx:])
        )

    # allocate local temporary storage
    if len(inner_dims) != 0:
        # The dataflow writes the dimensions it computes by itself in full shape, for
        # each point of the map range on the other dimensions. The shape is taken from
        # the data written by the dataflow, not from the domain of this field: the
        # dataflow of a scan is lowered on the domain of the whole field operator,
        # which is the union of the domains of the fields in a tuple result.
        assert isinstance(dataflow_output_desc, dace.data.Array)
        for dim in inner_dims:
            dim_index = field_dims.index(dim)
            field_subset = (
                dace_subsets.Range(field_subset[:dim_index])
                + dace_subsets.Range.from_string(f"0:{dataflow_output_desc.shape[dim_index]}")
                + dace_subsets.Range(field_subset[dim_index + 1 :])
            )
        field_name, _ = sdfg_builder.add_temp_array_like(ctx.sdfg, dataflow_output_desc)
    elif len(field_shape) == 0:  # zero-dimensional field
        field_name, _ = sdfg_builder.add_temp_scalar(ctx.sdfg, dataflow_output_desc.dtype)
        field_subset = dace_subsets.Range.from_string("0")
    else:
        field_name, _ = sdfg_builder.add_temp_array(
            ctx.sdfg, field_shape, dataflow_output_desc.dtype
        )
    field_node = ctx.state.add_access(field_name)

    # and here the edge writing the dataflow result data through the map exit node
    last_node_removed = output_edge.connect(map_exit, field_node, field_subset)

    if len(inner_dims) != 0 and not last_node_removed:
        # The dataflow of a scan is lowered to a nested SDFG that writes the column
        # into a transient data container. This container is expected to be removed
        # by the connection above, so that the nested SDFG writes the result field
        # directly, see `gtir_to_sdfg_scan._handle_dataflow_result_of_nested_sdfg()`.
        raise ValueError("The scan nested SDFG is expected to write directly to the result field.")

    return gtir_to_sdfg_types.FieldopData(
        field_node, ts.FieldType(field_dims, output_edge.result.gt_dtype), tuple(field_origin)
    )


def create_field_operator(
    ctx: gtir_to_sdfg.SubgraphContext,
    domain: gtir_domain.FieldopDomain,
    node_type: ts.FieldType | ts.TupleType,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    input_edges: Iterable[gtir_to_sdfg_lambda.DataflowInputEdge],
    output: MaybeNestedInTuple[gtir_to_sdfg_lambda.DataflowOutputEdge | None],
    output_domain: infer_domain.DomainAccess,
    inner_dims: Sequence[gtx_common.Dimension] = (),
) -> gtir_to_sdfg_types.FieldopResult:
    """
    Helper method to build the output of a field operator, which can consist of
    a single field or a tuple of fields.

    Args:
        ctx: The SDFG context in which to lower the field operator.
        domain: The domain of the field operator, used as map range.
        node_type: The GT4Py type of the IR node that produces this field.
        sdfg_builder: The object used to build the map scope in the provided SDFG.
        input_edges: List of edges to pass input data into the dataflow.
        output: Edge corresponding to the dataflow output, or a tuple of edges in
            case the field operator computes a tuple of fields. The edge is `None`
            for the fields that do not need to be computed.
        output_domain: The domain of the result, as inferred by gt4py, in the form
            of a tuple in case the field operator computes a tuple of fields.
        inner_dims: The dimensions that the dataflow computes by itself, therefore
            excluded from the map range. This is the column dimension of a scan.

    Returns:
        The descriptor of the field operator result, which is a single field or a
        tuple of fields defined on the domain of the field operator.
    """
    map_ranges = [domain_range for domain_range in domain if domain_range.dim not in inner_dims]

    if len(domain) == 0:
        # create a trivial map for zero-dimensional fields
        map_entry, map_exit = sdfg_builder.add_map("fieldop", ctx.state, {"__gt4py_zerodim": "0"})
    elif len(map_ranges) == 0:
        # The dataflow computes all dimensions of the domain by itself, therefore
        # no map scope is needed. This is the case of a scan field operator on a
        # 1d domain, containing only the column dimension.
        map_entry, map_exit = (None, None)
    else:
        # create map range corresponding to the field operator domain
        map_entry, map_exit = sdfg_builder.add_map(
            "fieldop",
            ctx.state,
            {
                gtir_to_sdfg_utils.get_map_variable(
                    domain_range.dim
                ): f"{domain_range.start}:{domain_range.stop}"
                for domain_range in map_ranges
            },
        )

    # here we setup the edges passing through the map entry node
    for edge in input_edges:
        edge.connect(map_entry)

    # Note that `output_symbols` below is not used, we only need the tree-like
    # structure to get the type of each nested field in the `tree_map` visitor.
    output_symbols = (
        gtir_to_sdfg_utils.make_symbol_tree("__gtir_unused_dummy_var", node_type)
        if isinstance(node_type, ts.TupleType)
        else im.sym("__gtir_unused_dummy_var", node_type)
    )

    return gtx_utils.tree_map(
        lambda edge, field_domain, sym: _create_field_operator_impl(
            ctx, sdfg_builder, edge, field_domain, sym.type, map_exit, inner_dims, len(domain) == 0
        )
    )(output, output_domain, output_symbols)
