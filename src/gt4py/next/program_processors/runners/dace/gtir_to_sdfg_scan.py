# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the lowering of scan field operator.

This builtin translator implements the `PrimitiveTranslator` protocol as other
translators in `gtir_to_sdfg_primitives` module. This module implements the scan
translator, separately from the `gtir_to_sdfg_primitives` module, because the
parsing of input arguments as well as the construction of the map scope differ
from a regular field operator, which requires slightly different helper methods.
Besides, the function code is quite large, another reason to keep it separate
from other translators.

The current GTIR representation of the scan operator is based on iterator view.
This is likely to change in the future, to enable GTIR optimizations for scan.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable

import dace
from dace import subsets as dace_subsets

from gt4py import eve
from gt4py.eve.extended_typing import MaybeNestedInTuple
from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms import infer_domain
from gt4py.next.program_processors.runners.dace import (
    gtir_dataflow,
    gtir_domain,
    gtir_to_sdfg,
    gtir_to_sdfg_types,
    gtir_to_sdfg_utils,
)
from gt4py.next.type_system import type_info as ti, type_specifications as ts


def _parse_scan_fieldop_arg(
    node: gtir.Expr,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    field_domain: gtir_domain.FieldopDomain,
) -> gtir_dataflow.MemletExpr | tuple[gtir_dataflow.MemletExpr | tuple[Any, ...], ...]:
    """Helper method to visit an expression passed as argument to a scan field operator.

    On the innermost level, a scan operator is lowered to a loop region which computes
    column elements in the vertical dimension.

    It differs from the helper method `gtir_to_sdfg_primitives` in that field arguments
    are passed in full shape along the vertical dimension, rather than as iterator.
    """

    def _parse_fieldop_arg_impl(
        arg: gtir_to_sdfg_types.FieldopData,
    ) -> gtir_dataflow.MemletExpr:
        arg_expr = arg.get_local_view(field_domain, ctx.sdfg)
        if isinstance(arg_expr, gtir_dataflow.MemletExpr):
            return arg_expr
        # In scan field operator, the arguments to the vertical stencil are passed by value.
        # Therefore, the full field shape is passed as `MemletExpr` rather than `IteratorExpr`.
        field_type = ts.FieldType(
            dims=[dim for dim, _ in arg_expr.field_domain], dtype=arg_expr.gt_dtype
        )
        return gtir_dataflow.MemletExpr(
            arg_expr.field, field_type, arg_expr.get_memlet_subset(ctx.sdfg)
        )

    arg = sdfg_builder.visit(node, ctx=ctx)

    if isinstance(arg, gtir_to_sdfg_types.FieldopData):
        return _parse_fieldop_arg_impl(arg)
    else:
        # handle tuples of fields
        return gtx_utils.tree_map(lambda x: _parse_fieldop_arg_impl(x))(arg)


def _create_scan_field_operator_impl(
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    output_edge: gtir_dataflow.DataflowOutputEdge | None,
    output_domain: infer_domain.NonTupleDomainAccess,
    output_type: ts.FieldType,
    map_exit: dace.nodes.MapExit,
) -> gtir_to_sdfg_types.FieldopData | None:
    """
    Helper method to allocate a temporary array that stores one field computed
    by the scan field operator.

    This method is called by `_create_scan_field_operator()`.

    Similar to `gtir_to_sdfg_primitives._create_field_operator_impl()` but
    for scan field operators. It differs in that the scan loop region produces
    a field along the vertical dimension, rather than a single point.
    Therefore, the memlet subset will write a slice into the result array, that
    corresponds to the full vertical shape for each horizontal grid point.

    Refer to `gtir_to_sdfg_primitives._create_field_operator_impl()` for
    the description of function arguments and return values.
    """
    if output_edge is None:
        # According to domain inference, this tuple field does not need to be computed.
        assert output_domain == infer_domain.DomainAccessDescriptor.NEVER
        return None
    assert isinstance(output_domain, domain_utils.SymbolicDomain)
    field_domain = gtir_domain.get_field_domain(output_domain)

    dataflow_output_desc = output_edge.result.dc_node.desc(ctx.sdfg)
    assert isinstance(dataflow_output_desc, dace.data.Array)

    if isinstance(output_edge.result.gt_dtype, ts.ScalarType):
        assert isinstance(output_type.dtype, ts.ScalarType)
        if output_edge.result.gt_dtype != output_type.dtype:
            raise TypeError(
                f"Type mismatch, expected {output_type.dtype} got {output_edge.result.gt_dtype}."
            )
        # the scan field operator computes a column of scalar values
        assert len(dataflow_output_desc.shape) == 1
    else:
        raise NotImplementedError("scan with list output is not supported")

    # the memory layout of the output field follows the field operator compute domain
    field_dims, field_origin, field_shape = gtir_domain.get_field_layout(field_domain)
    field_indices = gtir_domain.get_domain_indices(field_dims, field_origin)
    field_subset = dace_subsets.Range.from_indices(field_indices)

    # the vertical dimension used as scan column is computed by the `LoopRegion`
    # inside the map scope, therefore it is excluded from the map range
    scan_dim_index = [sdfg_builder.is_column_axis(dim) for dim in field_dims].index(True)

    # the map scope writes the full-shape dimension corresponding to the scan column
    field_subset = (
        dace_subsets.Range(field_subset[:scan_dim_index])
        + dace_subsets.Range.from_string(f"0:{dataflow_output_desc.shape[0]}")
        + dace_subsets.Range(field_subset[scan_dim_index + 1 :])
    )

    # Create the final data storage, that is outside of the surrounding Map.
    field_name, field_desc = sdfg_builder.add_temp_array(
        ctx.sdfg, field_shape, dataflow_output_desc.dtype
    )
    field_node = ctx.state.add_access(field_name)

    # Now connect the final output node with the output of the nested SDFG.
    #  Up to now the nested SDFG is writing into a transient data container that
    #  has the size to hold one column. The function below, that does the connection,
    #  will remove that data container.
    inner_map_output_temporary_removed = output_edge.connect(map_exit, field_node, field_subset)
    assert ctx.state.in_degree(field_node) == 1

    field_node_path = ctx.state.memlet_path(next(iter(ctx.state.in_edges(field_node))))
    assert field_node_path[-1].dst is field_node

    if inner_map_output_temporary_removed:
        # The original output of the nested SDFG, the one that would be inside the Map,
        #  has been deleted and the nested SDFG writes directly into the output.
        #  In this case we have to adapt the stride of the array inside the nested SDFG.
        nsdfg_scan = field_node_path[0].src
        assert isinstance(nsdfg_scan, dace.nodes.NestedSDFG)
        inner_output_name = field_node_path[0].src_conn
        inner_output_desc = nsdfg_scan.sdfg.arrays[inner_output_name]
        assert len(inner_output_desc.shape) == 1

        outside_output_stride = field_desc.strides[scan_dim_index]
        assert str(outside_output_stride).isdigit()
        # The stride of the temporary array is constant, so we can just set it on the inside.
        inner_output_desc.set_shape(
            new_shape=inner_output_desc.shape, strides=(outside_output_stride,)
        )
    else:
        # The AccessNode on the inside of the Map was not removed but remains there.
        #  Thus we do not have to update the strides, we do however, make some checks.
        in_map_temporary_output_field = field_node_path[0].src
        assert in_map_temporary_output_field == output_edge.result.dc_node

        assert ctx.state.in_degree(in_map_temporary_output_field) == 1
        assert ctx.state.out_degree(in_map_temporary_output_field) == 1
        inner_edge = next(iter(ctx.state.in_edges(in_map_temporary_output_field)))
        nsdfg_scan = inner_edge.src
        assert isinstance(nsdfg_scan, dace.nodes.NestedSDFG)

        inner_output_name = inner_edge.src_conn
        inner_output_desc = nsdfg_scan.sdfg.arrays[inner_output_name]

        assert len(inner_output_desc.shape) == 1
        assert str(inner_output_desc.strides[0]).isdigit()
        assert inner_output_desc.shape == dataflow_output_desc.shape
        assert inner_output_desc.strides == dataflow_output_desc.strides

    return gtir_to_sdfg_types.FieldopData(
        field_node, ts.FieldType(field_dims, output_edge.result.gt_dtype), tuple(field_origin)
    )


def _create_scan_field_operator(
    ctx: gtir_to_sdfg.SubgraphContext,
    field_domain: gtir_domain.FieldopDomain,
    node_type: ts.FieldType | ts.TupleType,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    input_edges: Iterable[gtir_dataflow.DataflowInputEdge],
    output: MaybeNestedInTuple[gtir_dataflow.DataflowOutputEdge | None],
    output_domain: infer_domain.DomainAccess,
) -> gtir_to_sdfg_types.FieldopResult:
    """
    Helper method to build the output of a field operator, which can consist of
    a single field or a tuple of fields.

    Similar to `gtir_to_sdfg_primitives._create_field_operator()` but for scan
    field operators. The main difference is that the scan vertical dimension is
    excluded from the map range. This because the vertical dimension is traversed
    by a loop region in a mapped nested SDFG.

    Refer to `gtir_to_sdfg_primitives._create_field_operator()` for the
    description of function arguments and return values.
    """
    dims, _, _ = gtir_domain.get_field_layout(field_domain)

    # create a map scope to execute the `LoopRegion` over the horizontal domain
    if len(dims) == 1:
        # We construct the scan field operator on the horizontal domain, while the
        # vertical dimension (the column axis) is computed by the loop region.
        # If the field operator computes only the column axis (a 1d scan field operator),
        # there is no horizontal domain, therefore the map scope is not needed.
        # This case currently produces wrong CUDA code because of a DaCe issue
        # (see https://github.com/GridTools/gt4py/issues/1136).
        # The corresponding GT4Py tests are disabled (pytest marker `uses_scan_1d_field`).
        map_entry, map_exit = (None, None)
    else:
        # create map range corresponding to the field operator domain
        map_entry, map_exit = sdfg_builder.add_map(
            "fieldop",
            ctx.state,
            ndrange={
                gtir_to_sdfg_utils.get_map_variable(r.dim): f"{r.start}:{r.stop}"
                for r in field_domain
                if not sdfg_builder.is_column_axis(r.dim)
            },
        )
        assert (len(map_entry.params) + 1) == len(field_domain)

    # here we setup the edges passing through the map entry node
    for edge in input_edges:
        edge.connect(map_entry)

    if isinstance(node_type, ts.FieldType):
        assert isinstance(output, gtir_dataflow.DataflowOutputEdge)
        assert isinstance(output_domain, domain_utils.SymbolicDomain)
        return _create_scan_field_operator_impl(
            ctx, sdfg_builder, output, output_domain, node_type, map_exit
        )
    else:
        # Handle tuples of fields. note that the symbol name 'x' in the call below
        # is not used, we only need the tree structure of the `TupleType` definition
        # to pass to `tree_map()` in order to retrieve the type of each field.
        output_symbol_tree = gtir_to_sdfg_utils.make_symbol_tree("x", node_type)
        return gtx_utils.tree_map(
            lambda edge_, domain_, sym_, ctx_=ctx: (
                _create_scan_field_operator_impl(
                    ctx_,
                    sdfg_builder,
                    edge_,
                    domain_,
                    sym_.type,
                    map_exit,
                )
            )
        )(output, output_domain, output_symbol_tree)


def _scan_input_name(input_name: str) -> str:
    """
    Helper function to make naming of input connectors in the scan nested SDFG
    consistent throughut this module scope.
    """
    return f"__gtir_scan_input_{input_name}"


def _scan_output_name(input_name: str) -> str:
    """
    Same as above, but for the output connecters in the scan nested SDFG.
    """
    return f"__gtir_scan_output_{input_name}"


def _lower_lambda_to_nested_sdfg(
    lambda_node: gtir.Lambda,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    field_domain: gtir_domain.FieldopDomain,
    init_data: gtir_to_sdfg_types.FieldopResult,
    lambda_symbols: dict[str, ts.DataType],
    scan_forward: bool,
    scan_carry_symbol: gtir.Sym,
) -> tuple[gtir_to_sdfg.SubgraphContext, gtir_to_sdfg_types.FieldopResult]:
    """
    Helper method to lower the lambda node representing the scan stencil dataflow
    inside a separate SDFG.

    In regular field operators, where the computation of a grid point is independent
    from other points, therefore the stencil can be lowered to a mapped tasklet
    dataflow, and the map range is defined on the full domain.
    The scan field operator has to carry an intermediate result while the stencil
    is applied on vertical levels, which is input to the computation of next level
    (an accumulator function, for example). Therefore, the points on the vertical
    dimension are computed inside a `LoopRegion` construct.
    This function creates the `LoopRegion` inside a nested SDFG, which will be
    mapped by the caller to the horizontal domain in the field operator context.

    Args:
        lambda_node: The lambda representing the stencil expression on the horizontal level.
        ctx: The SDFG context where the scan field operator is translated.
        sdfg_builder: The SDFG builder object to access the field operator context.
        field_domain: The field operator domain, with all horizontal and vertical dimensions.
        init_data: The data produced in the field operator context that is used
            to initialize the scan carry value.
        lambda_symbols: List of symbols used as parameters of the stencil expressions.
        scan_forward: When True, the loop should range starting from the origin;
            when False, traverse towards origin.
        scan_carry_symbol: The symbol used in the stencil expression to carry the
            intermediate result along the vertical dimension.

    Returns:
        A tuple of two elements:
          - The subgraph context containing the `LoopRegion` along the vertical
            dimension, to be instantied as a nested SDFG in the field operator context.
          - The inner fields, that is 1d arrays with vertical shape containing
            the output of the stencil computation. These fields will have to be
            mapped to outer arrays by the caller. The caller is responsible to ensure
            that inner and outer arrays use the same strides.
    """

    # We pass an empty set as symbolic arguments, which implies that all scalar
    # inputs of the scan nested SDFG will be represented as scalar data containers.
    # The reason why we do not check for dace symbolic expressions and do not map
    # them to inner symbols is that the scan expression should not contain any domain
    # expression (no field operator inside).
    assert not any(
        eve.walk_values(lambda_node).map(
            lambda node: cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain"))
        )
    )
    # the lambda expression, i.e. body of the scan, will be created inside a nested SDFG.
    lambda_translator, lambda_ctx = sdfg_builder.setup_nested_context(
        lambda_node, "scan", ctx, lambda_symbols, symbolic_inputs=set()
    )

    # We set `using_explicit_control_flow=True` because the vertical scan is lowered to a `LoopRegion`.
    # This property is used by pattern matching in SDFG transformation framework
    # to skip those transformations that do not yet support control flow blocks.
    lambda_ctx.sdfg.using_explicit_control_flow = True

    # use the vertical dimension in the domain as scan dimension
    scan_domain = next(r for r in field_domain if sdfg_builder.is_column_axis(r.dim))

    # extract the scan loop range
    scan_loop_var = gtir_to_sdfg_utils.get_map_variable(scan_domain.dim)

    # in case the scan operator computes a list (not a scalar), we need to add an extra dimension
    def get_scan_output_shape(
        scan_init_data: gtir_to_sdfg_types.FieldopData,
    ) -> list[dace.symbolic.SymExpr]:
        scan_column_size = scan_domain.stop - scan_domain.start
        if isinstance(scan_init_data.gt_type, ts.ScalarType):
            return [scan_column_size]
        assert isinstance(scan_init_data.gt_type, ts.ListType)
        assert scan_init_data.gt_type.offset_type
        offset_type = scan_init_data.gt_type.offset_type
        offset_provider_type = sdfg_builder.get_offset_provider_type(offset_type.value)
        assert isinstance(offset_provider_type, gtx_common.NeighborConnectivityType)
        list_size = offset_provider_type.max_neighbors
        return [scan_column_size, dace.symbolic.SymExpr(list_size)]

    if isinstance(init_data, tuple):
        lambda_result_shape = gtx_utils.tree_map(get_scan_output_shape)(init_data)
    else:
        assert init_data is not None
        lambda_result_shape = get_scan_output_shape(init_data)

    # Create the body of the initialization state
    # This dataflow will write the initial value of the scan carry variable.
    init_state = lambda_ctx.state
    scan_carry_input = (
        gtir_to_sdfg_utils.make_symbol_tree(scan_carry_symbol.id, scan_carry_symbol.type)
        if isinstance(scan_carry_symbol.type, ts.TupleType)
        else scan_carry_symbol
    )

    def init_scan_carry(sym: gtir.Sym) -> None:
        scan_carry_dataname = str(sym.id)
        scan_carry_desc = lambda_ctx.sdfg.data(scan_carry_dataname)
        input_scan_carry_dataname = _scan_input_name(scan_carry_dataname)
        input_scan_carry_desc = scan_carry_desc.clone()
        lambda_ctx.sdfg.add_datadesc(input_scan_carry_dataname, input_scan_carry_desc)
        scan_carry_desc.transient = True
        init_state.add_nedge(
            init_state.add_access(input_scan_carry_dataname),
            init_state.add_access(scan_carry_dataname),
            lambda_ctx.sdfg.make_array_memlet(input_scan_carry_dataname),
        )

    if isinstance(scan_carry_input, tuple):
        gtx_utils.tree_map(init_scan_carry)(scan_carry_input)
    else:
        init_scan_carry(scan_carry_input)

    # Create a loop region over the vertical dimension corresponding to the scan column
    if scan_forward:
        scan_loop = dace.sdfg.state.LoopRegion(
            label="scan",
            condition_expr=f"{scan_loop_var} < {scan_domain.stop}",
            loop_var=scan_loop_var,
            initialize_expr=f"{scan_loop_var} = {scan_domain.start}",
            update_expr=f"{scan_loop_var} = {scan_loop_var} + 1",
            inverted=False,
        )
    else:
        scan_loop = dace.sdfg.state.LoopRegion(
            label="scan",
            condition_expr=f"{scan_loop_var} >= {scan_domain.start}",
            loop_var=scan_loop_var,
            initialize_expr=f"{scan_loop_var} = {scan_domain.stop} - 1",
            update_expr=f"{scan_loop_var} = {scan_loop_var} - 1",
            inverted=False,
        )
    lambda_ctx.sdfg.add_node(scan_loop, ensure_unique_name=True)
    lambda_ctx.sdfg.add_edge(init_state, scan_loop, dace.InterstateEdge())

    # Inside the loop region, create a 'compute' and an 'update' state.
    # The body of the 'compute' state implements the stencil expression for one vertical level.
    # The 'update' state writes the value computed by the stencil into the scan carry variable,
    # in order to make it available to the next vertical level.
    compute_state = scan_loop.add_state("scan_compute")
    compute_ctx = gtir_to_sdfg.SubgraphContext(lambda_ctx.sdfg, compute_state)
    update_state = scan_loop.add_state_after(compute_state, "scan_update")

    # inside the 'compute' state, visit the list of arguments to be passed to the stencil
    stencil_args = [
        _parse_scan_fieldop_arg(im.ref(p.id), compute_ctx, lambda_translator, field_domain)
        for p in lambda_node.params
    ]
    # stil inside the 'compute' state, generate the dataflow representing the stencil
    # to be applied on the horizontal domain
    lambda_input_edges, lambda_result = gtir_dataflow.translate_lambda_to_dataflow(
        compute_ctx.sdfg, compute_ctx.state, lambda_translator, lambda_node, stencil_args
    )
    # connect the dataflow input directly to the source data nodes, without passing through a map node;
    # the reason is that the map for horizontal domain is outside the scan loop region
    for edge in lambda_input_edges:
        edge.connect(map_entry=None)
    # connect the dataflow output nodes, called 'scan_result' below, to a global field called 'output'
    output_column_index = dace.symbolic.pystr_to_symbolic(scan_loop_var) - scan_domain.start

    def connect_scan_output(
        scan_output_edge: gtir_dataflow.DataflowOutputEdge,
        scan_output_shape: list[dace.symbolic.SymExpr],
        scan_carry_sym: gtir.Sym,
    ) -> gtir_to_sdfg_types.FieldopData:
        scan_result = scan_output_edge.result
        if isinstance(scan_result.gt_dtype, ts.ScalarType):
            assert scan_result.gt_dtype == scan_carry_sym.type
            # the scan field operator computes a column of scalar values
            assert len(scan_output_shape) == 1
            output_subset = dace_subsets.Range.from_string(str(output_column_index))
        else:
            raise NotImplementedError("scan with list output is not supported.")
        scan_result_data = scan_result.dc_node.data
        scan_result_desc = scan_result.dc_node.desc(lambda_ctx.sdfg)
        scan_result_subset = dace_subsets.Range.from_array(scan_result_desc)

        # `sym` represents the global output data, that is the nested-SDFG output connector
        scan_carry_data = str(scan_carry_sym.id)
        output = _scan_output_name(scan_carry_data)
        lambda_ctx.sdfg.add_array(output, scan_output_shape, scan_result_desc.dtype)
        output_node = compute_state.add_access(output)

        # in the 'compute' state, we write the current vertical level data to the output field
        # (the output field is mapped to an external array)
        compute_state.add_nedge(
            scan_result.dc_node,
            output_node,
            dace.Memlet(data=output, subset=output_subset, other_subset=scan_result_subset),
        )

        # in the 'update' state, the value of the current vertical level is written
        # to the scan carry variable for the next loop iteration
        update_state.add_nedge(
            update_state.add_access(scan_result_data),
            update_state.add_access(scan_carry_data),
            dace.Memlet(
                data=scan_result_data, subset=scan_result_subset, other_subset=scan_result_subset
            ),
        )

        output_type = ts.FieldType(dims=[scan_domain.dim], dtype=scan_result.gt_dtype)
        return gtir_to_sdfg_types.FieldopData(output_node, output_type, origin=(scan_domain.start,))

    # write the stencil result (value on one vertical level) into a 1D field
    # with full vertical shape representing one column
    if isinstance(scan_carry_input, tuple):
        assert isinstance(lambda_result_shape, tuple)
        lambda_output = gtx_utils.tree_map(connect_scan_output)(
            lambda_result, lambda_result_shape, scan_carry_input
        )
    else:
        assert isinstance(lambda_result[0], gtir_dataflow.DataflowOutputEdge)
        assert isinstance(lambda_result_shape, list)
        lambda_output = connect_scan_output(lambda_result[0], lambda_result_shape, scan_carry_input)

    # in case tuples are passed as argument, isolated access nodes might be left in the state,
    # because not all tuple fields are necessarily accessed inside the lambda scope
    for data_node in compute_state.data_nodes():
        if compute_state.degree(data_node) == 0:
            # By construction there should never be isolated transient nodes.
            # Therefore, the assert below implements a sanity check, that allows
            # the exceptional case (encountered in one GT4Py test) where the carry
            # variable is not used, so not a scan indeed because no data dependency.
            assert (not data_node.desc(lambda_ctx.sdfg).transient) or data_node.data.startswith(
                scan_carry_symbol.id
            )
            compute_state.remove_node(data_node)

    return lambda_ctx, lambda_output


def _connect_nested_sdfg_output_to_temporaries(
    inner_ctx: gtir_to_sdfg.SubgraphContext,
    outer_ctx: gtir_to_sdfg.SubgraphContext,
    nsdfg_node: dace.nodes.NestedSDFG,
    inner_data: gtir_to_sdfg_types.FieldopData,
) -> gtir_dataflow.DataflowOutputEdge:
    """
    Helper function to create the edges to write output data from the nested SDFG
    to temporary arrays in the parent SDFG, denoted as outer context.

    Args:
        inner_ctx: The inner SDFG context, where the scan `LoopRegion` is translated.
        outer_ctx: The outer SDFG context, where the field operator is translated.
        nsdfg_node: The nested SDFG node in the outer context.
        inner_data: The data produced by the scan `LoopRegion` in the inner context.

    Returns:
        An object representing the output data connection of this field operator.
    """
    assert isinstance(inner_data.gt_type, ts.FieldType)
    inner_dataname = inner_data.dc_node.data
    inner_desc = inner_ctx.sdfg.data(inner_dataname)
    outer_dataname, outer_desc = outer_ctx.sdfg.add_temp_transient_like(inner_desc)
    outer_node = outer_ctx.state.add_access(outer_dataname)
    outer_ctx.state.add_edge(
        nsdfg_node,
        inner_dataname,
        outer_node,
        None,
        dace.Memlet.from_array(outer_dataname, outer_desc),
    )
    output_expr = gtir_dataflow.ValueExpr(outer_node, inner_data.gt_type.dtype)
    return gtir_dataflow.DataflowOutputEdge(outer_ctx.state, output_expr)


def _remove_nested_sdfg_connector(
    inner_ctx: gtir_to_sdfg.SubgraphContext,
    nsdfg_node: dace.nodes.NestedSDFG,
    inner_data: gtir_to_sdfg_types.FieldopData,
) -> None:
    inner_data.dc_node.desc(inner_ctx.sdfg).transient = True
    nsdfg_node.out_connectors.pop(inner_data.dc_node.data)


def translate_scan(
    node: gtir.Node,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopResult:
    """
    Generates the dataflow subgraph for the `as_fieldop` builtin with a scan operator.

    It differs from `translate_as_fieldop()` in that the horizontal domain is lowered
    to a map scope, while the scan column computation is lowered to a `LoopRegion`
    on the vertical dimension, that is inside the horizontal map.
    The current design choice is to keep the map scope on the outer level, and
    the `LoopRegion` inside. This choice follows the GTIR representation where
    the `scan` operator is called inside the `as_fieldop` node.

    Implements the `PrimitiveTranslator` protocol.
    """
    assert isinstance(node, gtir.FunCall)
    assert cpm.is_call_to(node.fun, "as_fieldop")
    assert isinstance(node.type, (ts.FieldType, ts.TupleType))

    fun_node = node.fun
    assert len(fun_node.args) == 2
    scan_expr, scan_domain_expr = fun_node.args
    assert cpm.is_call_to(scan_expr, "scan")

    # parse the domain of the scan field operator
    assert isinstance(scan_domain_expr.type, ts.DomainType)
    field_domain = gtir_domain.get_field_domain(
        domain_utils.SymbolicDomain.from_expr(scan_domain_expr)
    )

    # parse scan parameters
    assert len(scan_expr.args) == 3
    stencil_expr = scan_expr.args[0]
    assert isinstance(stencil_expr, gtir.Lambda)

    # params[0]: the lambda parameter to propagate the scan carry on the vertical dimension
    scan_carry = stencil_expr.params[0].id
    scan_carry_type = stencil_expr.params[0].type
    assert isinstance(scan_carry_type, ts.DataType)

    # params[1]: boolean flag for forward/backward scan
    assert isinstance(scan_expr.args[1], gtir.Literal) and ti.is_logical(scan_expr.args[1].type)
    scan_forward = scan_expr.args[1].value == "True"

    # params[2]: the expression that computes the value for scan initialization
    init_expr = scan_expr.args[2]
    # visit the initialization value of the scan expression
    init_data = sdfg_builder.visit(init_expr, ctx=ctx)

    # define the set of symbols available in the lambda context, which consists of
    # the carry argument and all lambda function arguments
    lambda_arg_types: list[ts.DataType] = [scan_carry_type] + [
        arg.type for arg in node.args if isinstance(arg.type, ts.DataType)
    ]
    lambda_symbols: dict[str, ts.DataType] = {
        str(p.id): arg_type
        for p, arg_type in zip(stencil_expr.params, lambda_arg_types, strict=True)
    }

    # lower the scan stencil expression in a separate SDFG context
    lambda_ctx, lambda_output = _lower_lambda_to_nested_sdfg(
        stencil_expr,
        ctx,
        sdfg_builder,
        field_domain,
        init_data,
        lambda_symbols,
        scan_forward,
        im.sym(scan_carry, scan_carry_type),
    )

    # visit the arguments to be passed to the lambda expression
    # this must be executed before visiting the lambda expression, in order to populate
    # the data descriptor with the correct field domain offsets for field arguments
    lambda_args = [sdfg_builder.visit(arg, ctx=ctx) for arg in node.args]
    lambda_args_mapping = [
        (im.sym(_scan_input_name(scan_carry), scan_carry_type), init_data),
    ] + [(param, arg) for param, arg in zip(stencil_expr.params[1:], lambda_args, strict=True)]

    lambda_arg_nodes = dict(
        itertools.chain(
            *[gtir_to_sdfg_types.flatten_tuple(psym, arg) for psym, arg in lambda_args_mapping]
        )
    )

    # parse the dataflow output symbols
    if isinstance(scan_carry_type, ts.TupleType):
        lambda_flat_outs = {
            str(sym.id): sym.type
            for sym in gtir_to_sdfg_utils.flatten_tuple_fields(
                _scan_output_name(scan_carry), scan_carry_type
            )
        }
    else:
        lambda_flat_outs = {_scan_output_name(scan_carry): scan_carry_type}

    # build the mapping of symbols from nested SDFG to field operator context
    nsdfg_symbols_mapping = {str(sym): sym for sym in lambda_ctx.sdfg.free_symbols}
    for psym, arg in lambda_args_mapping:
        nsdfg_symbols_mapping |= gtir_to_sdfg_utils.get_arg_symbol_mapping(psym.id, arg, ctx.sdfg)

    # the scan nested SDFG is ready: it is instantiated in the field operator context
    # where the map scope over the horizontal domain lives
    nsdfg_node = ctx.state.add_nested_sdfg(
        lambda_ctx.sdfg,
        inputs=set(lambda_arg_nodes.keys()),
        outputs=set(lambda_flat_outs.keys()),
        symbol_mapping=nsdfg_symbols_mapping,
    )

    input_edges = []
    for input_connector, outer_arg in lambda_arg_nodes.items():
        assert not lambda_ctx.sdfg.arrays[input_connector].transient
        if outer_arg is None:
            # This argument has empty domain, which means that it should not be
            # used inside the nested SDFG, and does not need to be connected outside.
            assert all(
                node.data != input_connector
                for node in lambda_ctx.sdfg.all_nodes_recursive()
                if isinstance(node, dace.nodes.AccessNode)
            )
            lambda_ctx.sdfg.arrays[input_connector].transient = True
        else:
            arg_desc = outer_arg.dc_node.desc(ctx.sdfg)
            input_subset = dace_subsets.Range.from_array(arg_desc)
            input_edge = gtir_dataflow.MemletInputEdge(
                ctx.state, outer_arg.dc_node, input_subset, nsdfg_node, input_connector
            )
            input_edges.append(input_edge)

    # for output connections, we create temporary arrays that contain the computation
    # results of a column slice for each point in the horizontal domain
    output_tree = gtx_utils.tree_map(
        lambda output_data, output_domain: _connect_nested_sdfg_output_to_temporaries(
            lambda_ctx, ctx, nsdfg_node, output_data
        )
        if output_domain != infer_domain.DomainAccessDescriptor.NEVER
        else _remove_nested_sdfg_connector(lambda_ctx, nsdfg_node, output_data)
    )(lambda_output, node.annex.domain)

    # we call a helper method to create a map scope that will compute the entire field
    return _create_scan_field_operator(
        ctx, field_domain, node.type, sdfg_builder, input_edges, output_tree, node.annex.domain
    )
