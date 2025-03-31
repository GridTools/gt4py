# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the lowering of scan field operator.

This builtin translator implements the `PrimitiveTranslator` protocol as other
translators in `gtir_builtin_translators` module. This module implements the scan
translator, separately from the `gtir_builtin_translators` module, because the
parsing of input arguments as well as the construction of the map scope differ
from a regular field operator, which requires slightly different helper methods.
Besides, the function code is quite large, another reason to keep it separate
from other translators.

The current GTIR representation of the scan operator is based on iterator view.
This is likely to change in the future, to enable GTIR optimizations for scan.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Iterable

import dace
from dace import subsets as dace_subsets

from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.program_processors.runners.dace import (
    gtir_builtin_translators as gtir_translators,
    gtir_dataflow,
    gtir_sdfg,
    gtir_sdfg_utils,
)
from gt4py.next.type_system import type_info as ti, type_specifications as ts


if TYPE_CHECKING:
    from gt4py.next.program_processors.runners.dace import gtir_sdfg


def _parse_scan_fieldop_arg(
    node: gtir.Expr,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
    domain: gtir_translators.FieldopDomain,
) -> gtir_dataflow.MemletExpr | tuple[gtir_dataflow.MemletExpr | tuple[Any, ...], ...]:
    """Helper method to visit an expression passed as argument to a scan field operator.

    On the innermost level, a scan operator is lowered to a loop region which computes
    column elements in the vertical dimension.

    It differs from the helper method `gtir_builtin_translators` in that field arguments
    are passed in full shape along the vertical dimension, rather than as iterator.
    """

    def _parse_fieldop_arg_impl(
        arg: gtir_translators.FieldopData,
    ) -> gtir_dataflow.MemletExpr:
        arg_expr = arg.get_local_view(domain, sdfg)
        if isinstance(arg_expr, gtir_dataflow.MemletExpr):
            return arg_expr
        # In scan field operator, the arguments to the vertical stencil are passed by value.
        # Therefore, the full field shape is passed as `MemletExpr` rather than `IteratorExpr`.
        return gtir_dataflow.MemletExpr(
            arg_expr.field, arg_expr.gt_dtype, arg_expr.get_memlet_subset(sdfg)
        )

    arg = sdfg_builder.visit(node, sdfg=sdfg, head_state=state)

    if isinstance(arg, gtir_translators.FieldopData):
        return _parse_fieldop_arg_impl(arg)
    else:
        # handle tuples of fields
        return gtx_utils.tree_map(lambda x: _parse_fieldop_arg_impl(x))(arg)


def _create_scan_field_operator_impl(
    sdfg_builder: gtir_sdfg.SDFGBuilder,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    domain: gtir_translators.FieldopDomain,
    output_edge: gtir_dataflow.DataflowOutputEdge,
    output_type: ts.FieldType,
    map_exit: dace.nodes.MapExit,
) -> gtir_translators.FieldopData:
    """
    Helper method to allocate a temporary array that stores one field computed
    by the scan field operator.

    This method is called by `_create_scan_field_operator()`.

    Similar to `gtir_builtin_translators._create_field_operator_impl()` but
    for scan field operators. It differs in that the scan loop region produces
    a field along the vertical dimension, rather than a single point.
    Therefore, the memlet subset will write a slice into the result array, that
    corresponds to the full vertical shape for each horizontal grid point.

    Refer to `gtir_builtin_translators._create_field_operator_impl()` for
    the description of function arguments and return values.
    """
    dataflow_output_desc = output_edge.result.dc_node.desc(sdfg)
    assert isinstance(dataflow_output_desc, dace.data.Array)

    # the memory layout of the output field follows the field operator compute domain
    field_dims, field_origin, field_shape = gtir_translators.get_field_layout(domain)
    field_indices = gtir_translators.get_domain_indices(field_dims, field_origin)
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

    if isinstance(output_edge.result.gt_dtype, ts.ScalarType):
        assert isinstance(output_type.dtype, ts.ScalarType)
        if output_edge.result.gt_dtype != output_type.dtype:
            raise TypeError(
                f"Type mismatch, expected {output_type.dtype} got {output_edge.result.gt_dtype}."
            )
        field_dtype = output_edge.result.gt_dtype
        # the scan field operator computes a column of scalar values
        assert len(dataflow_output_desc.shape) == 1
    else:
        assert isinstance(output_type.dtype, ts.ListType)
        assert isinstance(output_edge.result.gt_dtype.element_type, ts.ScalarType)
        field_dtype = output_edge.result.gt_dtype.element_type
        if field_dtype != output_type.dtype.element_type:
            raise TypeError(
                f"Type mismatch, expected {output_type.dtype.element_type} got {field_dtype}."
            )
        # the scan field operator computes a list of scalar values for each column level
        # 1st dim: column level, 2nd dim: list of scalar values (e.g. `neighbors`)
        assert len(dataflow_output_desc.shape) == 2
        # the lines below extend the array with the local dimension added by the field operator
        assert output_edge.result.gt_dtype.offset_type is not None
        field_shape = [*field_shape, dataflow_output_desc.shape[1]]
        field_subset = field_subset + dace_subsets.Range.from_string(
            f"0:{dataflow_output_desc.shape[1]}"
        )

    # allocate local temporary storage
    field_name, field_desc = sdfg_builder.add_temp_array(
        sdfg, field_shape, dataflow_output_desc.dtype
    )
    # the inner and outer strides have to match
    scan_output_stride = field_desc.strides[scan_dim_index]
    # also consider the stride of the local dimension, in case the scan field operator computes a list
    local_strides = field_desc.strides[len(field_dims) :]
    assert len(local_strides) == (1 if isinstance(output_edge.result.gt_dtype, ts.ListType) else 0)
    new_inner_strides = [scan_output_stride, *local_strides]
    dataflow_output_desc.set_shape(dataflow_output_desc.shape, new_inner_strides)

    # and here the edge writing the dataflow result data through the map exit node
    field_node = state.add_access(field_name)
    output_edge.connect(map_exit, field_node, field_subset)

    return gtir_translators.FieldopData(
        field_node, ts.FieldType(field_dims, output_edge.result.gt_dtype), tuple(field_origin)
    )


def _create_scan_field_operator(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    domain: gtir_translators.FieldopDomain,
    node_type: ts.FieldType | ts.TupleType,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
    input_edges: Iterable[gtir_dataflow.DataflowInputEdge],
    output_tree: gtir_dataflow.DataflowOutputEdge
    | tuple[gtir_dataflow.DataflowOutputEdge | tuple[Any, ...], ...],
) -> gtir_translators.FieldopResult:
    """
    Helper method to build the output of a field operator, which can consist of
    a single field or a tuple of fields.

    Similar to `gtir_builtin_translators._create_field_operator()` but for scan
    field operators. The main difference is that the scan vertical dimension is
    excluded from the map range. This because the vertical dimension is traversed
    by a loop region in a mapped nested SDFG.

    Refer to `gtir_builtin_translators._create_field_operator()` for the
    description of function arguments and return values.
    """
    domain_dims, _, _ = gtir_translators.get_field_layout(domain)

    # create a map scope to execute the `LoopRegion` over the horizontal domain
    if len(domain_dims) == 1:
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
            state,
            ndrange={
                gtir_sdfg_utils.get_map_variable(dim): f"{lower_bound}:{upper_bound}"
                for dim, lower_bound, upper_bound in domain
                if not sdfg_builder.is_column_axis(dim)
            },
        )

    # here we setup the edges passing through the map entry node
    for edge in input_edges:
        edge.connect(map_entry)

    if isinstance(node_type, ts.FieldType):
        assert isinstance(output_tree, gtir_dataflow.DataflowOutputEdge)
        return _create_scan_field_operator_impl(
            sdfg_builder, sdfg, state, domain, output_tree, node_type, map_exit
        )
    else:
        # handle tuples of fields
        # the symbol name 'x' in the call below is not used, we only need
        # the tree structure of the `TupleType` definition to pass to `tree_map()`
        output_symbol_tree = gtir_sdfg_utils.make_symbol_tree("x", node_type)
        return gtx_utils.tree_map(
            lambda output_edge, output_sym: (
                _create_scan_field_operator_impl(
                    sdfg_builder,
                    sdfg,
                    state,
                    domain,
                    output_edge,
                    output_sym.type,
                    map_exit,
                )
            )
        )(output_tree, output_symbol_tree)


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
    sdfg: dace.SDFG,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
    domain: gtir_translators.FieldopDomain,
    init_data: gtir_translators.FieldopResult,
    lambda_symbols: dict[str, ts.DataType],
    scan_forward: bool,
    scan_carry_symbol: gtir.Sym,
) -> tuple[dace.SDFG, gtir_translators.FieldopResult]:
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
        sdfg: The SDFG where the scan field operator is translated.
        sdfg_builder: The SDFG builder object to access the field operator context.
        domain: The field operator domain, with all horizontal and vertical dimensions.
        init_data: The data produced in the field operator context that is used
            to initialize the scan carry value.
        lambda_symbols: List of symbols used as parameters of the stencil expressions.
        scan_forward: When True, the loop should range starting from the origin;
            when False, traverse towards origin.
        scan_carry_symbol: The symbol used in the stencil expression to carry the
            intermediate result along the vertical dimension.

    Returns:
        A tuple of two elements:
          - An SDFG containing the `LoopRegion` computation along the vertical
            dimension, to be instantied as a nested SDFG in the field operator context.
          - The inner fields, that is 1d arrays with vertical shape containing
            the output of the stencil computation. These fields will have to be
            mapped to outer arrays by the caller. The caller is responsible to ensure
            that inner and outer arrays use the same strides.
    """

    # the lambda expression, i.e. body of the scan, will be created inside a nested SDFG.
    nsdfg = dace.SDFG(sdfg_builder.unique_nsdfg_name(sdfg, "scan"))
    nsdfg.debuginfo = gtir_sdfg_utils.debug_info(lambda_node, default=sdfg.debuginfo)
    # We set `using_explicit_control_flow=True` because the vertical scan is lowered to a `LoopRegion`.
    # This property is used by pattern matching in SDFG transformation framework
    # to skip those transformations that do not yet support control flow blocks.
    nsdfg.using_explicit_control_flow = True
    lambda_translator = sdfg_builder.setup_nested_context(nsdfg, sdfg, lambda_symbols)

    # use the vertical dimension in the domain as scan dimension
    scan_domain = [
        (dim, lower_bound, upper_bound)
        for dim, lower_bound, upper_bound in domain
        if sdfg_builder.is_column_axis(dim)
    ]
    assert len(scan_domain) == 1
    scan_dim, scan_lower_bound, scan_upper_bound = scan_domain[0]

    # extract the scan loop range
    scan_loop_var = gtir_sdfg_utils.get_map_variable(scan_dim)

    # in case the scan operator computes a list (not a scalar), we need to add an extra dimension
    def get_scan_output_shape(
        scan_init_data: gtir_translators.FieldopData,
    ) -> list[dace.symbolic.SymExpr]:
        scan_column_size = scan_upper_bound - scan_lower_bound
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
        lambda_result_shape = get_scan_output_shape(init_data)

    # Create the body of the initialization state
    # This dataflow will write the initial value of the scan carry variable.
    init_state = nsdfg.add_state("scan_init", is_start_block=True)
    scan_carry_input = (
        gtir_sdfg_utils.make_symbol_tree(scan_carry_symbol.id, scan_carry_symbol.type)
        if isinstance(scan_carry_symbol.type, ts.TupleType)
        else scan_carry_symbol
    )

    def init_scan_carry(sym: gtir.Sym) -> None:
        scan_carry_dataname = str(sym.id)
        scan_carry_desc = nsdfg.data(scan_carry_dataname)
        input_scan_carry_dataname = _scan_input_name(scan_carry_dataname)
        input_scan_carry_desc = scan_carry_desc.clone()
        nsdfg.add_datadesc(input_scan_carry_dataname, input_scan_carry_desc)
        scan_carry_desc.transient = True
        init_state.add_nedge(
            init_state.add_access(input_scan_carry_dataname),
            init_state.add_access(scan_carry_dataname),
            nsdfg.make_array_memlet(input_scan_carry_dataname),
        )

    if isinstance(scan_carry_input, tuple):
        gtx_utils.tree_map(init_scan_carry)(scan_carry_input)
    else:
        init_scan_carry(scan_carry_input)

    # Create a loop region over the vertical dimension corresponding to the scan column
    if scan_forward:
        scan_loop = dace.sdfg.state.LoopRegion(
            label="scan",
            condition_expr=f"{scan_loop_var} < {scan_upper_bound}",
            loop_var=scan_loop_var,
            initialize_expr=f"{scan_loop_var} = {scan_lower_bound}",
            update_expr=f"{scan_loop_var} = {scan_loop_var} + 1",
            inverted=False,
        )
    else:
        scan_loop = dace.sdfg.state.LoopRegion(
            label="scan",
            condition_expr=f"{scan_loop_var} >= {scan_lower_bound}",
            loop_var=scan_loop_var,
            initialize_expr=f"{scan_loop_var} = {scan_upper_bound} - 1",
            update_expr=f"{scan_loop_var} = {scan_loop_var} - 1",
            inverted=False,
        )
    nsdfg.add_node(scan_loop)
    nsdfg.add_edge(init_state, scan_loop, dace.InterstateEdge())

    # Inside the loop region, create a 'compute' and an 'update' state.
    # The body of the 'compute' state implements the stencil expression for one vertical level.
    # The 'update' state writes the value computed by the stencil into the scan carry variable,
    # in order to make it available to the next vertical level.
    compute_state = scan_loop.add_state("scan_compute")
    update_state = scan_loop.add_state_after(compute_state, "scan_update")

    # inside the 'compute' state, visit the list of arguments to be passed to the stencil
    stencil_args = [
        _parse_scan_fieldop_arg(im.ref(p.id), nsdfg, compute_state, lambda_translator, domain)
        for p in lambda_node.params
    ]
    # stil inside the 'compute' state, generate the dataflow representing the stencil
    # to be applied on the horizontal domain
    lambda_input_edges, lambda_result = gtir_dataflow.translate_lambda_to_dataflow(
        nsdfg, compute_state, lambda_translator, lambda_node, stencil_args
    )
    # connect the dataflow input directly to the source data nodes, without passing through a map node;
    # the reason is that the map for horizontal domain is outside the scan loop region
    for edge in lambda_input_edges:
        edge.connect(map_entry=None)
    # connect the dataflow output nodes, called 'scan_result' below, to a global field called 'output'
    output_column_index = dace.symbolic.pystr_to_symbolic(scan_loop_var) - scan_lower_bound

    def connect_scan_output(
        scan_output_edge: gtir_dataflow.DataflowOutputEdge,
        scan_output_shape: list[dace.symbolic.SymExpr],
        scan_carry_sym: gtir.Sym,
    ) -> gtir_translators.FieldopData:
        scan_result = scan_output_edge.result
        if isinstance(scan_result.gt_dtype, ts.ScalarType):
            assert scan_result.gt_dtype == scan_carry_sym.type
            # the scan field operator computes a column of scalar values
            assert len(scan_output_shape) == 1
            output_subset = dace_subsets.Range.from_string(str(output_column_index))
        else:
            assert isinstance(scan_carry_sym.type, ts.ListType)
            assert scan_result.gt_dtype.element_type == scan_carry_sym.type.element_type
            # the scan field operator computes a list of scalar values for each column level
            assert len(scan_output_shape) == 2
            output_subset = dace_subsets.Range.from_string(
                f"{output_column_index}, 0:{scan_output_shape[1]}"
            )
        scan_result_data = scan_result.dc_node.data
        scan_result_desc = scan_result.dc_node.desc(nsdfg)

        # `sym` represents the global output data, that is the nested-SDFG output connector
        scan_carry_data = str(scan_carry_sym.id)
        output = _scan_output_name(scan_carry_data)
        nsdfg.add_array(output, scan_output_shape, scan_result_desc.dtype)
        output_node = compute_state.add_access(output)

        # in the 'compute' state, we write the current vertical level data to the output field
        # (the output field is mapped to an external array)
        compute_state.add_nedge(
            scan_result.dc_node, output_node, dace.Memlet(data=output, subset=output_subset)
        )

        # in the 'update' state, the value of the current vertical level is written
        # to the scan carry variable for the next loop iteration
        update_state.add_nedge(
            update_state.add_access(scan_result_data),
            update_state.add_access(scan_carry_data),
            dace.Memlet.from_array(scan_result_data, scan_result_desc),
        )

        output_type = ts.FieldType(dims=[scan_dim], dtype=scan_result.gt_dtype)
        return gtir_translators.FieldopData(output_node, output_type, origin=(scan_lower_bound,))

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
        data_desc = data_node.desc(nsdfg)
        if compute_state.degree(data_node) == 0:
            # By construction there should never be isolated transient nodes.
            # Therefore, the assert below implements a sanity check, that allows
            # the exceptional case (encountered in one GT4Py test) where the carry
            # variable is not used, so not a scan indeed because no data dependency.
            assert (not data_desc.transient) or data_node.data.startswith(scan_carry_symbol.id)
            compute_state.remove_node(data_node)

    return nsdfg, lambda_output


def _connect_nested_sdfg_output_to_temporaries(
    sdfg: dace.SDFG,
    nsdfg: dace.SDFG,
    nsdfg_node: dace.nodes.NestedSDFG,
    outer_state: dace.SDFGState,
    inner_data: gtir_translators.FieldopData,
) -> gtir_dataflow.DataflowOutputEdge:
    """
    Helper function to create the edges to write output data from the nested SDFG
    to temporary arrays in the parent SDFG, denoted as outer context.

    Args:
        sdfg: The SDFG representing the outer context, where the field operator is translated.
        nsdfg: The SDFG where the scan `LoopRegion` is translated.
        nsdfg_node: The nested SDFG node in the outer context.
        outer_state: The state in outer context where the field operator is translated.
        inner_data: The data produced by the scan `LoopRegion` in the inner context.

    Returns:
        An object representing the output data connection of this field operator.
    """
    assert isinstance(inner_data.gt_type, ts.FieldType)
    inner_dataname = inner_data.dc_node.data
    inner_desc = nsdfg.data(inner_dataname)
    outer_dataname, outer_desc = sdfg.add_temp_transient_like(inner_desc)
    outer_node = outer_state.add_access(outer_dataname)
    outer_state.add_edge(
        nsdfg_node,
        inner_dataname,
        outer_node,
        None,
        dace.Memlet.from_array(outer_dataname, outer_desc),
    )
    output_expr = gtir_dataflow.ValueExpr(outer_node, inner_data.gt_type.dtype)
    return gtir_dataflow.DataflowOutputEdge(outer_state, output_expr)


def translate_scan(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
) -> gtir_translators.FieldopResult:
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
    scan_expr, domain_expr = fun_node.args
    assert cpm.is_call_to(scan_expr, "scan")

    # parse the domain of the scan field operator
    domain = gtir_translators.extract_domain(domain_expr)

    # parse scan parameters
    assert len(scan_expr.args) == 3
    stencil_expr = scan_expr.args[0]
    assert isinstance(stencil_expr, gtir.Lambda)

    # params[0]: the lambda parameter to propagate the scan carry on the vertical dimension
    scan_carry = str(stencil_expr.params[0].id)

    # params[1]: boolean flag for forward/backward scan
    assert isinstance(scan_expr.args[1], gtir.Literal) and ti.is_logical(scan_expr.args[1].type)
    scan_forward = scan_expr.args[1].value == "True"

    # params[2]: the expression that computes the value for scan initialization
    init_expr = scan_expr.args[2]
    # visit the initialization value of the scan expression
    init_data = sdfg_builder.visit(init_expr, sdfg=sdfg, head_state=state)
    # extract type definition of the scan carry
    scan_carry_type = (
        init_data.gt_type
        if isinstance(init_data, gtir_translators.FieldopData)
        else gtir_translators.get_tuple_type(init_data)
    )

    # define the set of symbols available in the lambda context, which consists of
    # the carry argument and all lambda function arguments
    lambda_arg_types = [scan_carry_type] + [
        arg.type for arg in node.args if isinstance(arg.type, ts.DataType)
    ]
    lambda_symbols = {
        str(p.id): arg_type
        for p, arg_type in zip(stencil_expr.params, lambda_arg_types, strict=True)
    }

    # lower the scan stencil expression in a separate SDFG context
    nsdfg, lambda_output = _lower_lambda_to_nested_sdfg(
        stencil_expr,
        sdfg,
        sdfg_builder,
        domain,
        init_data,
        lambda_symbols,
        scan_forward,
        im.sym(scan_carry, scan_carry_type),
    )

    # visit the arguments to be passed to the lambda expression
    # this must be executed before visiting the lambda expression, in order to populate
    # the data descriptor with the correct field domain offsets for field arguments
    lambda_args = [sdfg_builder.visit(arg, sdfg=sdfg, head_state=state) for arg in node.args]
    lambda_args_mapping = [
        (im.sym(_scan_input_name(scan_carry), scan_carry_type), init_data),
    ] + [
        (im.sym(param.id, arg.gt_type), arg)
        for param, arg in zip(stencil_expr.params[1:], lambda_args, strict=True)
    ]

    lambda_arg_nodes = dict(
        itertools.chain(
            *[gtir_translators.flatten_tuples(psym.id, arg) for psym, arg in lambda_args_mapping]
        )
    )

    # parse the dataflow output symbols
    if isinstance(scan_carry_type, ts.TupleType):
        lambda_flat_outs = {
            str(sym.id): sym.type
            for sym in gtir_sdfg_utils.flatten_tuple_fields(
                _scan_output_name(scan_carry), scan_carry_type
            )
        }
    else:
        lambda_flat_outs = {_scan_output_name(scan_carry): scan_carry_type}

    # build the mapping of symbols from nested SDFG to field operator context
    nsdfg_symbols_mapping = {str(sym): sym for sym in nsdfg.free_symbols}
    for psym, arg in lambda_args_mapping:
        nsdfg_symbols_mapping |= gtir_translators.get_arg_symbol_mapping(psym.id, arg, sdfg)

    # the scan nested SDFG is ready: it is instantiated in the field operator context
    # where the map scope over the horizontal domain lives
    nsdfg_node = state.add_nested_sdfg(
        nsdfg,
        sdfg,
        inputs=set(lambda_arg_nodes.keys()),
        outputs=set(lambda_flat_outs.keys()),
        symbol_mapping=nsdfg_symbols_mapping,
    )

    lambda_input_edges = []
    for input_connector, outer_arg in lambda_arg_nodes.items():
        arg_desc = outer_arg.dc_node.desc(sdfg)
        input_subset = dace_subsets.Range.from_array(arg_desc)
        input_edge = gtir_dataflow.MemletInputEdge(
            state, outer_arg.dc_node, input_subset, nsdfg_node, input_connector
        )
        lambda_input_edges.append(input_edge)

    # for output connections, we create temporary arrays that contain the computation
    # results of a column slice for each point in the horizontal domain
    lambda_output_tree = gtx_utils.tree_map(
        lambda lambda_output_data: _connect_nested_sdfg_output_to_temporaries(
            sdfg, nsdfg, nsdfg_node, state, lambda_output_data
        )
    )(lambda_output)

    # we call a helper method to create a map scope that will compute the entire field
    return _create_scan_field_operator(
        sdfg, state, domain, node.type, sdfg_builder, lambda_input_edges, lambda_output_tree
    )
