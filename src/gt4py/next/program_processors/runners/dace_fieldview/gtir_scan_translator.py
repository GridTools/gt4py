# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional

import dace
from dace import subsets as dace_subsets

from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.program_processors.runners.dace_common import utility as dace_utils
from gt4py.next.program_processors.runners.dace_fieldview import (
    gtir_builtin_translators as gtir_translators,
    gtir_dataflow,
    gtir_sdfg,
    utility as dace_gtir_utils,
)
from gt4py.next.type_system import type_info as ti, type_specifications as ts


if TYPE_CHECKING:
    from gt4py.next.program_processors.runners.dace_fieldview import gtir_sdfg


def _parse_fieldop_arg(
    node: gtir.Expr,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
    domain: gtir_translators.FieldopDomain,
) -> gtir_dataflow.MemletExpr | tuple[gtir_dataflow.MemletExpr | tuple[Any, ...], ...]:
    """Helper method to visit an expression passed as argument to a field operator."""

    def _parse_fieldop_arg_impl(
        arg: gtir_translators.FieldopData,
    ) -> gtir_dataflow.MemletExpr:
        arg_expr = arg.get_local_view(domain)
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


def _create_field_operator_impl(
    sdfg_builder: gtir_sdfg.SDFGBuilder,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    domain: gtir_translators.FieldopDomain,
    output_edge: gtir_dataflow.DataflowOutputEdge,
    output_type: ts.FieldType,
    map_exit: dace.nodes.MapExit,
    scan_dim: gtx_common.Dimension,
) -> gtir_translators.FieldopData:
    """
    Similar to `_create_scan_field_operator_impl()` but for scan field operators.
    """
    dataflow_output_desc = output_edge.result.dc_node.desc(sdfg)
    assert isinstance(dataflow_output_desc, dace.data.Array)

    domain_dims, domain_offset, domain_shape = gtir_translators.get_field_layout(domain)
    domain_indices = gtir_translators.get_domain_indices(domain_dims, domain_offset)
    domain_subset = dace_subsets.Range.from_indices(domain_indices)

    # the vertical dimension should not belong to the field operator domain
    # but we need to write it to the output field
    scan_dim_index = domain_dims.index(scan_dim)

    field_subset = (
        dace_subsets.Range(domain_subset[:scan_dim_index])
        + dace_subsets.Range.from_string(f"0:{dataflow_output_desc.shape[0]}")
        + dace_subsets.Range(domain_subset[scan_dim_index + 1 :])
    )

    if isinstance(output_edge.result.gt_dtype, ts.ScalarType):
        assert isinstance(output_type.dtype, ts.ScalarType)
        if output_edge.result.gt_dtype != output_type.dtype:
            raise TypeError(
                f"Type mismatch, expected {output_type.dtype} got {output_edge.result.gt_dtype}."
            )
        field_dtype = output_edge.result.gt_dtype
        field_dims, field_shape, field_offset = (domain_dims, domain_shape, domain_offset)
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
        assert len(dataflow_output_desc.shape) == 2
        # extend the array with the local dimensions added by the field operator (e.g. `neighbors`)
        assert output_edge.result.gt_dtype.offset_type is not None
        field_dims = [*domain_dims, output_edge.result.gt_dtype.offset_type]
        field_shape = [*domain_shape, dataflow_output_desc.shape[1]]
        field_offset = [*domain_offset, dataflow_output_desc.offset[1]]
        field_subset = field_subset + dace_subsets.Range.from_string(
            f"0:{dataflow_output_desc.shape[1]}"
        )

    # allocate local temporary storage
    assert dataflow_output_desc.dtype == dace_utils.as_dace_type(field_dtype)
    field_name, field_desc = sdfg_builder.add_temp_array(
        sdfg, field_shape, dataflow_output_desc.dtype
    )
    # the inner and outer strides have to match
    scan_output_stride = field_desc.strides[scan_dim_index]
    # also consider the stride of the local dimension, in case the scan field operator computes a list
    local_strides = field_desc.strides[len(domain_dims) :]
    assert len(local_strides) == (1 if isinstance(output_edge.result.gt_dtype, ts.ListType) else 0)
    new_inner_strides = [scan_output_stride, *local_strides]
    dataflow_output_desc.set_shape(dataflow_output_desc.shape, new_inner_strides)

    # and here the edge writing the dataflow result data through the map exit node
    field_node = state.add_access(field_name)
    output_edge.connect(map_exit, field_node, field_subset)

    return gtir_translators.FieldopData(
        field_node,
        ts.FieldType(field_dims, field_dtype),
        offset=(field_offset if set(field_offset) != {0} else None),
    )


def _create_field_operator(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    domain: gtir_translators.FieldopDomain,
    node_type: ts.FieldType | ts.TupleType,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
    input_edges: Iterable[gtir_dataflow.DataflowInputEdge],
    output_edges: gtir_dataflow.DataflowOutputEdge
    | tuple[gtir_dataflow.DataflowOutputEdge | tuple[Any, ...], ...],
    scan_dim: gtx_common.Dimension,
) -> gtir_translators.FieldopResult:
    """
    Helper method to build the output of a field operator, which can consist of
    a single field or a tuple of fields.

    A tuple of fields is returned when one stencil computes a grid point on multiple
    fields: for each field, this method will call `_create_field_operator_impl()`.

    Args:
        sdfg: The SDFG that represents the scope of the field data.
        state: The SDFG state where to create an access node to the field data.
        domain: The domain of the field operator that computes the field.
        node_type: The GT4Py type of the IR node that produces this field.
        sdfg_builder: The object used to build the map scope in the provided SDFG.
        input_edges: List of edges to pass input data into the dataflow.
        output_edges: Single edge or tuple of edges representing the dataflow output data.
        scan_dim: Column dimension used in scan field operators.

    Returns:
        The descriptor of the field operator result, which can be either a single field
        or a tuple fields.
    """
    domain_dims, _, _ = gtir_translators.get_field_layout(domain)

    assert scan_dim in domain_dims
    if len(domain_dims) == 1:
        # We construct the scan field operator on the horizontal domain, while the
        # vertical dimension (the column axis) is computed by the loop region.
        # If the field operator computes only the column axis (a 1d scan field operator),
        # there is no horizontal domain, therefore the map scope is not needed.
        # This case currently triggers a DaCe issue and produces wrong CUDA code,
        # thus the corresponding test is disabled (see pytest marker `uses_scan_1d_field`).
        map_entry, map_exit = (None, None)
    else:
        # create map range corresponding to the field operator domain
        map_entry, map_exit = sdfg_builder.add_map(
            "fieldop",
            state,
            ndrange={
                dace_gtir_utils.get_map_variable(dim): f"{lower_bound}:{upper_bound}"
                for dim, lower_bound, upper_bound in domain
                if dim != scan_dim
            },
        )

    # here we setup the edges passing through the map entry node
    for edge in input_edges:
        edge.connect(map_entry)

    if isinstance(node_type, ts.FieldType):
        assert isinstance(output_edges, gtir_dataflow.DataflowOutputEdge)
        return _create_field_operator_impl(
            sdfg_builder, sdfg, state, domain, output_edges, node_type, map_exit, scan_dim
        )
    else:
        # handle tuples of fields
        output_symbol_tree = dace_gtir_utils.make_symbol_tree("x", node_type)
        return gtx_utils.tree_map(
            lambda output_edge, output_sym: (
                _create_field_operator_impl(
                    sdfg_builder,
                    sdfg,
                    state,
                    domain,
                    output_edge,
                    output_sym.type,
                    map_exit,
                    scan_dim,
                )
            )
        )(output_edges, output_symbol_tree)


def translate_scan(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
) -> gtir_translators.FieldopResult:
    assert isinstance(node, gtir.FunCall)
    assert cpm.is_call_to(node.fun, "as_fieldop")
    assert isinstance(node.type, (ts.FieldType, ts.TupleType))

    fun_node = node.fun
    assert len(fun_node.args) == 2
    scan_expr, domain_expr = fun_node.args
    assert cpm.is_call_to(scan_expr, "scan")

    # parse the domain of the scan field operator
    domain = gtir_translators.extract_domain(domain_expr)

    # use the vertical dimension in the domain as scan dimension
    scan_domain = [
        (dim, lower_bound, upper_bound)
        for dim, lower_bound, upper_bound in domain
        if sdfg_builder.is_column_axis(dim)
    ]
    assert len(scan_domain) == 1
    scan_dim, scan_lower_bound, scan_upper_bound = scan_domain[0]

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

    # make naming consistent throughut this function scope
    def scan_input_name(input_name: str) -> str:
        return f"__gtir_scan_input_{input_name}"

    def scan_output_name(input_name: str) -> str:
        return f"__gtir_scan_output_{input_name}"

    # define the set of symbols available in the lambda context, which consists of
    # the carry argument and all lambda function arguments
    lambda_arg_types = [scan_carry_type] + [
        arg.type for arg in node.args if isinstance(arg.type, ts.DataType)
    ]
    lambda_symbols = {
        str(p.id): arg_type
        for p, arg_type in zip(stencil_expr.params, lambda_arg_types, strict=True)
    }

    # visit the arguments to be passed to the lambda expression
    # this must be executed before visiting the lambda expression, in order to populate
    # the data descriptor with the correct field domain offsets for field arguments
    lambda_args = [sdfg_builder.visit(arg, sdfg=sdfg, head_state=state) for arg in node.args]
    lambda_args_mapping = {
        scan_input_name(scan_carry): init_data,
    } | {
        str(param.id): arg for param, arg in zip(stencil_expr.params[1:], lambda_args, strict=True)
    }

    # parse the dataflow input and output symbols
    lambda_flat_args: dict[str, gtir_translators.FieldopData] = {}
    # the field offset is set to `None` when it is zero in all dimensions
    lambda_field_offsets: dict[str, Optional[list[dace.symbolic.SymExpr]]] = {}
    for param, outer_arg in lambda_args_mapping.items():
        tuple_fields = gtir_translators.flatten_tuples(param, outer_arg)
        lambda_field_offsets |= {tsym: tfield.offset for tsym, tfield in tuple_fields}
        lambda_flat_args |= dict(tuple_fields)
    if isinstance(scan_carry_type, ts.TupleType):
        lambda_flat_outs = {
            str(sym.id): sym.type
            for sym in dace_gtir_utils.flatten_tuple_fields(
                scan_output_name(scan_carry), scan_carry_type
            )
        }
    else:
        lambda_flat_outs = {scan_output_name(scan_carry): scan_carry_type}

    # the lambda expression, i.e. body of the scan, will be created inside a nested SDFG.
    nsdfg = dace.SDFG(sdfg_builder.unique_nsdfg_name(sdfg, "scan"))
    nsdfg.debuginfo = dace_utils.debug_info(node, default=sdfg.debuginfo)
    lambda_translator = sdfg_builder.nested_context(
        stencil_expr, nsdfg, lambda_symbols, lambda_field_offsets
    )

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

    # extract the scan loop range
    scan_loop_var = dace_gtir_utils.get_map_variable(scan_dim)

    # create a loop region for lambda call over the scan dimension
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
    init_state = nsdfg.add_state("scan_init", is_start_block=True)
    nsdfg.add_edge(init_state, scan_loop, dace.InterstateEdge())
    compute_state = scan_loop.add_state("scan_compute")
    update_state = scan_loop.add_state_after(compute_state, "scan_update")

    # visit the list of arguments to be passed to the scan expression
    stencil_args = [
        _parse_fieldop_arg(im.ref(p.id), nsdfg, compute_state, lambda_translator, domain)
        for p in stencil_expr.params
    ]

    # generate the dataflow representing the scan field operator
    lambda_input_edges, lambda_result = gtir_dataflow.translate_lambda_to_dataflow(
        nsdfg, compute_state, lambda_translator, stencil_expr, args=stencil_args
    )

    # now initialize the scan carry
    scan_carry_input = (
        dace_gtir_utils.make_symbol_tree(scan_carry, scan_carry_type)
        if isinstance(scan_carry_type, ts.TupleType)
        else im.sym(scan_carry, scan_carry_type)
    )

    def init_scan_carry(sym: gtir.Sym) -> None:
        scan_carry_dataname = str(sym.id)
        scan_carry_desc = nsdfg.data(scan_carry_dataname)
        input_scan_carry_dataname = scan_input_name(scan_carry_dataname)
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

    # connect the dataflow input directly to the source data nodes, without passing through a map node;
    # the reason is that the map for horizontal domain is outside the scan loop region
    for edge in lambda_input_edges:
        edge.connect(map_entry=None)

    # connect the dataflow result nodes to the carry variables
    output_column_index = dace.symbolic.pystr_to_symbolic(scan_loop_var) - scan_lower_bound

    def connect_scan_output(
        scan_output_edge: gtir_dataflow.DataflowOutputEdge,
        scan_output_shape: list[dace.symbolic.SymExpr],
        sym: gtir.Sym,
    ) -> gtir_translators.FieldopData:
        scan_result = scan_output_edge.result
        if isinstance(scan_result.gt_dtype, ts.ScalarType):
            assert scan_result.gt_dtype == sym.type
            # the scan field operator computes a column of scalar values
            assert len(scan_output_shape) == 1
            output_subset = dace_subsets.Range.from_string(str(output_column_index))
        else:
            assert isinstance(sym.type, ts.ListType)
            assert scan_result.gt_dtype.element_type == sym.type.element_type
            # the scan field operator computes a list of scalar values for each column level
            assert len(scan_output_shape) == 2
            output_subset = dace_subsets.Range.from_string(
                f"{output_column_index}, 0:{scan_output_shape[1]}"
            )
        scan_result_data = scan_result.dc_node.data
        scan_result_desc = scan_result.dc_node.desc(nsdfg)

        # `sym` represents the global output data, that is the nested-SDFG output connector
        lambda_output = str(sym.id)
        output = scan_output_name(lambda_output)
        nsdfg.add_array(output, scan_output_shape, scan_result_desc.dtype)
        output_node = compute_state.add_access(output)

        compute_state.add_nedge(
            scan_result.dc_node, output_node, dace.Memlet(data=output, subset=output_subset)
        )

        update_state.add_nedge(
            update_state.add_access(scan_result_data),
            update_state.add_access(lambda_output),
            dace.Memlet(data=lambda_output, subset="0"),
        )

        output_type = ts.FieldType(dims=[scan_dim], dtype=scan_result.gt_dtype)
        return gtir_translators.FieldopData(output_node, output_type, offset=scan_lower_bound)

    if isinstance(scan_carry_input, tuple):
        assert isinstance(lambda_result, tuple)
        assert isinstance(lambda_result_shape, tuple)
        lambda_output = gtx_utils.tree_map(connect_scan_output)(
            lambda_result, lambda_result_shape, scan_carry_input
        )
    else:
        assert isinstance(lambda_result, gtir_dataflow.DataflowOutputEdge)
        assert isinstance(lambda_result_shape, list)
        lambda_output = connect_scan_output(lambda_result, lambda_result_shape, scan_carry_input)

    # in case tuples are passed as argument, isolated access nodes might be left in the state,
    # because not all tuple fields are necessarily accessed inside the lambda scope
    for data_node in compute_state.data_nodes():
        data_desc = data_node.desc(nsdfg)
        if (compute_state.degree(data_node) == 0) and (
            (not data_desc.transient)
            or data_node.data.startswith(
                scan_carry
            )  # exceptional case where the carry variable is not used, not a scan indeed
        ):
            compute_state.remove_node(data_node)

    # build the mapping of symbols from nested SDFG to parent SDFG
    nsdfg_symbols_mapping = {str(sym): sym for sym in nsdfg.free_symbols}
    for inner_dataname, outer_arg in lambda_flat_args.items():
        inner_desc = nsdfg.data(inner_dataname)
        outer_desc = outer_arg.dc_node.desc(sdfg)
        nsdfg_symbols_mapping |= {
            str(nested_symbol): parent_symbol
            for nested_symbol, parent_symbol in zip(
                [*inner_desc.shape, *inner_desc.strides],
                [*outer_desc.shape, *outer_desc.strides],
                strict=True,
            )
            if isinstance(nested_symbol, dace.symbol)
        }

    # the scan nested SDFG is ready, now we need to instantiate it inside the map implementing the field operator
    nsdfg_node = state.add_nested_sdfg(
        nsdfg,
        sdfg,
        inputs=set(lambda_flat_args.keys()),
        outputs=set(lambda_flat_outs.keys()),
        symbol_mapping=nsdfg_symbols_mapping,
    )

    lambda_input_edges = []
    for input_connector, outer_arg in lambda_flat_args.items():
        arg_desc = outer_arg.dc_node.desc(sdfg)
        input_subset = dace_subsets.Range.from_array(arg_desc)
        input_edge = gtir_dataflow.MemletInputEdge(
            state, outer_arg.dc_node, input_subset, nsdfg_node, input_connector
        )
        lambda_input_edges.append(input_edge)

    def construct_output_edge(
        scan_data: gtir_translators.FieldopData,
    ) -> gtir_dataflow.DataflowOutputEdge:
        assert isinstance(scan_data.gt_type, ts.FieldType)
        inner_data = scan_data.dc_node.data
        inner_desc = nsdfg.data(inner_data)
        output_data, output_desc = sdfg.add_temp_transient_like(inner_desc)
        output_node = state.add_access(output_data)
        state.add_edge(
            nsdfg_node,
            inner_data,
            output_node,
            None,
            dace.Memlet.from_array(output_data, output_desc),
        )
        output_expr = gtir_dataflow.ValueExpr(output_node, scan_data.gt_type.dtype)
        return gtir_dataflow.DataflowOutputEdge(state, output_expr)

    output_edges = (
        construct_output_edge(lambda_output)
        if isinstance(lambda_output, gtir_translators.FieldopData)
        else gtx_utils.tree_map(construct_output_edge)(lambda_output)
    )

    return _create_field_operator(
        sdfg, state, domain, node.type, sdfg_builder, lambda_input_edges, output_edges, scan_dim
    )
