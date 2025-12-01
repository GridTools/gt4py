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

from typing import Iterable, Sequence

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
from gt4py.next.program_processors.runners.dace.lowering import (
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
) -> MaybeNestedInTuple[gtir_dataflow.MemletExpr]:
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
        return gtx_utils.tree_map(_parse_fieldop_arg_impl)(arg)


def _create_scan_field_operator_impl(
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    output_edge: gtir_dataflow.DataflowOutputEdge | None,
    output_domain: infer_domain.NonTupleDomainAccess,
    output_type: ts.FieldType,
    map_exit: dace.nodes.MapExit | None,
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

    Another difference is that this function is called on all fields inside a tuple,
    in case of tuple return. Note that a regular field operator only computes a
    single field, never a tuple of fields. For tuples, it can happen that one of
    the nested fields is not used, outside the scan field operator, and therefore
    does not need to be computed. Then, the domain inferred by gt4py on this field
    is empty and the corresponding `output_edge` argument to this function is None.
    In this case, the function does not allocate an array node for the output field
    and returns None.

    Refer to `gtir_to_sdfg_primitives._create_field_operator_impl()` for
    the description of function arguments.
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

    # Now connect the output connector on the nested SDFG with the result field
    #  outside the scan map scope. For 1D domain, containing only the column dimension,
    #  there is no map scope, since the map range is only on the horizontal domain.
    #  Up to now the nested SDFG is writing into a transient data container that
    #  has the size to hold one column. The function below, that does the connection,
    #  will remove that transient and write directly to the result field.
    inner_map_output_temporary_removed = output_edge.connect(map_exit, field_node, field_subset)
    if not inner_map_output_temporary_removed:
        raise ValueError("The scan nested SDFG is expected to write directly to the result field.")

    assert ctx.state.in_degree(field_node) == 1
    field_node_path = ctx.state.memlet_path(next(iter(ctx.state.in_edges(field_node))))
    assert field_node_path[-1].dst is field_node

    # The temporary node which the nested SDFG was writing to has been deleted,
    #  and the nested SDFG will write directly to the result field. Thus, we have
    #  to modify the stride of the scan column array inside the nested SDFG to match
    #  the strides outside.
    nsdfg_scan = field_node_path[0].src
    assert isinstance(nsdfg_scan, dace.nodes.NestedSDFG)
    inner_output_name = field_node_path[0].src_conn
    inner_output_desc = nsdfg_scan.sdfg.arrays[inner_output_name]
    assert len(inner_output_desc.shape) == 1

    # The result field on the outside is a transient array, allocated inside this
    # function, so we know that its stride is constant. We just need to set it on
    # the inside array, and we do not need to map any stride symbol.
    outside_output_stride = field_desc.strides[scan_dim_index]
    assert str(outside_output_stride).isdigit()
    inner_output_desc.set_shape(new_shape=inner_output_desc.shape, strides=(outside_output_stride,))

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
    description of function arguments. Note that the return value is different,
    because the scan field operator can return a tuple of fields, while a regular
    field operator return a single field. The domain of the nested fields, in
    a tuple, can be empty, in case the nested field is not used outside the scan.
    In this case, the corresponding `output` edge will be None and this function
    will also return None for the corresponding field inside the tree-like result.
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

    # Note that `output_symbol` below is not used, we only need the tree-like
    # structure to get the type of each nested field in the `tree_map` visitor.
    dummy_output_symbol = (
        gtir_to_sdfg_utils.make_symbol_tree("__gtir_unused_dummy_var", node_type)
        if isinstance(node_type, ts.TupleType)
        else im.sym("__gtir_unused_dummy_var", node_type)
    )

    return gtx_utils.tree_map(
        lambda edge, domain, sym: (
            _create_scan_field_operator_impl(
                ctx,
                sdfg_builder,
                edge,
                domain,
                sym.type,
                map_exit,
            )
        )
    )(output, output_domain, dummy_output_symbol)


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
    lambda_params: Sequence[gtir.Sym],
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
        lambda_node: The lambda representing the stencil expression on the scan dimension.
        ctx: The SDFG context where the scan field operator is translated.
        sdfg_builder: The SDFG builder object to access the field operator context.
        field_domain: The field operator domain, with all horizontal and vertical dimensions.
        init_data: The data produced in the field operator context that is used
            to initialize the scan carry value.
        lambda_params: List of symbols used as parameters of the lambda expression.
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
        lambda_node,
        "scan",
        ctx,
        lambda_params,
        symbolic_inputs=set(),
        capture_scope_symbols=False,
    )

    # We set `using_explicit_control_flow=True` because the vertical scan is lowered to a `LoopRegion`.
    # This property is used by pattern matching in SDFG transformation framework
    # to skip those transformations that do not yet support control flow blocks.
    lambda_ctx.sdfg.using_explicit_control_flow = True

    # We use the entry state for initialization of the scan carry variable.
    init_state = lambda_ctx.state

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

    # Create the body of the initialization state, which will initialize the scan
    # carry variable. Note that we do it here, after having visited the lambda node
    # and having built 'compute_state', for a specific reason: the scan carry variable
    # was initially a lambda parameter, and threfore lowered as a non-transient node,
    # but it needs to be changed into transient (see `scan_carry_desc.transient = True`)
    # because it is only used as internal state. The input data will a different node,
    # which is copied to the scan carry variable in the initialization state.
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
        # Note that we set `transient=True` because the lowering expects the dataflow
        # of nested SDDFG to write to some internal temporary nodes. These data elements
        # should be turned into globals by the caller and handled as output connections.
        lambda_ctx.sdfg.add_array(output, scan_output_shape, scan_result_desc.dtype, transient=True)
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
    lambda_output = gtx_utils.tree_map(connect_scan_output)(
        lambda_result, lambda_result_shape, scan_carry_input
    )

    # Corner case where the scan computation, on one level, does not depend on
    # the result from previous level. In this case, the state information from
    # previous level is not used, therefore we could find isolated access nodes.
    # In case of tuples, it might be that only some of the fields are used.
    # In case of scalars, this is probably a misuse of scan in application code:
    # it could have been represented as a pure field operator.
    for arg in gtx_utils.flatten_nested_tuple((stencil_args[0],)):
        state_node = arg.dc_node
        if compute_state.degree(state_node) == 0:
            compute_state.remove_node(state_node)

    return lambda_ctx, lambda_output


def _handle_dataflow_result_of_nested_sdfg(
    nsdfg_node: dace.nodes.NestedSDFG,
    inner_ctx: gtir_to_sdfg.SubgraphContext,
    outer_ctx: gtir_to_sdfg.SubgraphContext,
    inner_data: gtir_to_sdfg_types.FieldopData,
    field_domain: infer_domain.NonTupleDomainAccess,
) -> gtir_dataflow.DataflowOutputEdge | None:
    assert isinstance(inner_data.gt_type, ts.FieldType)
    inner_dataname = inner_data.dc_node.data
    inner_desc = inner_ctx.sdfg.data(inner_dataname)
    assert inner_desc.transient

    if isinstance(field_domain, domain_utils.SymbolicDomain):
        # The field is used outside the nested SDFG, therefore it needs to be copied
        # to a temporary array in the parent SDFG (outer context).
        inner_desc.transient = False
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
    else:
        # The field is not used outside the nested SDFG. It is likely just storage
        # for some internal state, accessed during column scan, and can be turned
        # into a transient array inside the nested SDFG.
        assert field_domain == infer_domain.DomainAccessDescriptor.NEVER
        nsdfg_node.out_connectors.pop(inner_dataname)
        return None


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

    # define the symbols passed as parameter to the lambda expression, which consists
    # of the carry argument and all lambda function arguments
    lambda_arg_types: list[ts.DataType] = [scan_carry_type] + [
        arg.type for arg in node.args if isinstance(arg.type, ts.DataType)
    ]
    lambda_params = [
        im.sym(p.id, arg_type)
        for p, arg_type in zip(stencil_expr.params, lambda_arg_types, strict=True)
    ]

    # lower the scan stencil expression in a separate SDFG context
    lambda_ctx, lambda_output = _lower_lambda_to_nested_sdfg(
        stencil_expr,
        ctx,
        sdfg_builder,
        field_domain,
        init_data,
        lambda_params,
        scan_forward,
        im.sym(scan_carry, scan_carry_type),
    )

    # visit the arguments to be passed to the lambda expression
    # this must be executed before visiting the lambda expression, in order to populate
    # the data descriptor with the correct field domain offsets for field arguments
    lambda_args = [sdfg_builder.visit(arg, ctx=ctx) for arg in node.args]
    lambda_args_mapping = [
        (im.sym(_scan_input_name(scan_carry), scan_carry_type), init_data),
    ] + [
        (gt_symbol, arg)
        for gt_symbol, arg in zip(stencil_expr.params[1:], lambda_args, strict=True)
    ]

    lambda_arg_nodes, symbolic_args = gtir_to_sdfg.flatten_tuple_args(lambda_args_mapping)

    # The lambda expression of a scan field operator should never capture symbols
    # from the ouside scope, therefore we call `add_nested_sdfg()` with `capture_outer_data=False`.
    nsdfg_node, input_memlets = sdfg_builder.add_nested_sdfg(
        node=stencil_expr,
        inner_ctx=lambda_ctx,
        outer_ctx=ctx,
        symbolic_args=symbolic_args,
        data_args=lambda_arg_nodes,
        inner_result=lambda_output,
        capture_outer_data=False,
    )

    # Block the inlining of NestedSDFG containing a scan.
    #  We do this to avoid a bug in DaCe simplify transformations, see
    #  [issue#2182](https://github.com/spcl/dace/issues/2182) for more. Before the bug
    #  was hidden but once we started running `TrivialTaskletElimination` it was no
    #  longer the case. The solution is to block the inlining and keep scans localized
    #  inside their NestedSDFG.
    # NOTE: Currently there is no transformation that takes advantages in any way
    #   of a scan and they are mostly inside Maps anyway, except of unit tests,
    #   where inlining is not possible anyway.
    nsdfg_node.no_inline = True

    input_edges = []
    for input_connector, memlet in input_memlets.items():
        src_node = lambda_arg_nodes[input_connector]
        assert src_node is not None
        input_edge = gtir_dataflow.MemletInputEdge(
            ctx.state, src_node.dc_node, memlet.src_subset, nsdfg_node, input_connector
        )
        input_edges.append(input_edge)

    # for output connections, we create temporary arrays that contain the computation
    # results of a column slice for each point in the horizontal domain
    output_tree = gtx_utils.tree_map(
        lambda output_data, output_domain: _handle_dataflow_result_of_nested_sdfg(
            nsdfg_node=nsdfg_node,
            inner_ctx=lambda_ctx,
            outer_ctx=ctx,
            inner_data=output_data,
            field_domain=output_domain,
        )
    )(lambda_output, node.annex.domain)

    # we call a helper method to create a map scope that will compute the entire field
    return _create_scan_field_operator(
        ctx, field_domain, node.type, sdfg_builder, input_edges, output_tree, node.annex.domain
    )
