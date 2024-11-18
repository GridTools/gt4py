# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import Optional, Sequence, cast

import dace
from dace.sdfg.state import LoopRegion

import gt4py.eve as eve
from gt4py.next import Dimension, DimensionKind, common
from gt4py.next.ffront import fbuiltins as gtx_fbuiltins
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir import Expr, FunCall, Literal, Sym, SymRef
from gt4py.next.program_processors.runners.dace_common import utility as dace_utils
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation as tt

from .itir_to_tasklet import (
    Context,
    GatherOutputSymbolsPass,
    PythonTaskletCodegen,
    SymbolExpr,
    TaskletExpr,
    ValueExpr,
    closure_to_tasklet_sdfg,
    is_scan,
)
from .utility import (
    add_mapped_nested_sdfg,
    flatten_list,
    get_used_connectivities,
    map_nested_sdfg_symbols,
    new_array_symbols,
    unique_var_name,
)


def _get_scan_args(stencil: Expr) -> tuple[bool, Literal]:
    """
    Parse stencil expression to extract the scan arguments.

    Returns
    -------
    tuple(is_forward, init_carry)
        The output tuple fields verify the following semantics:
            - is_forward: forward boolean flag
            - init_carry: carry initial value
    """
    stencil_fobj = cast(FunCall, stencil)
    is_forward = stencil_fobj.args[1]
    assert isinstance(is_forward, Literal) and type_info.is_logical(is_forward.type)
    init_carry = stencil_fobj.args[2]
    assert isinstance(init_carry, Literal)
    return is_forward.value == "True", init_carry


def _get_scan_dim(
    column_axis: Dimension,
    storage_types: dict[str, ts.TypeSpec],
    output: SymRef,
    use_field_canonical_representation: bool,
) -> tuple[str, int, ts.ScalarType]:
    """
    Extract information about the scan dimension.

    Returns
    -------
    tuple(scan_dim_name, scan_dim_index, scan_dim_dtype)
        The output tuple fields verify the following semantics:
            - scan_dim_name: name of the scan dimension
            - scan_dim_index: domain index of the scan dimension
            - scan_dim_dtype: data type along the scan dimension
    """
    output_type = storage_types[output.id]
    assert isinstance(output_type, ts.FieldType)
    sorted_dims = [
        dim
        for _, dim in (
            dace_utils.get_sorted_dims(output_type.dims)
            if use_field_canonical_representation
            else enumerate(output_type.dims)
        )
    ]
    return (column_axis.value, sorted_dims.index(column_axis), output_type.dtype)


def _make_array_shape_and_strides(
    name: str,
    dims: Sequence[Dimension],
    offset_provider_type: common.OffsetProviderType,
    sort_dims: bool,
) -> tuple[list[dace.symbol], list[dace.symbol]]:
    """
    Parse field dimensions and allocate symbols for array shape and strides.

    For local dimensions, the size is known at compile-time and therefore
    the corresponding array shape dimension is set to an integer literal value.

    Returns
    -------
    tuple(shape, strides)
        The output tuple fields are arrays of dace symbolic expressions.
    """
    dtype = dace.dtype_to_typeclass(gtx_fbuiltins.IndexType)
    sorted_dims = dace_utils.get_sorted_dims(dims) if sort_dims else list(enumerate(dims))
    connectivity_types = dace_utils.filter_connectivity_types(offset_provider_type)
    shape = [
        (
            connectivity_types[dim.value].max_neighbors
            if dim.kind == DimensionKind.LOCAL
            # we reuse the same gt4py symbol for field size passed as scalar argument which is used in closure domain
            else dace.symbol(dace_utils.field_size_symbol_name(name, i), dtype)
        )
        for i, dim in sorted_dims
    ]
    strides = [
        dace.symbol(dace_utils.field_stride_symbol_name(name, i), dtype) for i, _ in sorted_dims
    ]
    return shape, strides


def _check_no_lifts(node: itir.StencilClosure):
    """
    Parse stencil closure ITIR to check that lift expressions only appear as child nodes in neighbor reductions.

    Returns
    -------
    True if lifts do not appear in the ITIR exception lift expressions in neighbor reductions. False otherwise.
    """
    neighbors_call_count = 0
    for fun in eve.walk_values(node).if_isinstance(itir.FunCall).getattr("fun"):
        if getattr(fun, "id", "") == "neighbors":
            neighbors_call_count = 3
        elif getattr(fun, "id", "") == "lift" and neighbors_call_count != 1:
            return False
        neighbors_call_count = max(0, neighbors_call_count - 1)
    return True


class ItirToSDFG(eve.NodeVisitor):
    param_types: list[ts.TypeSpec]
    storage_types: dict[str, ts.TypeSpec]
    column_axis: Optional[Dimension]
    offset_provider_type: common.OffsetProviderType
    unique_id: int
    use_field_canonical_representation: bool

    def __init__(
        self,
        param_types: list[ts.TypeSpec],
        offset_provider_type: common.OffsetProviderType,
        tmps: list[itir.Temporary],
        use_field_canonical_representation: bool,
        column_axis: Optional[Dimension] = None,
    ):
        self.param_types = param_types
        self.column_axis = column_axis
        self.offset_provider_type = offset_provider_type
        self.storage_types = {}
        self.tmps = tmps
        self.use_field_canonical_representation = use_field_canonical_representation

    def add_storage(self, sdfg: dace.SDFG, name: str, type_: ts.TypeSpec, sort_dimensions: bool):
        if isinstance(type_, ts.FieldType):
            shape, strides = _make_array_shape_and_strides(
                name, type_.dims, self.offset_provider_type, sort_dimensions
            )
            dtype = dace_utils.as_dace_type(type_.dtype)
            sdfg.add_array(name, shape=shape, strides=strides, dtype=dtype)

        elif isinstance(type_, ts.ScalarType):
            dtype = dace_utils.as_dace_type(type_)
            if name in sdfg.symbols:
                assert sdfg.symbols[name].dtype == dtype
            else:
                sdfg.add_symbol(name, dtype)

        else:
            raise NotImplementedError()
        self.storage_types[name] = type_

    def add_storage_for_temporaries(
        self, node_params: list[Sym], defs_state: dace.SDFGState, program_sdfg: dace.SDFG
    ) -> dict[str, str]:
        symbol_map: dict[str, TaskletExpr] = {}
        # The shape of temporary arrays might be defined based on scalar values passed as program arguments.
        # Here we collect these values in a symbol map.
        for sym in node_params:
            if isinstance(sym.type, ts.ScalarType):
                name_ = str(sym.id)
                symbol_map[name_] = SymbolExpr(name_, dace_utils.as_dace_type(sym.type))

        tmp_symbols: dict[str, str] = {}
        for tmp in self.tmps:
            tmp_name = str(tmp.id)

            # We visit the domain of the temporary field, passing the set of available symbols.
            assert isinstance(tmp.domain, itir.FunCall)
            domain_ctx = Context(program_sdfg, defs_state, symbol_map)
            tmp_domain = self._visit_domain(tmp.domain, domain_ctx)

            if isinstance(tmp.type, ts.TupleType):
                raise NotImplementedError("Temporaries of tuples are not supported.")
            assert isinstance(tmp.type, ts.FieldType) and isinstance(tmp.dtype, ts.ScalarType)

            # We store the FieldType for this temporary array.
            self.storage_types[tmp_name] = tmp.type

            # N.B.: skip generation of symbolic strides and just let dace assign default strides, for now.
            # Another option, in the future, is to use symbolic strides and apply auto-tuning or some heuristics
            # to assign optimal stride values.
            tmp_shape, _ = new_array_symbols(tmp_name, len(tmp.type.dims))
            _, tmp_array = program_sdfg.add_array(
                tmp_name, tmp_shape, dace_utils.as_dace_type(tmp.dtype), transient=True
            )

            # Loop through all dimensions to visit the symbolic expressions for array shape and offset.
            # These expressions are later mapped to interstate symbols.
            for (_, (begin, end)), shape_sym in zip(tmp_domain, tmp_array.shape):
                # The temporary field has a dimension range defined by `begin` and `end` values.
                # Therefore, the actual size is given by the difference `end.value - begin.value`.
                # Instead of allocating the actual size, we allocate space to enable indexing from 0
                # because we want to avoid using dace array offsets (which will be deprecated soon).
                # The result should still be valid, but the stencil will be using only a subset
                # of the array.
                if not (isinstance(begin, SymbolExpr) and begin.value == "0"):
                    warnings.warn(
                        f"Domain start offset for temporary {tmp_name} is ignored.", stacklevel=2
                    )
                tmp_symbols[str(shape_sym)] = end.value

        return tmp_symbols

    def create_memlet_at(self, field_name: str, index: dict[str, str]):
        field_type = self.storage_types[field_name]
        assert isinstance(field_type, ts.FieldType)
        if self.use_field_canonical_representation:
            field_index = [
                index[dim.value] for _, dim in dace_utils.get_sorted_dims(field_type.dims)
            ]
        else:
            field_index = [index[dim.value] for dim in field_type.dims]
        subset = ", ".join(field_index)
        return dace.Memlet(data=field_name, subset=subset)

    def get_output_nodes(
        self, closure: itir.StencilClosure, sdfg: dace.SDFG, state: dace.SDFGState
    ) -> dict[str, dace.nodes.AccessNode]:
        # Visit output node, which could be a `make_tuple` expression, to collect the required access nodes
        output_symbols_pass = GatherOutputSymbolsPass(sdfg, state)
        output_symbols_pass.visit(closure.output)
        # Visit output node again to generate the corresponding tasklet
        context = Context(sdfg, state, output_symbols_pass.symbol_refs)
        translator = PythonTaskletCodegen(
            self.offset_provider_type, context, self.use_field_canonical_representation
        )
        output_nodes = flatten_list(translator.visit(closure.output))
        return {node.value.data: node.value for node in output_nodes}

    def visit_FencilDefinition(self, node: itir.FencilDefinition):
        program_sdfg = dace.SDFG(name=node.id)
        program_sdfg.debuginfo = dace_utils.debug_info(node)
        entry_state = program_sdfg.add_state("program_entry", is_start_block=True)

        # Filter neighbor tables from offset providers.
        connectivity_types = get_used_connectivities(node, self.offset_provider_type)

        # Add program parameters as SDFG storages.
        for param, type_ in zip(node.params, self.param_types):
            self.add_storage(
                program_sdfg, str(param.id), type_, self.use_field_canonical_representation
            )

        if self.tmps:
            tmp_symbols = self.add_storage_for_temporaries(node.params, entry_state, program_sdfg)
            # on the first interstate edge define symbols for shape and offsets of temporary arrays
            last_state = program_sdfg.add_state("init_symbols_for_temporaries")
            program_sdfg.add_edge(
                entry_state, last_state, dace.InterstateEdge(assignments=tmp_symbols)
            )
        else:
            last_state = entry_state

        # Add connectivities as SDFG storages.
        for offset, connectivity_type in connectivity_types.items():
            scalar_type = tt.from_dtype(connectivity_type.dtype)
            type_ = ts.FieldType(
                [connectivity_type.source_dim, connectivity_type.neighbor_dim], scalar_type
            )
            self.add_storage(
                program_sdfg,
                dace_utils.connectivity_identifier(offset),
                type_,
                sort_dimensions=False,
            )

        # Create a nested SDFG for all stencil closures.
        for closure in node.closures:
            # Translate the closure and its stencil's body to an SDFG.
            closure_sdfg, input_names, output_names = self.visit(
                closure, array_table=program_sdfg.arrays
            )

            # Create a new state for the closure.
            last_state = program_sdfg.add_state_after(last_state)

            # Create memlets to transfer the program parameters
            input_mapping = {
                name: dace.Memlet.from_array(name, program_sdfg.arrays[name])
                for name in input_names
            }
            output_mapping = {
                name: dace.Memlet.from_array(name, program_sdfg.arrays[name])
                for name in output_names
            }

            symbol_mapping = map_nested_sdfg_symbols(program_sdfg, closure_sdfg, input_mapping)

            # Insert the closure's SDFG as a nested SDFG of the program.
            nsdfg_node = last_state.add_nested_sdfg(
                sdfg=closure_sdfg,
                parent=program_sdfg,
                inputs=set(input_names),
                outputs=set(output_names),
                symbol_mapping=symbol_mapping,
                debuginfo=closure_sdfg.debuginfo,
            )

            # Add access nodes for the program parameters and connect them to the nested SDFG's inputs via edges.
            for inner_name, memlet in input_mapping.items():
                access_node = last_state.add_access(inner_name, debuginfo=nsdfg_node.debuginfo)
                last_state.add_edge(access_node, None, nsdfg_node, inner_name, memlet)

            for inner_name, memlet in output_mapping.items():
                access_node = last_state.add_access(inner_name, debuginfo=nsdfg_node.debuginfo)
                last_state.add_edge(nsdfg_node, inner_name, access_node, None, memlet)

        # Create the call signature for the SDFG.
        #  Only the arguments requiered by the Fencil, i.e. `node.params` are added as positional arguments.
        #  The implicit arguments, such as the offset providers or the arguments created by the translation process, must be passed as keywords only arguments.
        program_sdfg.arg_names = [str(a) for a in node.params]

        program_sdfg.validate()
        return program_sdfg

    def visit_StencilClosure(
        self, node: itir.StencilClosure, array_table: dict[str, dace.data.Array]
    ) -> tuple[dace.SDFG, list[str], list[str]]:
        assert _check_no_lifts(node)

        # Create the closure's nested SDFG and single state.
        closure_sdfg = dace.SDFG(name="closure")
        closure_sdfg.debuginfo = dace_utils.debug_info(node)
        closure_state = closure_sdfg.add_state("closure_entry")
        closure_init_state = closure_sdfg.add_state_before(closure_state, "closure_init", True)

        assert all(
            isinstance(inp, SymRef) for inp in node.inputs
        )  # backend only supports SymRef inputs, not `index` calls
        input_names = [str(inp.id) for inp in node.inputs]  # type: ignore[union-attr]  # ensured by assert
        neighbor_tables = get_used_connectivities(node, self.offset_provider_type)
        connectivity_names = [
            dace_utils.connectivity_identifier(offset) for offset in neighbor_tables.keys()
        ]

        output_nodes = self.get_output_nodes(node, closure_sdfg, closure_state)
        output_names = [k for k, _ in output_nodes.items()]

        # Add DaCe arrays for inputs, outputs and connectivities to closure SDFG.
        input_transients_mapping = {}
        for name in [*input_names, *connectivity_names, *output_names]:
            if name in closure_sdfg.arrays:
                assert name in input_names and name in output_names
                # In case of closures with in/out fields, there is risk of race condition
                # between read/write access nodes in the (asynchronous) map tasklet.
                transient_name = unique_var_name()
                closure_sdfg.add_array(
                    transient_name,
                    shape=array_table[name].shape,
                    strides=array_table[name].strides,
                    dtype=array_table[name].dtype,
                    transient=True,
                )
                closure_init_state.add_nedge(
                    closure_init_state.add_access(name, debuginfo=closure_sdfg.debuginfo),
                    closure_init_state.add_access(transient_name, debuginfo=closure_sdfg.debuginfo),
                    dace.Memlet.from_array(name, closure_sdfg.arrays[name]),
                )
                input_transients_mapping[name] = transient_name
            elif isinstance(self.storage_types[name], ts.FieldType):
                closure_sdfg.add_array(
                    name,
                    shape=array_table[name].shape,
                    strides=array_table[name].strides,
                    dtype=array_table[name].dtype,
                )
            else:
                assert isinstance(self.storage_types[name], ts.ScalarType)

        input_field_names = [
            input_name
            for input_name in input_names
            if isinstance(self.storage_types[input_name], ts.FieldType)
        ]

        # Closure outputs should all be fields
        assert all(
            isinstance(self.storage_types[output_name], ts.FieldType)
            for output_name in output_names
        )

        # Update symbol table and get output domain of the closure
        program_arg_syms: dict[str, TaskletExpr] = {}
        for name, type_ in self.storage_types.items():
            if isinstance(type_, ts.ScalarType):
                dtype = dace_utils.as_dace_type(type_)
                if name in input_names:
                    out_name = unique_var_name()
                    closure_sdfg.add_scalar(out_name, dtype, transient=True)
                    out_tasklet = closure_init_state.add_tasklet(
                        f"get_{name}",
                        {},
                        {"__result"},
                        f"__result = {name}",
                        debuginfo=closure_sdfg.debuginfo,
                    )
                    access = closure_init_state.add_access(
                        out_name, debuginfo=closure_sdfg.debuginfo
                    )
                    value = ValueExpr(access, dtype)
                    memlet = dace.Memlet(data=out_name, subset="0")
                    closure_init_state.add_edge(out_tasklet, "__result", access, None, memlet)
                    program_arg_syms[name] = value
                else:
                    program_arg_syms[name] = SymbolExpr(name, dtype)
            else:
                assert isinstance(type_, ts.FieldType)
                # make shape symbols (corresponding to field size) available as arguments to domain visitor
                if name in input_names or name in output_names:
                    field_symbols = [
                        val
                        for val in closure_sdfg.arrays[name].shape
                        if isinstance(val, dace.symbol) and str(val) not in input_names
                    ]
                    for sym in field_symbols:
                        sym_name = str(sym)
                        program_arg_syms[sym_name] = SymbolExpr(sym, sym.dtype)
        closure_ctx = Context(closure_sdfg, closure_state, program_arg_syms)
        closure_domain = self._visit_domain(node.domain, closure_ctx)

        # Map SDFG tasklet arguments to parameters
        input_local_names = [
            (
                input_transients_mapping[input_name]
                if input_name in input_transients_mapping
                else (
                    input_name
                    if input_name in input_field_names
                    else cast(ValueExpr, program_arg_syms[input_name]).value.data
                )
            )
            for input_name in input_names
        ]
        input_memlets = [
            dace.Memlet.from_array(name, closure_sdfg.arrays[name])
            for name in [*input_local_names, *connectivity_names]
        ]

        # create and write to transient that is then copied back to actual output array to avoid aliasing of
        # same memory in nested SDFG with different names
        output_connectors_mapping = {unique_var_name(): output_name for output_name in output_names}
        # scan operator should always be the first function call in a closure
        if is_scan(node.stencil):
            assert len(output_connectors_mapping) == 1, "Scan does not support multiple outputs"
            transient_name, output_name = next(iter(output_connectors_mapping.items()))

            nsdfg, map_ranges, scan_dim_index = self._visit_scan_stencil_closure(
                node, closure_sdfg.arrays, closure_domain, transient_name
            )
            results = [transient_name]

            _, (scan_lb, scan_ub) = closure_domain[scan_dim_index]
            output_subset = f"{scan_lb.value}:{scan_ub.value}"

            domain_subset = {
                dim: (
                    f"i_{dim}"
                    if f"i_{dim}" in map_ranges
                    else f"0:{closure_sdfg.arrays[output_name].shape[scan_dim_index]}"
                )
                for dim, _ in closure_domain
            }
            output_memlets = [self.create_memlet_at(output_name, domain_subset)]
        else:
            nsdfg, map_ranges, results = self._visit_parallel_stencil_closure(
                node, closure_sdfg.arrays, closure_domain
            )

            output_subset = "0"

            output_memlets = [
                self.create_memlet_at(output_name, {dim: f"i_{dim}" for dim, _ in closure_domain})
                for output_name in output_connectors_mapping.values()
            ]

        input_mapping = {
            param: arg for param, arg in zip([*input_names, *connectivity_names], input_memlets)
        }
        output_mapping = {param: memlet for param, memlet in zip(results, output_memlets)}

        symbol_mapping = map_nested_sdfg_symbols(closure_sdfg, nsdfg, input_mapping)

        nsdfg_node, map_entry, map_exit = add_mapped_nested_sdfg(
            closure_state,
            sdfg=nsdfg,
            map_ranges=map_ranges or {"__dummy": "0"},
            inputs=input_mapping,
            outputs=output_mapping,
            symbol_mapping=symbol_mapping,
            output_nodes=output_nodes,
            debuginfo=nsdfg.debuginfo,
        )
        access_nodes = {edge.data.data: edge.dst for edge in closure_state.out_edges(map_exit)}
        for edge in closure_state.in_edges(map_exit):
            memlet = edge.data
            if memlet.data not in output_connectors_mapping:
                continue
            transient_access = closure_state.add_access(memlet.data, debuginfo=nsdfg.debuginfo)
            closure_state.add_edge(
                nsdfg_node,
                edge.src_conn,
                transient_access,
                None,
                dace.Memlet(data=memlet.data, subset=output_subset, debuginfo=nsdfg.debuginfo),
            )
            inner_memlet = dace.Memlet(
                data=memlet.data, subset=output_subset, other_subset=memlet.subset
            )
            closure_state.add_edge(transient_access, None, map_exit, edge.dst_conn, inner_memlet)
            closure_state.remove_edge(edge)
            access_nodes[memlet.data].data = output_connectors_mapping[memlet.data]

        return closure_sdfg, input_field_names + connectivity_names, output_names

    def _visit_scan_stencil_closure(
        self,
        node: itir.StencilClosure,
        array_table: dict[str, dace.data.Array],
        closure_domain: tuple[
            tuple[str, tuple[ValueExpr | SymbolExpr, ValueExpr | SymbolExpr]], ...
        ],
        output_name: str,
    ) -> tuple[dace.SDFG, dict[str, str | dace.subsets.Subset], int]:
        # extract scan arguments
        is_forward, init_carry_value = _get_scan_args(node.stencil)
        # select the scan dimension based on program argument for column axis
        assert self.column_axis
        assert isinstance(node.output, SymRef)
        scan_dim, scan_dim_index, scan_dtype = _get_scan_dim(
            self.column_axis,
            self.storage_types,
            node.output,
            self.use_field_canonical_representation,
        )

        assert isinstance(node.output, SymRef)
        neighbor_tables = get_used_connectivities(node, self.offset_provider_type)
        assert all(
            isinstance(inp, SymRef) for inp in node.inputs
        )  # backend only supports SymRef inputs, not `index` calls
        input_names = [str(inp.id) for inp in node.inputs]  # type: ignore[union-attr]  # ensured by assert
        connectivity_names = [
            dace_utils.connectivity_identifier(offset) for offset in neighbor_tables.keys()
        ]

        # find the scan dimension, same as output dimension, and exclude it from the map domain
        map_ranges = {}
        for dim, (lb, ub) in closure_domain:
            lb_str = lb.value.data if isinstance(lb, ValueExpr) else lb.value
            ub_str = ub.value.data if isinstance(ub, ValueExpr) else ub.value
            if not dim == scan_dim:
                map_ranges[f"i_{dim}"] = f"{lb_str}:{ub_str}"
            else:
                scan_lb_str = lb_str
                scan_ub_str = ub_str

        # the scan operator is implemented as an SDFG to be nested in the closure SDFG
        scan_sdfg = dace.SDFG(name="scan")
        scan_sdfg.debuginfo = dace_utils.debug_info(node)

        # the carry value of the scan operator exists only in the scope of the scan sdfg
        scan_carry_name = unique_var_name()
        scan_sdfg.add_scalar(
            scan_carry_name, dtype=dace_utils.as_dace_type(scan_dtype), transient=True
        )

        # create a loop region for lambda call over the scan dimension
        scan_loop_var = f"i_{scan_dim}"
        if is_forward:
            scan_loop = LoopRegion(
                label="scan",
                condition_expr=f"{scan_loop_var} < {scan_ub_str}",
                loop_var=scan_loop_var,
                initialize_expr=f"{scan_loop_var} = {scan_lb_str}",
                update_expr=f"{scan_loop_var} = {scan_loop_var} + 1",
                inverted=False,
            )
        else:
            scan_loop = LoopRegion(
                label="scan",
                condition_expr=f"{scan_loop_var} >= {scan_lb_str}",
                loop_var=scan_loop_var,
                initialize_expr=f"{scan_loop_var} = {scan_ub_str} - 1",
                update_expr=f"{scan_loop_var} = {scan_loop_var} - 1",
                inverted=False,
            )
        scan_sdfg.add_node(scan_loop)
        compute_state = scan_loop.add_state("lambda_compute", is_start_block=True)
        update_state = scan_loop.add_state("lambda_update")
        scan_loop.add_edge(compute_state, update_state, dace.InterstateEdge())

        start_state = scan_sdfg.add_state("start", is_start_block=True)
        scan_sdfg.add_edge(start_state, scan_loop, dace.InterstateEdge())

        # tasklet for initialization of carry
        carry_init_tasklet = start_state.add_tasklet(
            "get_carry_init_value",
            {},
            {"__result"},
            f"__result = {init_carry_value}",
            debuginfo=scan_sdfg.debuginfo,
        )
        start_state.add_edge(
            carry_init_tasklet,
            "__result",
            start_state.add_access(scan_carry_name, debuginfo=scan_sdfg.debuginfo),
            None,
            dace.Memlet(data=scan_carry_name, subset="0"),
        )

        # add storage to scan SDFG for inputs
        for name in [*input_names, *connectivity_names]:
            assert name not in scan_sdfg.arrays
            if isinstance(self.storage_types[name], ts.FieldType):
                scan_sdfg.add_array(
                    name,
                    shape=array_table[name].shape,
                    strides=array_table[name].strides,
                    dtype=array_table[name].dtype,
                )
            else:
                scan_sdfg.add_scalar(
                    name,
                    dtype=dace_utils.as_dace_type(cast(ts.ScalarType, self.storage_types[name])),
                )
        # add storage to scan SDFG for output
        scan_sdfg.add_array(
            output_name,
            shape=(array_table[node.output.id].shape[scan_dim_index],),
            strides=(array_table[node.output.id].strides[scan_dim_index],),
            dtype=array_table[node.output.id].dtype,
        )

        # implement the lambda function as a nested SDFG that computes a single item in the scan dimension
        lambda_domain = {dim: f"i_{dim}" for dim, _ in closure_domain}
        input_arrays = [(scan_carry_name, scan_dtype)] + [
            (name, self.storage_types[name]) for name in input_names
        ]
        connectivity_arrays = [(scan_sdfg.arrays[name], name) for name in connectivity_names]
        lambda_context, lambda_outputs = closure_to_tasklet_sdfg(
            node,
            self.offset_provider_type,
            lambda_domain,
            input_arrays,
            connectivity_arrays,
            self.use_field_canonical_representation,
        )

        lambda_input_names = [name for name, _ in input_arrays]
        lambda_output_names = [connector.value.data for connector in lambda_outputs]

        input_memlets = [
            dace.Memlet.from_array(name, scan_sdfg.arrays[name]) for name in lambda_input_names
        ]
        connectivity_memlets = [
            dace.Memlet.from_array(name, scan_sdfg.arrays[name]) for name in connectivity_names
        ]
        input_mapping = {param: arg for param, arg in zip(lambda_input_names, input_memlets)}
        connectivity_mapping = {
            param: arg for param, arg in zip(connectivity_names, connectivity_memlets)
        }
        array_mapping = {**input_mapping, **connectivity_mapping}
        symbol_mapping = map_nested_sdfg_symbols(scan_sdfg, lambda_context.body, array_mapping)

        scan_inner_node = compute_state.add_nested_sdfg(
            lambda_context.body,
            parent=scan_sdfg,
            inputs=set(lambda_input_names) | set(connectivity_names),
            outputs=set(lambda_output_names),
            symbol_mapping=symbol_mapping,
            debuginfo=lambda_context.body.debuginfo,
        )

        # connect scan SDFG to lambda inputs
        for name, memlet in array_mapping.items():
            access_node = compute_state.add_access(name, debuginfo=lambda_context.body.debuginfo)
            compute_state.add_edge(access_node, None, scan_inner_node, name, memlet)

        output_names = [output_name]
        assert len(lambda_output_names) == 1
        # connect lambda output to scan SDFG
        for name, connector in zip(output_names, lambda_output_names):
            compute_state.add_edge(
                scan_inner_node,
                connector,
                compute_state.add_access(name, debuginfo=lambda_context.body.debuginfo),
                None,
                dace.Memlet(data=name, subset=scan_loop_var),
            )

        update_state.add_nedge(
            update_state.add_access(output_name, debuginfo=lambda_context.body.debuginfo),
            update_state.add_access(scan_carry_name, debuginfo=lambda_context.body.debuginfo),
            dace.Memlet(data=output_name, subset=scan_loop_var, other_subset="0"),
        )

        return scan_sdfg, map_ranges, scan_dim_index

    def _visit_parallel_stencil_closure(
        self,
        node: itir.StencilClosure,
        array_table: dict[str, dace.data.Array],
        closure_domain: tuple[
            tuple[str, tuple[ValueExpr | SymbolExpr, ValueExpr | SymbolExpr]], ...
        ],
    ) -> tuple[dace.SDFG, dict[str, str | dace.subsets.Subset], list[str]]:
        neighbor_tables = get_used_connectivities(node, self.offset_provider_type)
        assert all(
            isinstance(inp, SymRef) for inp in node.inputs
        )  # backend only supports SymRef inputs, not `index` calls
        input_names = [str(inp.id) for inp in node.inputs]  # type: ignore[union-attr]  # ensured by assert
        connectivity_names = [
            dace_utils.connectivity_identifier(offset) for offset in neighbor_tables.keys()
        ]

        # find the scan dimension, same as output dimension, and exclude it from the map domain
        map_ranges = {}
        for dim, (lb, ub) in closure_domain:
            lb_str = lb.value.data if isinstance(lb, ValueExpr) else lb.value
            ub_str = ub.value.data if isinstance(ub, ValueExpr) else ub.value
            map_ranges[f"i_{dim}"] = f"{lb_str}:{ub_str}"

        # Create an SDFG for the tasklet that computes a single item of the output domain.
        index_domain = {dim: f"i_{dim}" for dim, _ in closure_domain}

        input_arrays = [(name, self.storage_types[name]) for name in input_names]
        connectivity_arrays = [(array_table[name], name) for name in connectivity_names]

        context, results = closure_to_tasklet_sdfg(
            node,
            self.offset_provider_type,
            index_domain,
            input_arrays,
            connectivity_arrays,
            self.use_field_canonical_representation,
        )

        return context.body, map_ranges, [r.value.data for r in results]

    def _visit_domain(
        self, node: itir.FunCall, context: Context
    ) -> tuple[tuple[str, tuple[SymbolExpr | ValueExpr, SymbolExpr | ValueExpr]], ...]:
        assert isinstance(node.fun, itir.SymRef)
        assert node.fun.id == "cartesian_domain" or node.fun.id == "unstructured_domain"

        bounds: list[tuple[str, tuple[ValueExpr, ValueExpr]]] = []

        for named_range in node.args:
            assert isinstance(named_range, itir.FunCall)
            assert isinstance(named_range.fun, itir.SymRef)
            assert len(named_range.args) == 3
            dimension = named_range.args[0]
            assert isinstance(dimension, itir.AxisLiteral)
            lower_bound = named_range.args[1]
            upper_bound = named_range.args[2]
            translator = PythonTaskletCodegen(
                self.offset_provider_type,
                context,
                self.use_field_canonical_representation,
            )
            lb = translator.visit(lower_bound)[0]
            ub = translator.visit(upper_bound)[0]
            bounds.append((dimension.value, (lb, ub)))

        return tuple(bounds)

    @staticmethod
    def _check_shift_offsets_are_literals(node: itir.StencilClosure):
        fun_calls = eve.walk_values(node).if_isinstance(itir.FunCall)
        shifts = [nd for nd in fun_calls if getattr(nd.fun, "id", "") == "shift"]
        for shift in shifts:
            if not all(isinstance(arg, (itir.Literal, itir.OffsetLiteral)) for arg in shift.args):
                return False
        return True
