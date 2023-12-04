# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Optional, cast

import dace

import gt4py.eve as eve
from gt4py.next import Dimension, DimensionKind, type_inference as next_typing
from gt4py.next.iterator import ir as itir, type_inference as itir_typing
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider
from gt4py.next.iterator.ir import Expr, FunCall, Literal, SymRef
from gt4py.next.type_system import type_specifications as ts, type_translation

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
    as_dace_type,
    connectivity_identifier,
    create_memlet_at,
    create_memlet_full,
    filter_neighbor_tables,
    flatten_list,
    get_sorted_dims,
    map_nested_sdfg_symbols,
    new_array_symbols,
    unique_name,
    unique_var_name,
)


def get_scan_args(stencil: Expr) -> tuple[bool, Literal]:
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
    assert isinstance(is_forward, Literal) and is_forward.type == "bool"
    init_carry = stencil_fobj.args[2]
    assert isinstance(init_carry, Literal)
    return is_forward.value == "True", init_carry


def get_scan_dim(
    column_axis: Dimension,
    storage_types: dict[str, ts.TypeSpec],
    output: SymRef,
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
    output_type = cast(ts.FieldType, storage_types[output.id])
    sorted_dims = [dim for _, dim in get_sorted_dims(output_type.dims)]
    return (
        column_axis.value,
        sorted_dims.index(column_axis),
        output_type.dtype,
    )


class ItirToSDFG(eve.NodeVisitor):
    param_types: list[ts.TypeSpec]
    storage_types: dict[str, ts.TypeSpec]
    column_axis: Optional[Dimension]
    offset_provider: dict[str, Any]
    node_types: dict[int, next_typing.Type]
    unique_id: int
    use_gpu_storage: bool

    def __init__(
        self,
        param_types: list[ts.TypeSpec],
        offset_provider: dict[str, NeighborTableOffsetProvider],
        column_axis: Optional[Dimension] = None,
        use_gpu_storage: bool = False,
    ):
        self.param_types = param_types
        self.column_axis = column_axis
        self.offset_provider = offset_provider
        self.storage_types = {}
        self.use_gpu_storage = use_gpu_storage

    def add_storage(self, sdfg: dace.SDFG, name: str, type_: ts.TypeSpec, has_offset: bool = True):
        if isinstance(type_, ts.FieldType):
            shape, strides = new_array_symbols(name, len(type_.dims))
            offset = (
                [dace.symbol(unique_name(f"{name}_offset{i}_")) for i in range(len(type_.dims))]
                if has_offset
                else None
            )
            dtype = as_dace_type(type_.dtype)
            storage = (
                dace.dtypes.StorageType.GPU_Global
                if self.use_gpu_storage
                else dace.dtypes.StorageType.Default
            )
            sdfg.add_array(
                name, shape=shape, strides=strides, offset=offset, dtype=dtype, storage=storage
            )

        elif isinstance(type_, ts.ScalarType):
            sdfg.add_symbol(name, as_dace_type(type_))

        else:
            raise NotImplementedError()
        self.storage_types[name] = type_

    def get_output_nodes(
        self, closure: itir.StencilClosure, sdfg: dace.SDFG, state: dace.SDFGState
    ) -> dict[str, dace.nodes.AccessNode]:
        # Visit output node, which could be a `make_tuple` expression, to collect the required access nodes
        output_symbols_pass = GatherOutputSymbolsPass(sdfg, state, self.node_types)
        output_symbols_pass.visit(closure.output)
        # Visit output node again to generate the corresponding tasklet
        context = Context(sdfg, state, output_symbols_pass.symbol_refs)
        translator = PythonTaskletCodegen(self.offset_provider, context, self.node_types)
        output_nodes = flatten_list(translator.visit(closure.output))
        return {node.value.data: node.value for node in output_nodes}

    def visit_FencilDefinition(self, node: itir.FencilDefinition):
        program_sdfg = dace.SDFG(name=node.id)
        last_state = program_sdfg.add_state("program_entry", True)
        self.node_types = itir_typing.infer_all(node)

        # Filter neighbor tables from offset providers.
        neighbor_tables = filter_neighbor_tables(self.offset_provider)

        # Add program parameters as SDFG storages.
        for param, type_ in zip(node.params, self.param_types):
            self.add_storage(program_sdfg, str(param.id), type_)

        # Add connectivities as SDFG storages.
        for offset, table in neighbor_tables:
            scalar_kind = type_translation.get_scalar_kind(table.table.dtype)
            local_dim = Dimension("ElementDim", kind=DimensionKind.LOCAL)
            type_ = ts.FieldType([table.origin_axis, local_dim], ts.ScalarType(scalar_kind))
            self.add_storage(program_sdfg, connectivity_identifier(offset), type_, has_offset=False)

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
                name: create_memlet_full(name, program_sdfg.arrays[name]) for name in input_names
            }
            output_mapping = {
                name: create_memlet_full(name, program_sdfg.arrays[name]) for name in output_names
            }

            symbol_mapping = map_nested_sdfg_symbols(program_sdfg, closure_sdfg, input_mapping)

            # Insert the closure's SDFG as a nested SDFG of the program.
            nsdfg_node = last_state.add_nested_sdfg(
                sdfg=closure_sdfg,
                parent=program_sdfg,
                inputs=set(input_names),
                outputs=set(output_names),
                symbol_mapping=symbol_mapping,
            )

            # Add access nodes for the program parameters and connect them to the nested SDFG's inputs via edges.
            for inner_name, memlet in input_mapping.items():
                access_node = last_state.add_access(inner_name)
                last_state.add_edge(access_node, None, nsdfg_node, inner_name, memlet)

            for inner_name, memlet in output_mapping.items():
                access_node = last_state.add_access(inner_name)
                last_state.add_edge(nsdfg_node, inner_name, access_node, None, memlet)

        program_sdfg.validate()
        return program_sdfg

    def visit_StencilClosure(
        self, node: itir.StencilClosure, array_table: dict[str, dace.data.Array]
    ) -> tuple[dace.SDFG, list[str], list[str]]:
        assert ItirToSDFG._check_no_lifts(node)

        # Create the closure's nested SDFG and single state.
        closure_sdfg = dace.SDFG(name="closure")
        closure_state = closure_sdfg.add_state("closure_entry")
        closure_init_state = closure_sdfg.add_state_before(closure_state, "closure_init", True)

        input_names = [str(inp.id) for inp in node.inputs]
        neighbor_tables = filter_neighbor_tables(self.offset_provider)
        connectivity_names = [connectivity_identifier(offset) for offset, _ in neighbor_tables]

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
                    storage=array_table[name].storage,
                    transient=True,
                )
                closure_init_state.add_nedge(
                    closure_init_state.add_access(name),
                    closure_init_state.add_access(transient_name),
                    create_memlet_full(name, closure_sdfg.arrays[name]),
                )
                input_transients_mapping[name] = transient_name
            elif isinstance(self.storage_types[name], ts.FieldType):
                closure_sdfg.add_array(
                    name,
                    shape=array_table[name].shape,
                    strides=array_table[name].strides,
                    dtype=array_table[name].dtype,
                    storage=array_table[name].storage,
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
                dtype = as_dace_type(type_)
                closure_sdfg.add_symbol(name, dtype)
                if name in input_names:
                    out_name = unique_var_name()
                    closure_sdfg.add_scalar(out_name, dtype, transient=True)
                    out_tasklet = closure_init_state.add_tasklet(
                        f"get_{name}", {}, {"__result"}, f"__result = {name}"
                    )
                    access = closure_init_state.add_access(out_name)
                    value = ValueExpr(access, dtype)
                    memlet = dace.Memlet.simple(out_name, "0")
                    closure_init_state.add_edge(out_tasklet, "__result", access, None, memlet)
                    program_arg_syms[name] = value
                else:
                    program_arg_syms[name] = SymbolExpr(name, dtype)
        closure_ctx = Context(closure_sdfg, closure_state, program_arg_syms)
        closure_domain = self._visit_domain(node.domain, closure_ctx)

        # Map SDFG tasklet arguments to parameters
        input_local_names = [
            input_transients_mapping[input_name]
            if input_name in input_transients_mapping
            else input_name
            if input_name in input_field_names
            else cast(ValueExpr, program_arg_syms[input_name]).value.data
            for input_name in input_names
        ]
        input_memlets = [
            create_memlet_full(name, closure_sdfg.arrays[name])
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

            output_memlets = [
                create_memlet_at(
                    output_name,
                    tuple(
                        f"i_{dim}"
                        if f"i_{dim}" in map_ranges
                        else f"0:{closure_sdfg.arrays[output_name].shape[scan_dim_index]}"
                        for dim, _ in closure_domain
                    ),
                )
            ]
        else:
            nsdfg, map_ranges, results = self._visit_parallel_stencil_closure(
                node, closure_sdfg.arrays, closure_domain
            )

            output_subset = "0"

            output_memlets = [
                create_memlet_at(output_name, tuple(idx for idx in map_ranges.keys()))
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
        )
        access_nodes = {edge.data.data: edge.dst for edge in closure_state.out_edges(map_exit)}
        for edge in closure_state.in_edges(map_exit):
            memlet = edge.data
            if memlet.data not in output_connectors_mapping:
                continue
            transient_access = closure_state.add_access(memlet.data)
            closure_state.add_edge(
                nsdfg_node,
                edge.src_conn,
                transient_access,
                None,
                dace.Memlet.simple(memlet.data, output_subset),
            )
            inner_memlet = dace.Memlet.simple(
                memlet.data, output_subset, other_subset_str=memlet.subset
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
        is_forward, init_carry_value = get_scan_args(node.stencil)
        # select the scan dimension based on program argument for column axis
        assert self.column_axis
        assert isinstance(node.output, SymRef)
        scan_dim, scan_dim_index, scan_dtype = get_scan_dim(
            self.column_axis,
            self.storage_types,
            node.output,
        )

        assert isinstance(node.output, SymRef)
        neighbor_tables = filter_neighbor_tables(self.offset_provider)
        input_names = [str(inp.id) for inp in node.inputs]
        connectivity_names = [connectivity_identifier(offset) for offset, _ in neighbor_tables]

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

        # create a state machine for lambda call over the scan dimension
        start_state = scan_sdfg.add_state("start", True)
        lambda_state = scan_sdfg.add_state("lambda_compute")
        end_state = scan_sdfg.add_state("end")

        # the carry value of the scan operator exists only in the scope of the scan sdfg
        scan_carry_name = unique_var_name()
        scan_sdfg.add_scalar(scan_carry_name, dtype=as_dace_type(scan_dtype), transient=True)

        # tasklet for initialization of carry
        carry_init_tasklet = start_state.add_tasklet(
            "get_carry_init_value", {}, {"__result"}, f"__result = {init_carry_value}"
        )
        start_state.add_edge(
            carry_init_tasklet,
            "__result",
            start_state.add_access(scan_carry_name),
            None,
            dace.Memlet.simple(scan_carry_name, "0"),
        )

        # TODO(edopao): replace state machine with dace loop construct
        scan_sdfg.add_loop(
            start_state,
            lambda_state,
            end_state,
            loop_var=f"i_{scan_dim}",
            initialize_expr=f"{scan_lb_str}" if is_forward else f"{scan_ub_str} - 1",
            condition_expr=f"i_{scan_dim} < {scan_ub_str}"
            if is_forward
            else f"i_{scan_dim} >= {scan_lb_str}",
            increment_expr=f"i_{scan_dim} + 1" if is_forward else f"i_{scan_dim} - 1",
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
                    name, dtype=as_dace_type(cast(ts.ScalarType, self.storage_types[name]))
                )
        # add storage to scan SDFG for output
        scan_sdfg.add_array(
            output_name,
            shape=(array_table[node.output.id].shape[scan_dim_index],),
            strides=(array_table[node.output.id].strides[scan_dim_index],),
            offset=(array_table[node.output.id].offset[scan_dim_index],),
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
            self.offset_provider,
            lambda_domain,
            input_arrays,
            connectivity_arrays,
            self.node_types,
        )

        lambda_input_names = [name for name, _ in input_arrays]
        lambda_output_names = [connector.value.data for connector in lambda_outputs]

        input_memlets = [
            create_memlet_full(name, scan_sdfg.arrays[name]) for name in lambda_input_names
        ]
        connectivity_memlets = [
            create_memlet_full(name, scan_sdfg.arrays[name]) for name in connectivity_names
        ]
        input_mapping = {param: arg for param, arg in zip(lambda_input_names, input_memlets)}
        connectivity_mapping = {
            param: arg for param, arg in zip(connectivity_names, connectivity_memlets)
        }
        array_mapping = {**input_mapping, **connectivity_mapping}
        symbol_mapping = map_nested_sdfg_symbols(scan_sdfg, lambda_context.body, array_mapping)

        scan_inner_node = lambda_state.add_nested_sdfg(
            lambda_context.body,
            parent=scan_sdfg,
            inputs=set(lambda_input_names) | set(connectivity_names),
            outputs=set(lambda_output_names),
            symbol_mapping=symbol_mapping,
        )

        # connect scan SDFG to lambda inputs
        for name, memlet in array_mapping.items():
            access_node = lambda_state.add_access(name)
            lambda_state.add_edge(access_node, None, scan_inner_node, name, memlet)

        output_names = [output_name]
        assert len(lambda_output_names) == 1
        # connect lambda output to scan SDFG
        for name, connector in zip(output_names, lambda_output_names):
            lambda_state.add_edge(
                scan_inner_node,
                connector,
                lambda_state.add_access(name),
                None,
                dace.Memlet.simple(name, f"i_{scan_dim}"),
            )

        # add state to scan SDFG to update the carry value at each loop iteration
        lambda_update_state = scan_sdfg.add_state_after(lambda_state, "lambda_update")
        lambda_update_state.add_memlet_path(
            lambda_update_state.add_access(output_name),
            lambda_update_state.add_access(scan_carry_name),
            memlet=dace.Memlet.simple(output_names[0], f"i_{scan_dim}", other_subset_str="0"),
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
        neighbor_tables = filter_neighbor_tables(self.offset_provider)
        input_names = [str(inp.id) for inp in node.inputs]
        conn_names = [connectivity_identifier(offset) for offset, _ in neighbor_tables]

        # find the scan dimension, same as output dimension, and exclude it from the map domain
        map_ranges = {}
        for dim, (lb, ub) in closure_domain:
            lb_str = lb.value.data if isinstance(lb, ValueExpr) else lb.value
            ub_str = ub.value.data if isinstance(ub, ValueExpr) else ub.value
            map_ranges[f"i_{dim}"] = f"{lb_str}:{ub_str}"

        # Create an SDFG for the tasklet that computes a single item of the output domain.
        index_domain = {dim: f"i_{dim}" for dim, _ in closure_domain}

        input_arrays = [(name, self.storage_types[name]) for name in input_names]
        connectivity_arrays = [(array_table[name], name) for name in conn_names]

        context, results = closure_to_tasklet_sdfg(
            node,
            self.offset_provider,
            index_domain,
            input_arrays,
            connectivity_arrays,
            self.node_types,
        )

        return context.body, map_ranges, [r.value.data for r in results]

    def _visit_domain(
        self, node: itir.FunCall, context: Context
    ) -> tuple[tuple[str, tuple[ValueExpr, ValueExpr]], ...]:
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
            translator = PythonTaskletCodegen(self.offset_provider, context, self.node_types)
            lb = translator.visit(lower_bound)[0]
            ub = translator.visit(upper_bound)[0]
            bounds.append((dimension.value, (lb, ub)))

        return tuple(sorted(bounds, key=lambda item: item[0]))

    @staticmethod
    def _check_no_lifts(node: itir.StencilClosure):
        if any(
            getattr(fun, "id", "") == "lift"
            for fun in eve.walk_values(node).if_isinstance(itir.FunCall).getattr("fun")
        ):
            return False
        return True

    @staticmethod
    def _check_shift_offsets_are_literals(node: itir.StencilClosure):
        fun_calls = eve.walk_values(node).if_isinstance(itir.FunCall)
        shifts = [nd for nd in fun_calls if getattr(nd.fun, "id", "") == "shift"]
        for shift in shifts:
            if not all(isinstance(arg, (itir.Literal, itir.OffsetLiteral)) for arg in shift.args):
                return False
        return True
