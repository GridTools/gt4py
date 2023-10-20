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
from gt4py.next.iterator.transforms import global_tmps
from gt4py.next.type_system import type_specifications as ts, type_translation

from .itir_to_tasklet import (
    Context,
    IteratorExpr,
    PythonTaskletCodegen,
    SymbolExpr,
    ValueExpr,
    closure_to_tasklet_sdfg,
    is_scan,
)
from .utility import (
    add_mapped_nested_sdfg,
    as_dace_type,
    as_scalar_type,
    connectivity_identifier,
    create_memlet_at,
    create_memlet_full,
    filter_neighbor_tables,
    flatten_list,
    get_sorted_dims,
    map_nested_sdfg_symbols,
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

    def __init__(
        self,
        param_types: list[ts.TypeSpec],
        offset_provider: dict[str, NeighborTableOffsetProvider],
        tmps: list[global_tmps.Temporary],
        column_axis: Optional[Dimension] = None,
    ):
        self.param_types = param_types
        self.column_axis = column_axis
        self.offset_provider = offset_provider
        self.storage_types = {}
        self.tmps = tmps

    def add_storage(self, sdfg: dace.SDFG, name: str, type_: ts.TypeSpec, has_offset: bool = True):
        if isinstance(type_, ts.FieldType):
            shape = [dace.symbol(unique_var_name()) for _ in range(len(type_.dims))]
            strides = [dace.symbol(unique_var_name()) for _ in range(len(type_.dims))]
            offset = (
                [dace.symbol(unique_var_name()) for _ in range(len(type_.dims))]
                if has_offset
                else None
            )
            dtype = as_dace_type(type_.dtype)
            sdfg.add_array(name, shape=shape, strides=strides, offset=offset, dtype=dtype)
        elif isinstance(type_, ts.ScalarType):
            sdfg.add_symbol(name, as_dace_type(type_))
        else:
            raise NotImplementedError()
        self.storage_types[name] = type_


    def generate_temporaries(
        self, node_params, defs_state: dace.SDFGState, program_sdfg: dace.SDFG
    ) -> dict[str, IteratorExpr | ValueExpr | SymbolExpr]:
        symbol_map: dict[str, IteratorExpr | ValueExpr | SymbolExpr] = {}
        for sym in node_params:
            if all([sym.id != tmp.id for tmp in self.tmps]) and sym.kind != "Iterator":
                name_ = str(sym.id)
                type_ = self.storage_types[name_]
                assert isinstance(type_, ts.ScalarType)
                symbol_map[name_] = SymbolExpr(name_, as_dace_type(type_))

        symbol_dtype = dace.int64
        tmp_symbols: dict[str, IteratorExpr | ValueExpr | SymbolExpr] = {}
        for tmp in self.tmps:
            tmp_name = str(tmp.id)
            assert isinstance(tmp.domain, itir.FunCall)
            self.node_types.update(itir_typing.infer_all(tmp.domain))
            domain_ctx = Context(program_sdfg, defs_state, symbol_map)
            tmp_domain = self._visit_domain(tmp.domain, domain_ctx)
            dims: list[Dimension] = []
            for dim, _ in tmp_domain:
                dims.append(
                    Dimension(
                        value=dim,
                        kind=(
                            DimensionKind.VERTICAL
                            if self.column_axis is not None and self.column_axis.value == dim
                            else DimensionKind.HORIZONTAL
                        ),
                    )
                )
            assert isinstance(tmp.dtype, str)
            type_ = ts.FieldType(dims=dims, dtype=as_scalar_type(tmp.dtype))
            self.add_storage(program_sdfg, tmp_name, type_)

            tmp_array = program_sdfg.arrays[tmp_name]
            tmp_array.transient = True

            stride_var = unique_var_name()
            program_sdfg.add_scalar(stride_var, symbol_dtype, transient=True)
            stride_node = defs_state.add_access(stride_var)
            defs_state.add_edge(
                defs_state.add_tasklet(
                    "stride",
                    code="__result = 1",
                    inputs={},
                    outputs={"__result"},
                ),
                "__result",
                stride_node,
                None,
                dace.Memlet.simple(stride_var, "0"),
            )
            for (_, (begin, end)), offset_sym, shape_sym, stride_sym in reversed(
                list(
                    zip(
                        tmp_domain,
                        tmp_array.offset,
                        tmp_array.shape,
                        tmp_array.strides,
                    )
                )
            ):
                offset_tasklet = defs_state.add_tasklet(
                    "offset",
                    code=f"__result = - {begin.value}",
                    inputs={},
                    outputs={"__result"},
                )
                offset_var = unique_var_name()
                program_sdfg.add_scalar(offset_var, symbol_dtype, transient=True)
                offset_node = defs_state.add_access(offset_var)
                defs_state.add_edge(
                    offset_tasklet,
                    "__result",
                    offset_node,
                    None,
                    dace.Memlet.simple(offset_var, "0"),
                )

                shape_tasklet = defs_state.add_tasklet(
                    "shape",
                    code=f"__result = {end.value} - {begin.value}",
                    inputs={},
                    outputs={"__result"},
                )
                shape_var = unique_var_name()
                program_sdfg.add_scalar(shape_var, symbol_dtype, transient=True)
                shape_node = defs_state.add_access(shape_var)
                defs_state.add_edge(
                    shape_tasklet,
                    "__result",
                    shape_node,
                    None,
                    dace.Memlet.simple(shape_var, "0"),
                )

                tmp_symbols[str(offset_sym)] = offset_var
                tmp_symbols[str(stride_sym)] = stride_var
                tmp_symbols[str(shape_sym)] = stride_var = shape_var

        return tmp_symbols

    def get_output_nodes(
        self, closure: itir.StencilClosure, context: Context
    ) -> dict[str, dace.nodes.AccessNode]:
        translator = PythonTaskletCodegen(self.offset_provider, context, self.node_types)
        output_nodes = flatten_list(translator.visit(closure.output))
        return {node.value.data: node.value for node in output_nodes}

    def visit_FencilDefinition(self, node: itir.FencilDefinition, **kargs):
        program_sdfg = dace.SDFG(name=node.id)
        entry_state = program_sdfg.add_state("program_entry")
        self.node_types = itir_typing.infer_all(node)

        # Filter neighbor tables from offset providers.
        neighbor_tables = filter_neighbor_tables(self.offset_provider)

        # Add program parameters as SDFG storages.
        for param, type_ in zip(node.params, self.param_types):
            self.add_storage(program_sdfg, str(param.id), type_)

        if self.tmps:
            tmp_symbols = self.generate_temporaries(node.params, entry_state, program_sdfg)

        # Add connectivities as SDFG storages.
        for offset, table in neighbor_tables:
            scalar_kind = type_translation.get_scalar_kind(table.table.dtype)
            local_dim = Dimension("ElementDim", kind=DimensionKind.LOCAL)
            type_ = ts.FieldType([table.origin_axis, local_dim], ts.ScalarType(scalar_kind))
            self.add_storage(program_sdfg, connectivity_identifier(offset), type_, has_offset=False)

        # Create a nested SDFG for all stencil closures.
        last_state = entry_state
        for closure in node.closures:
            ItirToSDFG._replace_ssa_identifiers(closure)

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

        if self.tmps:
            # on the first interstate edge define symbols for shape/stride/offsets of temporary arrays
            inter_state_edge = program_sdfg.out_edges(entry_state)[0]
            inter_state_edge.data.assignments.update(tmp_symbols)

        program_sdfg.validate()
        return program_sdfg

    def visit_StencilClosure(
        self, node: itir.StencilClosure, array_table: dict[str, dace.data.Array]
    ) -> tuple[dace.SDFG, list[str], list[str]]:
        assert ItirToSDFG._check_no_lifts(node)
        assert ItirToSDFG._check_shift_offsets_are_literals(node)

        # Create the closure's nested SDFG and single state.
        closure_sdfg = dace.SDFG(name="closure")
        closure_state = closure_sdfg.add_state("closure_entry")
        closure_init_state = closure_sdfg.add_state_before(closure_state, "closure_init")

        program_arg_syms: dict[str, ValueExpr | IteratorExpr | SymbolExpr] = {}
        closure_ctx = Context(closure_sdfg, closure_state, program_arg_syms)
        neighbor_tables = filter_neighbor_tables(self.offset_provider)

        input_names = [str(inp.id) for inp in node.inputs]
        conn_names = [connectivity_identifier(offset) for offset, _ in neighbor_tables]

        output_nodes = self.get_output_nodes(node, closure_ctx)
        output_names = [k for k, _ in output_nodes.items()]

        # Add DaCe arrays for inputs, outputs and connectivities to closure SDFG.
        input_transients_mapping = {}
        for name in [*input_names, *conn_names, *output_names]:
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
        for name, type_ in self.storage_types.items():
            if isinstance(type_, ts.ScalarType):
                if name in input_names:
                    dtype = as_dace_type(type_)
                    closure_sdfg.add_symbol(name, dtype)
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
                    program_arg_syms[name] = SymbolExpr(name, as_dace_type(type_))
        closure_domain = self._visit_domain(node.domain, closure_ctx)

        # Map SDFG tasklet arguments to parameters
        input_access_names = [
            input_transients_mapping[input_name]
            if input_name in input_transients_mapping
            else input_name
            if input_name in input_field_names
            else cast(ValueExpr, program_arg_syms[input_name]).value.data
            for input_name in input_names
        ]
        input_memlets = [
            create_memlet_full(name, closure_sdfg.arrays[name]) for name in input_access_names
        ]
        conn_memlets = [create_memlet_full(name, closure_sdfg.arrays[name]) for name in conn_names]

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

        input_mapping = {param: arg for param, arg in zip(input_names, input_memlets)}
        output_mapping = {param: arg_memlet for param, arg_memlet in zip(results, output_memlets)}
        conn_mapping = {param: arg for param, arg in zip(conn_names, conn_memlets)}

        array_mapping = {**input_mapping, **conn_mapping}
        symbol_mapping = map_nested_sdfg_symbols(closure_sdfg, nsdfg, array_mapping)

        nsdfg_node, map_entry, map_exit = add_mapped_nested_sdfg(
            closure_state,
            sdfg=nsdfg,
            map_ranges=map_ranges or {"__dummy": "0"},
            inputs=array_mapping,
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

        return closure_sdfg, input_field_names + conn_names, output_names

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
        start_state = scan_sdfg.add_state("start")
        lambda_state = scan_sdfg.add_state("lambda_compute")
        end_state = scan_sdfg.add_state("end")

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

        # add access nodes to SDFG for inputs
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

        connectivity_arrays = [(scan_sdfg.arrays[name], name) for name in connectivity_names]

        # implement the lambda closure as a nested SDFG that computes a single item of the map domain
        lambda_context, lambda_inputs, lambda_outputs = closure_to_tasklet_sdfg(
            node,
            self.offset_provider,
            {},
            [],
            connectivity_arrays,
            self.node_types,
        )

        connectivity_memlets = [
            create_memlet_full(name, scan_sdfg.arrays[name]) for name in connectivity_names
        ]
        connectivity_mapping = {
            param: arg for param, arg in zip(connectivity_names, connectivity_memlets)
        }

        lambda_input_names = [inner_name for inner_name, _ in lambda_inputs]
        symbol_mapping = map_nested_sdfg_symbols(
            scan_sdfg, lambda_context.body, connectivity_mapping
        )

        scan_inner_node = lambda_state.add_nested_sdfg(
            lambda_context.body,
            parent=scan_sdfg,
            inputs=set(lambda_input_names) | set(connectivity_names),
            outputs={connector.value.label for connector in lambda_outputs},
            symbol_mapping=symbol_mapping,
        )

        # the carry value of the scan operator exists in the scope of the scan sdfg
        scan_carry_name = unique_var_name()
        lambda_carry_name, _ = lambda_inputs[0]
        scan_sdfg.add_scalar(scan_carry_name, dtype=as_dace_type(scan_dtype), transient=True)

        carry_init_tasklet = start_state.add_tasklet(
            "get_carry_init_value", {}, {"__result"}, f"__result = {init_carry_value}"
        )
        carry_node1 = start_state.add_access(scan_carry_name)
        start_state.add_edge(
            carry_init_tasklet,
            "__result",
            carry_node1,
            None,
            dace.Memlet.simple(scan_carry_name, "0"),
        )

        carry_node2 = lambda_state.add_access(scan_carry_name)
        lambda_state.add_memlet_path(
            carry_node2,
            scan_inner_node,
            memlet=dace.Memlet.simple(scan_carry_name, "0"),
            src_conn=None,
            dst_conn=lambda_carry_name,
        )

        # connect access nodes to lambda inputs
        for (inner_name, _), data_name in zip(lambda_inputs[1:], input_names):
            if isinstance(self.storage_types[data_name], ts.FieldType):
                memlet = create_memlet_at(data_name, tuple(f"i_{dim}" for dim, _ in closure_domain))
            else:
                memlet = dace.Memlet.simple(data_name, "0")
            lambda_state.add_memlet_path(
                lambda_state.add_access(data_name),
                scan_inner_node,
                memlet=memlet,
                src_conn=None,
                dst_conn=inner_name,
            )

        for inner_name, memlet in connectivity_mapping.items():
            access_node = lambda_state.add_access(inner_name)
            lambda_state.add_memlet_path(
                access_node,
                scan_inner_node,
                memlet=memlet,
                src_conn=None,
                dst_conn=inner_name,
                propagate=True,
            )

        output_names = [output_name]
        assert len(lambda_outputs) == 1
        # connect lambda output to access node
        for lambda_connector, data_name in zip(lambda_outputs, output_names):
            scan_sdfg.add_array(
                data_name,
                shape=(array_table[node.output.id].shape[scan_dim_index],),
                strides=(array_table[node.output.id].strides[scan_dim_index],),
                offset=(array_table[node.output.id].offset[scan_dim_index],),
                dtype=array_table[node.output.id].dtype,
            )
            lambda_state.add_memlet_path(
                scan_inner_node,
                lambda_state.add_access(data_name),
                memlet=dace.Memlet.simple(data_name, f"i_{scan_dim}"),
                src_conn=lambda_connector.value.label,
                dst_conn=None,
            )

        # add state to scan SDFG to update the carry value at each loop iteration
        lambda_update_state = scan_sdfg.add_state_after(lambda_state, "lambda_update")
        result_node = lambda_update_state.add_access(output_names[0])
        carry_node3 = lambda_update_state.add_access(scan_carry_name)
        lambda_update_state.add_memlet_path(
            result_node,
            carry_node3,
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
        conn_arrays = [(array_table[name], name) for name in conn_names]

        context, _, results = closure_to_tasklet_sdfg(
            node,
            self.offset_provider,
            index_domain,
            input_arrays,
            conn_arrays,
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

    @staticmethod
    def _replace_ssa_identifiers(closure: itir.StencilClosure):
        _UNIQUE_NAME_SEPARATOR = "ᐞ"

        def __replace_in_expression(fun: itir.FunCall, p_old: str, p_new: str):
            for arg in fun.args:
                if isinstance(arg, itir.FunCall):
                    __replace_in_expression(arg, p_old, p_new)
                elif isinstance(arg, itir.SymRef) and arg.id == p_old:
                    arg.id = eve.SymbolRef(p_new)

        if isinstance(closure.stencil, itir.Lambda):
            for p in closure.stencil.params:
                p_new = ""
                p_old = str(p.id)
                match = p_old.split(_UNIQUE_NAME_SEPARATOR)
                p_new += match[0]
                for suffix in match[1:]:
                    p_new += f"{len(p_new)}_{suffix}"
                if p_new != p_old:
                    assert isinstance(closure.stencil.expr, itir.FunCall)
                    __replace_in_expression(closure.stencil.expr, p_old, p_new)
                    p.id = eve.SymbolName(p_new)
