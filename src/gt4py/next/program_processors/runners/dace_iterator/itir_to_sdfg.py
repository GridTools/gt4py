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

from typing import Any

import dace

import gt4py.eve as eve
from gt4py.next import type_inference as next_typing
from gt4py.next.common import Dimension, DimensionKind
from gt4py.next.iterator import ir as itir, type_inference as itir_typing
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider
from gt4py.next.type_system import type_specifications as ts, type_translation

from .itir_to_tasklet import (
    Context,
    IteratorExpr,
    PythonTaskletCodegen,
    ValueExpr,
    closure_to_tasklet_sdfg,
)
from .utility import (
    as_dace_type,
    connectivity_identifier,
    create_memlet_at,
    create_memlet_full,
    filter_neighbor_tables,
    map_nested_sdfg_symbols,
    unique_var_name,
)


class ItirToSDFG(eve.NodeVisitor):
    param_types: list[ts.TypeSpec]
    storages: dict[str, ts.TypeSpec]
    offset_provider: dict[str, Any]
    node_types: dict[int, next_typing.Type]
    unique_id: int

    def __init__(
        self,
        param_types: list[ts.TypeSpec],
        offset_provider: dict[str, NeighborTableOffsetProvider],
    ):
        self.param_types = param_types
        self.offset_provider = offset_provider
        self.storages = {}

    def add_storage(self, sdfg: dace.SDFG, name: str, type_: ts.TypeSpec):
        if isinstance(type_, ts.FieldType):
            shape = [dace.symbol(unique_var_name()) for _ in range(len(type_.dims))]
            strides = [dace.symbol(unique_var_name()) for _ in range(len(type_.dims))]
            dtype = as_dace_type(type_.dtype)
            sdfg.add_array(name, shape=shape, strides=strides, dtype=dtype)
        elif isinstance(type_, ts.ScalarType):
            sdfg.add_symbol(name, as_dace_type(type_))
        else:
            raise NotImplementedError()
        self.storages[name] = type_

    def visit_FencilDefinition(self, node: itir.FencilDefinition):
        program_sdfg = dace.SDFG(name=node.id)
        last_state = program_sdfg.add_state("program_entry")
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
            self.add_storage(program_sdfg, connectivity_identifier(offset), type_)

        # Create a nested SDFG for all stencil closures.
        for closure in node.closures:
            assert isinstance(closure.output, itir.SymRef)

            input_names = [str(inp.id) for inp in closure.inputs]
            connectivity_names = [connectivity_identifier(offset) for offset, _ in neighbor_tables]
            output_names = [str(closure.output.id)]

            # Translate the closure and its stencil's body to an SDFG.
            closure_sdfg = self.visit(closure, array_table=program_sdfg.arrays)

            # Create a new state for the closure.
            last_state = program_sdfg.add_state_after(last_state)

            # Add access nodes for the program parameters to the closure's state in the program.
            input_accesses = [last_state.add_access(name) for name in input_names]
            connectivity_accesses = [last_state.add_access(name) for name in connectivity_names]
            output_accesses = [last_state.add_access(name) for name in output_names]

            # Map symbols by matching outer and inner strides, shapes, while defaulting to same symbol
            symbol_mapping = {sym: sym for sym in closure_sdfg.free_symbols}
            for inner_name, access_node in zip(
                input_names + output_names, input_accesses + output_accesses
            ):
                outer_data = program_sdfg.arrays[access_node.data]
                inner_data = closure_sdfg.arrays[inner_name]
                for o_sh, i_sh in zip(outer_data.shape, inner_data.shape):
                    if str(i_sh) in closure_sdfg.free_symbols:
                        symbol_mapping[str(i_sh)] = o_sh
                for o_sh, i_sh in zip(outer_data.strides, inner_data.strides):
                    if str(i_sh) in closure_sdfg.free_symbols:
                        symbol_mapping[str(i_sh)] = o_sh

            # Insert the closure's SDFG as a nested SDFG of the program.
            nsdfg_node = last_state.add_nested_sdfg(
                sdfg=closure_sdfg,
                parent=None,
                inputs=set(input_names) | set(connectivity_names),
                outputs=set(output_names),
                symbol_mapping=symbol_mapping,
            )

            # Connect the access nodes to the nested SDFG's inputs via edges.
            for inner_name, access_node in zip(input_names, input_accesses):
                memlet = create_memlet_full(access_node.data, program_sdfg.arrays[access_node.data])
                last_state.add_edge(access_node, None, nsdfg_node, inner_name, memlet)

            for inner_name, access_node in zip(connectivity_names, connectivity_accesses):
                memlet = create_memlet_full(access_node.data, program_sdfg.arrays[access_node.data])
                last_state.add_edge(access_node, None, nsdfg_node, inner_name, memlet)

            for inner_name, access_node in zip(output_names, output_accesses):
                memlet = create_memlet_full(access_node.data, program_sdfg.arrays[access_node.data])
                last_state.add_edge(nsdfg_node, inner_name, access_node, None, memlet)

        program_sdfg.validate()
        return program_sdfg

    def visit_StencilClosure(
        self, node: itir.StencilClosure, array_table: dict[str, dace.data.Array]
    ) -> dace.SDFG:
        assert ItirToSDFG._check_no_lifts(node)
        assert ItirToSDFG._check_shift_offsets_are_literals(node)
        assert isinstance(node.output, itir.SymRef)

        neighbor_tables = filter_neighbor_tables(self.offset_provider)
        input_names = [str(inp.id) for inp in node.inputs]
        conn_names = [connectivity_identifier(offset) for offset, _ in neighbor_tables]
        output_names = [str(node.output.id)]

        # Create the closure's nested SDFG and single state.
        closure_sdfg = dace.SDFG(name="closure")
        closure_state = closure_sdfg.add_state("closure_entry")

        # Add DaCe arrays for inputs, output and connectivities to closure SDFG.
        for name in [*input_names, *conn_names, *output_names]:
            assert name not in closure_sdfg.arrays or (name in input_names and name in output_names)
            if name not in closure_sdfg.arrays:
                closure_sdfg.add_array(
                    name,
                    shape=array_table[name].shape,
                    strides=array_table[name].strides,
                    dtype=array_table[name].dtype,
                )

        # Get output domain of the closure
        program_arg_syms: dict[str, ValueExpr | IteratorExpr] = {}
        for name, type_ in self.storages.items():
            if isinstance(type_, ts.ScalarType):
                dtype = as_dace_type(type_)
                closure_sdfg.add_symbol(name, dtype)
                out_name = unique_var_name()
                closure_sdfg.add_scalar(out_name, dtype, transient=True)
                out_tasklet = closure_state.add_tasklet(
                    f"get_{name}", {}, {"__result"}, f"__result = {name}"
                )
                access = closure_state.add_access(out_name)
                value = ValueExpr(access, dtype)
                memlet = create_memlet_at(out_name, ("0",))
                closure_state.add_edge(out_tasklet, "__result", access, None, memlet)
                program_arg_syms[name] = value
        domain_ctx = Context(closure_sdfg, closure_state, program_arg_syms)
        closure_domain = self._visit_domain(node.domain, domain_ctx)
        map_domain = {
            f"i_{dim}": f"{lb.value.data}:{ub.value.data}" for dim, (lb, ub) in closure_domain
        }

        # Create an SDFG for the tasklet that computes a single item of the output domain.
        index_domain = {dim: f"i_{dim}" for dim, _ in closure_domain}

        input_arrays = [
            (closure_sdfg.arrays[name], name, self.storages[name]) for name in input_names
        ]
        conn_arrays = [(closure_sdfg.arrays[name], name) for name in conn_names]

        context, results = closure_to_tasklet_sdfg(
            node,
            self.offset_provider,
            index_domain,
            input_arrays,
            conn_arrays,
            self.node_types,
        )

        # Map SDFG tasklet arguments to parameters
        input_memlets = [
            create_memlet_full(name, closure_sdfg.arrays[name]) for name in input_names
        ]
        conn_memlet = [create_memlet_full(name, closure_sdfg.arrays[name]) for name in conn_names]
        output_memlets = []
        output_transient_names = {}
        for output_name in output_names:
            if output_name in input_names:
                # create and write to transient that is then copied back to actual output array to avoid aliasing of
                # same memory in nested SDFG with different names
                name = unique_var_name()
                output_transient_names[output_name] = name
                descriptor = closure_sdfg.arrays[output_name]
                closure_sdfg.add_array(
                    name,
                    shape=descriptor.shape,
                    strides=descriptor.strides,
                    dtype=descriptor.dtype,
                    transient=True,
                )
                memlet = create_memlet_at(name, tuple(idx for idx in map_domain.keys()))
            else:
                memlet = create_memlet_at(output_name, tuple(idx for idx in map_domain.keys()))
            output_memlets.append(memlet)

        input_mapping = {param: arg for param, arg in zip(input_names, input_memlets)}
        output_mapping = {param.value.data: arg for param, arg in zip(results, output_memlets)}
        conn_mapping = {param: arg for param, arg in zip(conn_names, conn_memlet)}

        array_mapping = {**input_mapping, **output_mapping, **conn_mapping}
        symbol_mapping = map_nested_sdfg_symbols(closure_sdfg, context.body, array_mapping)

        nsdfg_node, map_entry, map_exit = self._add_mapped_nested_sdfg(
            closure_state,
            sdfg=context.body,
            map_ranges=map_domain,
            inputs={**input_mapping, **conn_mapping},
            outputs=output_mapping,
            symbol_mapping=symbol_mapping,
            schedule=dace.ScheduleType.Sequential,
        )
        access_nodes = {edge.data.data: edge.dst for edge in closure_state.out_edges(map_exit)}
        for output_name, transient_name in output_transient_names.items():
            access_node = access_nodes[transient_name]
            in_edges = closure_state.in_edges(access_node)
            assert len(in_edges) == 1
            in_memlet = in_edges[0].data
            closure_state.add_edge(
                access_node,
                None,
                closure_state.add_access(output_name),
                None,
                dace.Memlet(
                    data=transient_name, subset=in_memlet.subset, other_subset=in_memlet.subset
                ),
            )

        for _, (lb, ub) in closure_domain:
            map_entry.add_in_connector(lb.value.data)
            map_entry.add_in_connector(ub.value.data)
            closure_state.add_edge(
                lb.value, None, map_entry, lb.value.data, create_memlet_at(lb.value.data, ("0",))
            )
            closure_state.add_edge(
                ub.value, None, map_entry, ub.value.data, create_memlet_at(ub.value.data, ("0",))
            )

        return closure_sdfg

    def _add_mapped_nested_sdfg(
        self,
        state: dace.SDFGState,
        map_ranges: dict[str, str | dace.subsets.Subset]
        | list[tuple[str, str | dace.subsets.Subset]],
        inputs: dict[str, dace.Memlet],
        outputs: dict[str, dace.Memlet],
        sdfg: dace.SDFG,
        symbol_mapping: dict[str, Any] | None = None,
        schedule: Any = dace.dtypes.ScheduleType.Default,
        unroll_map: bool = False,
        location: Any = None,
        debuginfo: Any = None,
        input_nodes: dict[str, dace.nodes.AccessNode] | None = None,
        output_nodes: dict[str, dace.nodes.AccessNode] | None = None,
    ) -> tuple[dace.nodes.NestedSDFG, dace.nodes.MapEntry, dace.nodes.MapExit]:
        if not symbol_mapping:
            symbol_mapping = {sym: sym for sym in sdfg.free_symbols}

        nsdfg_node = state.add_nested_sdfg(
            sdfg,
            None,
            set(inputs.keys()),
            set(outputs.keys()),
            symbol_mapping,
            name=sdfg.name,
            schedule=schedule,
            location=location,
            debuginfo=debuginfo,
        )

        map_entry, map_exit = state.add_map(
            f"{sdfg.name}_map", map_ranges, schedule, unroll_map, debuginfo
        )

        if input_nodes is None:
            input_nodes = {
                memlet.data: state.add_access(memlet.data) for name, memlet in inputs.items()
            }
        if output_nodes is None:
            output_nodes = {
                memlet.data: state.add_access(memlet.data) for name, memlet in outputs.items()
            }
        if not inputs:
            state.add_edge(map_entry, None, nsdfg_node, None, dace.Memlet())
        for name, memlet in inputs.items():
            state.add_memlet_path(
                input_nodes[memlet.data],
                map_entry,
                nsdfg_node,
                memlet=memlet,
                src_conn=None,
                dst_conn=name,
            )
        if not outputs:
            state.add_edge(nsdfg_node, None, map_exit, None, dace.Memlet())
        for name, memlet in outputs.items():
            state.add_memlet_path(
                nsdfg_node,
                map_exit,
                output_nodes[memlet.data],
                memlet=memlet,
                src_conn=name,
                dst_conn=None,
            )

        return nsdfg_node, map_entry, map_exit

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
