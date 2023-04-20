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
import sympy

import gt4py.eve as eve
from gt4py.next.common import Dimension, DimensionKind
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator import type_inference as itir_typing
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider
from gt4py.next.type_system import type_specifications as ts, type_translation

from .itir_to_tasklet import closure_to_tasklet_sdfg, Context
from .utility import (
    connectivity_identifier,
    create_memlet_at,
    create_memlet_full,
    filter_neighbor_tables,
    as_dace_type,
    map_nested_sdfg_symbols,
)


class ItirToSDFG(eve.NodeVisitor):
    param_types: list[ts.TypeSpec]
    storages: dict[str, ts.TypeSpec]
    offset_provider: dict[str, Any]
    node_types: dict[int, itir_typing.Type]

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
            shape = [dace.symbol(sdfg.temp_data_name()) for _ in range(len(type_.dims))]
            strides = [dace.symbol(sdfg.temp_data_name()) for _ in range(len(type_.dims))]
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

            # Insert the closure's SDFG as a nested SDFG of the program.
            nsdfg_node = last_state.add_nested_sdfg(
                sdfg=closure_sdfg,
                parent=None,
                inputs=set(input_names) | set(connectivity_names),
                outputs=set(output_names),
                symbol_mapping={sym: sym for sym in program_sdfg.free_symbols},
            )

            # Add access nodes for the program parameters to the closure's state in the program.
            input_accesses = [last_state.add_access(name) for name in input_names]
            connectivity_accesses = [last_state.add_access(name) for name in connectivity_names]
            output_accesses = [last_state.add_access(name) for name in output_names]

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
            closure_sdfg.add_array(
                name,
                shape=array_table[name].shape,
                strides=array_table[name].strides,
                dtype=array_table[name].dtype,
            )

        # Get output domain of the closure
        closure_domain = self._visit_domain(node.domain)
        map_domain = {f"i_{dim}": f"{lb}:{ub}" for dim, (lb, ub) in closure_domain}

        # Create an SDFG for the tasklet that computes a single item of the output domain.
        index_domain = {dim: f"i_{dim}" for dim, _ in closure_domain}

        input_arrays = [(closure_sdfg.arrays[name], name, self.storages[name]) for name in input_names]
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
        input_memlets = [create_memlet_full(name, closure_sdfg.arrays[name]) for name in input_names]
        conn_memlet = [create_memlet_full(name, closure_sdfg.arrays[name]) for name in conn_names]
        output_memlets = [
            create_memlet_at(name, tuple(idx for idx in map_domain.keys())) for name in output_names
        ]

        input_mapping = {param: arg for param, arg in zip(input_names, input_memlets)}
        output_mapping = {param.value.data: arg for param, arg in zip(results, output_memlets)}
        conn_mapping = {param: arg for param, arg in zip(conn_names, conn_memlet)}

        array_mapping = {**input_mapping, **output_mapping, **conn_mapping}
        symbol_mapping = map_nested_sdfg_symbols(closure_sdfg, context.body, array_mapping)

        self._add_mapped_nested_sdfg(
            closure_state,
            sdfg=context.body,
            map_ranges=map_domain,
            inputs={**input_mapping, **conn_mapping},
            outputs=output_mapping,
            symbol_mapping=symbol_mapping,
            schedule=dace.ScheduleType.Sequential,
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
        for name, memlet in inputs.items():
            state.add_memlet_path(
                input_nodes[memlet.data],
                map_entry,
                nsdfg_node,
                memlet=memlet,
                src_conn=None,
                dst_conn=name,
            )
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
        self, node: itir.FunCall
    ) -> tuple[tuple[str, tuple[sympy.Basic, sympy.Basic]], ...]:
        assert isinstance(node.fun, itir.SymRef)
        assert node.fun.id == "cartesian_domain" or node.fun.id == "unstructured_domain"

        bounds: list[tuple[str, tuple[sympy.Basic, sympy.Basic]]] = []

        for named_range in node.args:
            assert isinstance(named_range, itir.FunCall)
            assert isinstance(named_range.fun, itir.SymRef)
            assert len(named_range.args) == 3
            dimension = named_range.args[0]
            assert isinstance(dimension, itir.AxisLiteral)
            lower_bound = named_range.args[1]
            upper_bound = named_range.args[2]
            sym_lower_bound = dace.symbolic.pystr_to_symbolic(str(lower_bound))
            sym_upper_bound = dace.symbolic.pystr_to_symbolic(str(upper_bound))
            bounds.append((dimension.value, (sym_lower_bound, sym_upper_bound)))

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
