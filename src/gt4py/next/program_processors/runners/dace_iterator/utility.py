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

from gt4py.next.iterator.embedded import NeighborTableOffsetProvider
from gt4py.next.type_system import type_specifications as ts


def as_dace_type(type_: ts.ScalarType):
    if type_.kind == ts.ScalarKind.BOOL:
        return dace.bool_
    elif type_.kind == ts.ScalarKind.INT32:
        return dace.int32
    elif type_.kind == ts.ScalarKind.INT64:
        return dace.int64
    elif type_.kind == ts.ScalarKind.FLOAT32:
        return dace.float32
    elif type_.kind == ts.ScalarKind.FLOAT64:
        return dace.float64
    raise ValueError(f"scalar type {type_} not supported")


def filter_neighbor_tables(offset_provider: dict[str, Any]):
    return [
        (offset, table)
        for offset, table in offset_provider.items()
        if isinstance(table, NeighborTableOffsetProvider)
    ]


def connectivity_identifier(name: str):
    return f"__connectivity_{name}"


def create_memlet_full(source_identifier: str, source_array: dace.data.Array):
    bounds = [(0, size) for size in source_array.shape]
    subset = ", ".join(f"{lb}:{ub}" for lb, ub in bounds)
    return dace.Memlet(data=source_identifier, subset=subset)


def create_memlet_at(source_identifier: str, index: tuple[str, ...]):
    subset = ", ".join(index)
    return dace.Memlet(data=source_identifier, subset=subset)


def map_nested_sdfg_symbols(
    parent_sdfg: dace.SDFG, nested_sdfg: dace.SDFG, array_mapping: dict[str, dace.Memlet]
) -> dict[str, str]:
    symbol_mapping: dict[str, str] = {}
    for param, arg in array_mapping.items():
        arg_array = parent_sdfg.arrays[arg.data]
        param_array = nested_sdfg.arrays[param]
        if not isinstance(param_array, dace.data.Scalar):
            assert len(arg.subset.size()) == len(param_array.shape)
            for arg_shape, param_shape in zip(arg.subset.size(), param_array.shape):
                if isinstance(param_shape, dace.symbol):
                    symbol_mapping[str(param_shape)] = str(arg_shape)
            assert len(arg_array.strides) == len(param_array.strides)
            for arg_stride, param_stride in zip(arg_array.strides, param_array.strides):
                if isinstance(param_stride, dace.symbol):
                    symbol_mapping[str(param_stride)] = str(arg_stride)
        else:
            assert arg.subset.num_elements() == 1
    for sym in nested_sdfg.free_symbols:
        if str(sym) not in symbol_mapping:
            symbol_mapping[str(sym)] = str(sym)
    return symbol_mapping


def add_mapped_nested_sdfg(
    state: dace.SDFGState,
    map_ranges: dict[str, str | dace.subsets.Subset] | list[tuple[str, str | dace.subsets.Subset]],
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
            propagate=True,
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
            propagate=True,
        )

    return nsdfg_node, map_entry, map_exit


_unique_id = 0


def unique_name(prefix):
    global _unique_id
    _unique_id += 1
    return f"{prefix}_{_unique_id}"


def unique_var_name():
    return unique_name("__var")
