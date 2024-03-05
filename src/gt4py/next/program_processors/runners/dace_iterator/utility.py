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
import itertools
from typing import Any, Optional, Sequence

import dace

from gt4py.next import Dimension
from gt4py.next.common import NeighborTable
from gt4py.next.iterator.ir import Node
from gt4py.next.type_system import type_specifications as ts


def dace_debuginfo(
    node: Node, debuginfo: Optional[dace.dtypes.DebugInfo] = None
) -> Optional[dace.dtypes.DebugInfo]:
    location = node.location
    if location:
        return dace.dtypes.DebugInfo(
            start_line=location.line,
            start_column=location.column if location.column else 0,
            end_line=location.end_line if location.end_line else -1,
            end_column=location.end_column if location.end_column else 0,
            filename=location.filename,
        )
    return debuginfo


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
    raise ValueError(f"Scalar type '{type_}' not supported.")


def as_scalar_type(typestr: str) -> ts.ScalarType:
    try:
        kind = getattr(ts.ScalarKind, typestr.upper())
    except AttributeError as ex:
        raise ValueError(f"Data type {typestr} not supported.") from ex
    return ts.ScalarType(kind)


def filter_neighbor_tables(offset_provider: dict[str, Any]):
    return {
        offset: table
        for offset, table in offset_provider.items()
        if isinstance(table, NeighborTable)
    }


def connectivity_identifier(name: str):
    return f"__connectivity_{name}"


def create_memlet_full(source_identifier: str, source_array: dace.data.Array):
    return dace.Memlet.from_array(source_identifier, source_array)


def create_memlet_at(source_identifier: str, index: tuple[str, ...]):
    subset = ", ".join(index)
    return dace.Memlet(data=source_identifier, subset=subset)


def get_sorted_dims(dims: Sequence[Dimension]) -> Sequence[tuple[int, Dimension]]:
    return sorted(enumerate(dims), key=lambda v: v[1].value)


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
            memlet.data: state.add_access(memlet.data, debuginfo=debuginfo)
            for name, memlet in inputs.items()
        }
    if output_nodes is None:
        output_nodes = {
            memlet.data: state.add_access(memlet.data, debuginfo=debuginfo)
            for name, memlet in outputs.items()
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


def unique_name(prefix):
    unique_id = getattr(unique_name, "_unique_id", 0)  # static variable
    setattr(unique_name, "_unique_id", unique_id + 1)  # noqa: B010 [set-attr-with-constant]

    return f"{prefix}_{unique_id}"


def unique_var_name():
    return unique_name("_var")


def new_array_symbols(name: str, ndim: int) -> tuple[list[dace.symbol], list[dace.symbol]]:
    dtype = dace.int64
    shape = [dace.symbol(unique_name(f"{name}_shape{i}"), dtype) for i in range(ndim)]
    strides = [dace.symbol(unique_name(f"{name}_stride{i}"), dtype) for i in range(ndim)]
    return shape, strides


def flatten_list(node_list: list[Any]) -> list[Any]:
    return list(
        itertools.chain.from_iterable([
            flatten_list(e) if e.__class__ == list else [e] for e in node_list
        ])
    )
