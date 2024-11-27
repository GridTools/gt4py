# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from typing import Any

import dace

import gt4py.next.iterator.ir as itir
from gt4py import eve
from gt4py.next import common
from gt4py.next.ffront import fbuiltins as gtx_fbuiltins
from gt4py.next.program_processors.runners.dace_common import utility as dace_utils


def get_used_connectivities(
    node: itir.Node, offset_provider_type: common.OffsetProviderType
) -> dict[str, common.NeighborConnectivityType]:
    connectivities = dace_utils.filter_connectivity_types(offset_provider_type)
    offset_dims = set(eve.walk_values(node).if_isinstance(itir.OffsetLiteral).getattr("value"))
    return {offset: connectivities[offset] for offset in offset_dims if offset in connectivities}


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
    dtype = dace.dtype_to_typeclass(gtx_fbuiltins.IndexType)
    shape = [dace.symbol(dace_utils.field_size_symbol_name(name, i), dtype) for i in range(ndim)]
    strides = [
        dace.symbol(dace_utils.field_stride_symbol_name(name, i), dtype) for i in range(ndim)
    ]
    return shape, strides


def flatten_list(node_list: list[Any]) -> list[Any]:
    return list(
        itertools.chain.from_iterable(
            [flatten_list(e) if isinstance(e, list) else [e] for e in node_list]
        )
    )
