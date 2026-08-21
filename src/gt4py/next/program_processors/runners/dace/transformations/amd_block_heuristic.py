# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""AMD GPU thread-block-size / loop-blocking heuristic for map-style kernels.

Shared by `GPUSetBlockSize` and `LoopBlocking`, enabled through their
`amd_heuristic` constructor argument, so that both passes agree on the same
`Config` for the same Map.

The horizontal/vertical axis of a Map is identified positionally, matching
the existing convention in `GPUSetBlockSize.apply`: the last Map parameter
(`params[-1]`) is the horizontal/`x` dimension, the second-to-last
(`params[-2]`) is the vertical/`y` dimension. The heuristic therefore only
applies to genuinely two-dimensional, non-degenerate Maps.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple, Optional

import dace
from dace.sdfg import nodes as dace_nodes


class Config(NamedTuple):
    block_x: int
    block_y: int
    hlb: int
    vlb: int

    def __str__(self) -> str:
        return f"TB2D[{self.block_x},{self.block_y},1]_HLB[{self.hlb}]_VLB[{self.vlb}]"


FLAT = Config(256, 1, 0, 2)  # all threads horizontal, shallow vertical blocking
HBLK = Config(128, 2, 2, 0)  # 2 horizontal elements per thread
VBLK = Config(128, 2, 0, 2)  # 2 vertical levels per thread  (robust default)
DEEP = Config(256, 1, 0, 4)  # 4 vertical levels per thread

# Crossover between the two regimes. Nothing was measured between 67k and
# 164k, so the band in between falls back to VBLK, which is the only single
# config that stays within 1.08 geomean on both sweeps.
SMALL_NHORIZ = 70_000
LARGE_NHORIZ = 100_000

TILE = 256  # horizontal threads per block in every candidate config


def select_block_config(
    n_vert: int,
    n_horiz: int,
    independent_input_bytes: int,
    total_input_bytes: int,
    ratio: float,
    tasklet_count: int,
) -> Config:
    """Pick thread-block shape + loop-blocking factor for a map-style kernel."""

    # --- regime 1: horizontal extent small enough that halving the number of
    # horizontal tiles pays for itself ---------------------------------------
    if n_horiz <= SMALL_NHORIZ:
        # blockDim.y must not exceed n_vert, or (1 - 1/y) of the block idles.
        return FLAT if n_vert <= 2 else HBLK

    # --- untested band: no measurements between the two sweeps --------------
    if n_horiz < LARGE_NHORIZ:
        return VBLK

    # --- regime 2: global grid ----------------------------------------------
    # Vertex fields (n_horiz = 10*4^k + 2) are not a multiple of the 256-wide
    # tile and are pathological for HLB=2: 2.0x-2.4x on all three measured
    # vertex kernels, where VLB=2 is instead the outright winner.
    if n_horiz % TILE:
        return VBLK

    # Cell/edge fields.  Kernels whose inputs are mostly shared between
    # neighbouring elements (low independent/total ratio) still want
    # horizontal blocking; kernels with much independent input per element
    # want a long vertical run per thread to keep loads in flight.
    if ratio < 0.30:
        return HBLK
    return DEEP


def _resolve_int(value: Any) -> Optional[int]:
    """Return `value` as a concrete `int`, or `None` if it is still symbolic."""
    if str(value).isdigit():
        return int(value)
    return None


def _gt_tasklet_map_multiplier(node: dace_nodes.Node, scope_dict: dict[Any, Any]) -> int:
    """Compute the product of the exact sizes of all Map scopes enclosing `node`."""
    multiplier = 1
    scope = scope_dict[node]
    while scope is not None:
        multiplier *= math.prod(scope.map.range.size_exact())
        scope = scope_dict[scope]
    return multiplier


def gt_count_weighted_tasklets(sdfg: dace.SDFG, _base_multiplier: int = 1) -> int:
    """Count the Tasklets in `sdfg`, weighting each by its enclosing Map sizes.

    A Tasklet nested inside one or more Maps is counted once per iteration that
    the enclosing Map(s) will perform, i.e. it is multiplied by the (exact)
    size of every Map scope that contains it. Nested SDFGs are processed
    recursively, carrying over the multiplier of their surrounding scope.

    Args:
        sdfg: The SDFG to inspect.
        _base_multiplier: Internal parameter used to propagate the multiplier
            of an enclosing scope into a nested SDFG.

    Returns:
        The total, Map-weighted number of Tasklets in `sdfg`.
    """
    total = 0
    for state in sdfg.states():
        scope_dict = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, dace_nodes.Tasklet):
                total += _base_multiplier * _gt_tasklet_map_multiplier(node, scope_dict)
            elif isinstance(node, dace_nodes.NestedSDFG):
                nested_multiplier = _base_multiplier * _gt_tasklet_map_multiplier(node, scope_dict)
                total += gt_count_weighted_tasklets(node.sdfg, nested_multiplier)
    return total


def gt_count_weighted_tasklets_in_map(state: dace.SDFGState, map_entry: dace_nodes.MapEntry) -> int:
    """Count the Tasklets inside `map_entry`, weighted by any nested Map sizes.

    Unlike `gt_count_weighted_tasklets()`, this only looks at the single Map
    `map_entry` (and any Nested SDFGs inside it) instead of the whole SDFG.
    Tasklets that are direct children of `map_entry` count once each; a
    Tasklet is only multiplied by a Map's (exact) size if it sits inside a
    further, inner Map nested within `map_entry` (applied recursively for
    multiple levels of nesting).

    Args:
        state: The state that contains `map_entry`.
        map_entry: The MapEntry whose scope should be inspected.

    Returns:
        The Map-weighted number of Tasklets inside `map_entry`.
    """
    scope_children = state.scope_children()
    total = 0
    for node in scope_children[map_entry]:
        if isinstance(node, dace_nodes.Tasklet):
            total += 1
        elif isinstance(node, dace_nodes.NestedSDFG):
            total += gt_count_weighted_tasklets(node.sdfg)
        elif isinstance(node, dace_nodes.MapEntry):
            inner_map_length = math.prod(node.map.range.size_exact())
            total += inner_map_length * gt_count_weighted_tasklets_in_map(state, node)
    return total


def _compute_input_bytes(
    map_entry: dace_nodes.MapEntry,
    state: dace.SDFGState,
    sdfg: dace.SDFG,
) -> tuple[int, int]:
    """Compute `(independent_input_bytes, total_input_bytes)` for `map_entry`.

    A byte contribution is "independent" if it does not depend on the Map's
    vertical iteration variable (`map_entry.map.params[0]`, per the
    positional horizontal/vertical convention used throughout this module).
    """
    independent_input_bytes = 0
    total_input_bytes = 0
    defining_param_name = map_entry.map.params[0]
    # Assume the compiler will optimize the code and load the same memory
    # address only once.
    seen_data: set[dace.Memlet] = set()
    for out_edge in state.out_edges(map_entry):
        if out_edge.data.src_subset is None:
            continue
        if out_edge.data in seen_data:
            continue
        seen_data.add(out_edge.data)
        out_edge_data_bytes = sdfg.arrays[out_edge.data.data].dtype.bytes
        if isinstance(out_edge.dst, dace_nodes.MapEntry):
            inner_map_length = math.prod(out_edge.dst.range.size_exact())
            total_input_bytes += inner_map_length * out_edge_data_bytes
        else:
            total_input_bytes += out_edge_data_bytes

        found_independent_input = True
        for edge_src_subset_range in out_edge.data.src_subset:
            start_of_range, end_of_range, _ = edge_src_subset_range
            if defining_param_name in str(start_of_range) or defining_param_name in str(
                end_of_range
            ):
                found_independent_input = False
                break
        if found_independent_input:
            if isinstance(out_edge.dst, dace_nodes.MapEntry):
                inner_map_length = math.prod(out_edge.dst.range.size_exact())
                independent_input_bytes += inner_map_length * out_edge_data_bytes
            elif (
                isinstance(out_edge.dst, dace_nodes.Tasklet) and out_edge.dst_conn == "__tlet_field"
            ):
                # Indirect access to the K level; not counted as independent input.
                pass
            else:
                independent_input_bytes += out_edge_data_bytes

    return independent_input_bytes, total_input_bytes


def compute_amd_block_config(
    map_entry: dace_nodes.MapEntry,
    state: dace.SDFGState,
    sdfg: dace.SDFG,
) -> Optional[Config]:
    """Compute the AMD thread-block/loop-blocking `Config` for `map_entry`.

    Returns `None` if `map_entry` is not a genuinely two-dimensional,
    non-degenerate Map, or if any of the required inputs cannot be resolved
    to a concrete `int` (e.g. because a Map range still contains a free
    symbol) -- callers should fall back to their non-heuristic behavior.
    """
    map_params = map_entry.map.params
    if len(map_params) != 2:
        return None

    map_size = map_entry.map.range.size_exact()
    n_vert = _resolve_int(map_size[0])
    n_horiz = _resolve_int(map_size[1])
    if n_vert is None or n_horiz is None or n_vert == 1 or n_horiz == 1:
        return None

    independent_input_bytes_raw, total_input_bytes_raw = _compute_input_bytes(
        map_entry, state, sdfg
    )
    independent_input_bytes = _resolve_int(independent_input_bytes_raw)
    total_input_bytes = _resolve_int(total_input_bytes_raw)
    if independent_input_bytes is None or not total_input_bytes:
        return None

    tasklet_count = _resolve_int(gt_count_weighted_tasklets_in_map(state, map_entry))
    if tasklet_count is None:
        return None

    ratio = independent_input_bytes / total_input_bytes

    return select_block_config(
        n_vert=n_vert,
        n_horiz=n_horiz,
        independent_input_bytes=independent_input_bytes,
        total_input_bytes=total_input_bytes,
        ratio=ratio,
        tasklet_count=tasklet_count,
    )
