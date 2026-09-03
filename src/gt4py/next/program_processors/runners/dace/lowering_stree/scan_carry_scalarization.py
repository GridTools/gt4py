# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Scalarize scan-carry transient arrays in per-map-point nested SDFGs.

``translate_scan`` allocates the scan-carry container for scalar leaves as a
1-D transient sized to the horizontal map extent — one slot per horizontal
map point — because at lowering time the scan loop and the horizontal map
are siblings in the schedule tree.  When ``stree.as_sdfg()`` materializes
the SDFG, the scan loop inside the horizontal ``MapScope`` becomes a nested
SDFG that is invoked once per map point; the carry array then lives inside
the nested SDFG but every access is indexed by exactly the enclosing map's
parameter symbol, so each invocation only ever touches a single slot of the
array.

This module implements an SDFG transformation, intended to run right after
``stree.as_sdfg()``, that replaces such a transient array by a transient
scalar, rewriting all its memlet subsets accordingly.  The match is purely
structural (no name coupling): any 1-D transient container that

- lives inside a nested SDFG placed directly inside a ``MapScope``,
- never escapes through the nested SDFG's connectors,
- is accessed only through scalar-point memlets that all use the very same
  subset expression,
- whose subset mentions at least one of the enclosing map's parameters, and
- whose subset does not mention any symbol that changes during a single
  invocation of the nested SDFG (loop variables, interstate assignments),

is shrunk to a scalar.  The last two conditions together are what makes the
rewrite sound: every access within one invocation resolves to the same
element.  Note that the subset also mentions the field origin (a free SDFG
symbol such as ``__out_IDim_range_0``), which is *not* a map parameter —
requiring the subset symbols to be a subset of the map parameters would
reject every real scan carry.  Beside reducing memory traffic per map invocation
(a variable-length heap-allocated array becomes register/stack storage), the
scalar form makes it impossible for horizontal map points to share carry
storage.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import dace
from dace import subsets as dace_subsets
from dace.sdfg import nodes as dace_nodes


@dataclass
class _CarryCandidate:
    """An internal transient of a nested SDFG that can be scalarized."""

    nsdfg: dace.SDFG
    name: str


def _scalar_point_accesses(nsdfg: dace.SDFG, name: str) -> list | None:
    """Return all memlets accessing ``name`` inside ``nsdfg``, or ``None`` if
    any access is not a single scalar point."""
    memlets = []
    for state in nsdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace_nodes.AccessNode) and node.data == name:
                for edge in state.all_edges(node):
                    if edge.data.data != name:
                        continue
                    if any(s != 1 for s in edge.data.subset.size()):
                        return None
                    memlets.append(edge.data)
    return memlets


def _symbols_varying_inside(nsdfg: dace.SDFG) -> set[str]:
    """Return the symbols whose value can change within one invocation of ``nsdfg``.

    These are the loop variables of the contained loop regions (in particular
    the scan column index) plus anything assigned on an interstate edge.  A
    container indexed by such a symbol addresses different elements over the
    invocation and must not be scalarized.
    """
    varying = set()
    for cfg in nsdfg.all_control_flow_regions(recursive=True):
        if isinstance(cfg, dace.sdfg.state.LoopRegion) and cfg.loop_variable:
            varying.add(str(cfg.loop_variable))
    for edge in nsdfg.all_interstate_edges(recursive=True):
        varying |= set(edge.data.assignments.keys())
    return varying


def find_carry_candidates(top_sdfg: dace.SDFG) -> list[_CarryCandidate]:
    candidates = []
    for state in top_sdfg.all_states():
        scope_dict = state.scope_dict()
        for node in state.nodes():
            if not isinstance(node, dace_nodes.NestedSDFG):
                continue
            entry = scope_dict[node]
            if entry is None or not isinstance(entry, dace_nodes.MapEntry):
                continue  # nested SDFG not directly inside a map scope
            map_symbols = set(entry.map.params)
            nsdfg = node.sdfg
            escaped = set(node.in_connectors) | set(node.out_connectors)
            for name, arr in nsdfg.arrays.items():
                if not arr.transient or name in escaped:
                    continue  # must be internal storage of the nested SDFG
                if len(arr.shape) != 1:
                    continue
                accesses = _scalar_point_accesses(nsdfg, name)
                if not accesses:
                    continue
                if len({str(memlet.subset) for memlet in accesses}) != 1:
                    continue  # accesses do not all address the same element
                syms = set()
                for memlet in accesses:
                    syms |= {str(s) for s in memlet.subset.free_symbols}
                if not (syms & map_symbols):
                    continue  # not a per-map-point slot
                if syms & _symbols_varying_inside(nsdfg):
                    continue  # index varies within one invocation
                candidates.append(_CarryCandidate(nsdfg=nsdfg, name=name))
    return candidates


def scalarize_scan_carries(top_sdfg: dace.SDFG) -> set[str]:
    """Shrink matching scan-carry transients to scalars, in place.

    Returns the names of the scalarized containers.
    """
    scalarized = set()
    for cand in find_carry_candidates(top_sdfg):
        nsdfg, name = cand.nsdfg, cand.name
        for state in nsdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace_nodes.AccessNode) and node.data == name:
                    for edge in state.all_edges(node):
                        if edge.data.data != name:
                            continue
                        edge.data.subset = dace_subsets.Range.from_string(
                            ",".join("0" for _ in edge.data.subset.ranges)
                        )
        scalar = dace.data.Scalar(nsdfg.arrays[name].dtype, transient=True)
        nsdfg.remove_data(name, validate=False)
        nsdfg.add_datadesc(name, copy.deepcopy(scalar), find_new_name=False)
        scalarized.add(name)
    return scalarized
