# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dace
from dace.transformation.dataflow import TrivialMapElimination
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.interstate import InlineTransients


class NoEmptyEdgeTrivialMapElimination(TrivialMapElimination):
    """Eliminate trivial maps like TrivialMapElimination, with additional conditions in can_be_applied."""

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if not super().can_be_applied(graph, expr_index, sdfg, permissive=permissive):
            return False

        map_entry = self.map_entry
        map_exit = graph.exit_node(map_entry)
        if map_entry.map.schedule not in {
            dace.ScheduleType.Sequential,
            dace.ScheduleType.CPU_Multicore,
        }:
            return False
        if any(
            edge.data.is_empty() for edge in (graph.in_edges(map_entry) + graph.out_edges(map_exit))
        ):
            return False
        return True


class InlineThreadLocalTransients(dace.transformation.SingleStateTransformation):
    """
    Inline and tile thread-local transients.

    Inlines transients like `dace.transformations.interstate.InlineTransients`, however only applies to OpenMP map
    scopes but also makes the resulting local arrays persistent and thread-local. This reproduces `cpu_kfirst`-style
    transient tiling.
    """

    map_entry = dace.transformation.transformation.PatternNode(dace.nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [dace.sdfg.utils.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry

        if not map_entry.schedule == dace.ScheduleType.CPU_Multicore:
            return False

        scope_subgraph = graph.scope_subgraph(map_entry, include_entry=False, include_exit=False)
        if len(scope_subgraph) > 1 or not isinstance(
            scope_subgraph.nodes()[0], dace.nodes.NestedSDFG
        ):
            return False

        candidates = InlineTransients._candidates(sdfg, graph, scope_subgraph.nodes()[0])
        return len(candidates) > 0

    def apply(self, graph, sdfg):
        map_entry = self.map_entry

        scope_subgraph = graph.scope_subgraph(map_entry, include_entry=False, include_exit=False)
        nsdfg_node = scope_subgraph.nodes()[0]
        candidates = InlineTransients._candidates(sdfg, graph, nsdfg_node)
        InlineTransients.apply_to(sdfg, nsdfg=nsdfg_node, save=False)
        for name in candidates:
            if name in sdfg.arrays:
                continue
            array: dace.data.Array = nsdfg_node.sdfg.arrays[name]
            shape = [dace.symbolic.overapproximate(s) for s in array.shape]
            strides = [1]
            total_size = shape[0]
            for s in reversed(shape[1:]):
                strides = [s * strides[0], *strides]
                total_size *= s
            array.shape = shape
            array.strides = strides
            array.total_size = total_size
            array.storage = dace.StorageType.CPU_ThreadLocal
            array.lifetime = dace.AllocationLifetime.Persistent


def nest_sequential_map_scopes(sdfg: dace.SDFG):
    """Nest map scopes of sequential maps.

    Nest scope subgraphs of sequential maps in NestedSDFG's to force eagerly offsetting pointers on each iteration, to
    avoid more complex pointer arithmetic on each Tasklet's invocation.
    This is performed in an inner-map-first order to avoid revisiting the graph after changes.
    """

    def _process_map(sdfg: dace.SDFG, state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
        for node in state.scope_children()[map_entry]:
            if isinstance(node, dace.nodes.NestedSDFG):
                nest_sequential_map_scopes(node.sdfg)
            elif isinstance(node, dace.nodes.MapEntry):
                _process_map(sdfg, state, node)
        if map_entry.schedule == dace.ScheduleType.Sequential:
            subgraph = state.scope_subgraph(map_entry, include_entry=False, include_exit=False)
            nest_state_subgraph(sdfg, state, subgraph)

    state: dace.SDFGState
    for state in sdfg.nodes():
        for map_entry in filter(
            lambda n: isinstance(n, dace.nodes.MapEntry), state.scope_children()[None]
        ):
            _process_map(sdfg, state, map_entry)
