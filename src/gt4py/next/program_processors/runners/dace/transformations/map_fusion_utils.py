# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import itertools
from typing import Optional

import dace
from dace import subsets as dace_subsets
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace.transformations import (
    splitting_tools as gtx_dace_split,
)


def copy_map_graph(
    sdfg: dace.SDFG,
    graph: dace.SDFGState,
    map_entry: dace_nodes.MapEntry,
    map_exit: dace_nodes.MapExit,
    suffix: Optional[str] = None,
) -> tuple[dace_nodes.MapEntry, dace_nodes.MapExit]:
    """Performs a full copy of the dataflow inside the Map.

    It will perform a deep copy of all the nodes between the given `map_entry`
    and `map_exit` nodes, including these two.

    Args:
        sdfg: The SDFG containing the map graph.
        graph: The SDFG state containing the map graph.
        map_entry: The entry node of the map graph.
        map_exit: The exit node of the map graph.
        suffix: String to append to the label of the copied nodes.

    Returns:
        A tuple of map entry and exit nodes, for the new map.
    """

    def _new_name(old_name: str) -> str:
        return old_name if suffix is None else f"{old_name}_{suffix}"

    new_nodes = {}
    new_data_names = {}
    new_data_descriptors = {}

    subgraph = graph.scope_subgraph(map_entry, include_entry=True, include_exit=True)
    map_nodes = subgraph.nodes()
    map_edges = subgraph.edges()

    new_map_entry = None
    new_map_exit = None

    for node in map_nodes:
        if isinstance(node, dace_nodes.AccessNode):
            data_name = node.data
            data_desc = node.desc(sdfg)
            if isinstance(data_desc, (dace.data.Array, dace.data.Scalar)):
                new_data_desc = data_desc.clone()
                assert data_desc.transient, "Only transient data descriptors can be"
                " copied otherwise global access nodes would require extra memory"
                " allocated from the outside."
                new_data_name = sdfg.add_datadesc(
                    _new_name(data_name), new_data_desc, find_new_name=True
                )
            else:
                raise ValueError(f"Unsupported data type: {type(data_desc)}")
            node_ = graph.add_access(new_data_name)
            new_data_names[data_name] = new_data_name
            new_data_descriptors[data_name] = new_data_desc
        elif isinstance(node, dace_nodes.NestedSDFG):
            node_ = graph.add_nested_sdfg(
                copy.deepcopy(node.sdfg),
                sdfg,
                node.in_connectors,
                node.out_connectors,
                node.symbol_mapping,
            )
            # ensure the correct reference to parent
            node_.sdfg.parent_nsdfg_node = node_
            node_.sdfg.parent = graph
        else:
            node_ = copy.deepcopy(node)
            # change label to a unique name
            if isinstance(node, (dace_nodes.MapEntry, dace_nodes.MapExit)):
                node_.map.label = _new_name(node.label)
            else:
                node_.label = _new_name(node.label)
            graph.add_node(node_)

        new_nodes[node] = node_
        if node == map_entry:
            new_map_entry = node_
        elif node == map_exit:
            new_map_exit = node_

    # we have to ensure that the exit node references the new map node
    assert new_map_entry is not None
    assert new_map_exit is not None
    new_map_exit.map = new_map_entry.map

    for edge in map_edges:
        copy_memlet = copy.deepcopy(edge.data)
        if edge.data.data in new_data_names:
            copy_memlet.data = new_data_names[edge.data.data]
        graph.add_edge(
            new_nodes[edge.src], edge.src_conn, new_nodes[edge.dst], edge.dst_conn, copy_memlet
        )

    for iedge in graph.in_edges(map_entry):
        copy_memlet = copy.deepcopy(iedge.data)
        graph.add_edge(iedge.src, iedge.src_conn, new_map_entry, iedge.dst_conn, copy_memlet)

    for oedge in graph.out_edges(map_exit):
        copy_memlet = copy.deepcopy(oedge.data)
        graph.add_edge(new_map_exit, oedge.src_conn, oedge.dst, oedge.dst_conn, copy_memlet)

    return new_map_entry, new_map_exit


def update_map_range(map_node: dace_nodes.Map, new_range: dace_subsets.Range) -> None:
    """Helper function to modify the range of a map.

    In a map graph the range is referenced in multiple nodes: the map entry,
    the exit node and the map itself. Therefore, we update the content of the list.

    Args:
        map_node: The map to modify.
        new_range: The range to set on the map.
    """
    for i, r in enumerate(new_range):
        map_node.range[i] = r


def copy_map_graph_with_new_range(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    map_entry: dace_nodes.MapEntry,
    map_exit: dace_nodes.MapExit,
    map_range: dace_subsets.Range,
    suffix: str,
) -> tuple[dace_nodes.MapEntry, dace_nodes.MapExit]:
    """Wrapper of `copy_map_graph` that additionally calls `update_map_range` on the new map.

    It will perform a full copy of the map scope, as described in `copy_map_graph`
    and additionally set the given range on the new map.

    Args:
        sdfg: The SDFG containing the map graph.
        graph: The SDFG state containing the map graph.
        map_entry: The entry node of the map graph.
        map_exit: The exit node of the map graph.
        map_range: The range to set on the new map.
        suffix: String to append to the label of the copied nodes.

    Returns:
        A tuple of map entry and exit nodes, for the new map.
    """
    new_map_entry, new_map_exit = copy_map_graph(sdfg, state, map_entry, map_exit, suffix)
    update_map_range(new_map_entry.map, map_range)
    assert new_map_entry.range == map_range
    assert new_map_exit.map.range == map_range
    return new_map_entry, new_map_exit


def delete_map(
    graph: dace.SDFGState, map_entry: dace_nodes.MapEntry, map_exit: dace_nodes.MapExit
) -> None:
    """Helper function to delete a map from a state graph.

    Args:
        graph: The SDFG state containing the map graph.
        map_entry: The entry node of the map graph.
        map_exit: The exit node of the map graph.
    """
    graph.remove_nodes_from(
        list(graph.scope_subgraph(map_entry, include_entry=True, include_exit=True).nodes())
    )


def split_overlapping_map_range(
    first_map: dace_nodes.Map,
    second_map: dace_nodes.Map,
) -> tuple[list[dace_subsets.Range], list[dace_subsets.Range]] | None:
    """Identifies the overlapping range of two maps and splits the range accordingly.

    For each map, the splitted range consists of multiple ranges, obtained from
    the combinatorial product of the splitted range for all map parameters.

    Args:
        first_map: One map (the order is not relevant).
        second_map: The other map.

    Returns:
        Two lists, each containing the ranges corresponding to the splitted range
        for the first and the second map, respectively.
    """
    first_map_params = set(first_map.params)
    second_map_params = set(second_map.params)
    if first_map_params != second_map_params:
        return None

    first_map_dict = dict(zip(first_map.params, first_map.range.ranges, strict=True))
    second_map_dict = dict(zip(second_map.params, second_map.range.ranges, strict=True))

    first_map_sorted_range = dace_subsets.Range(
        [first_map_dict[param] for param in sorted(first_map_params)]
    )
    second_map_sorted_range = dace_subsets.Range(
        [second_map_dict[param] for param in sorted(second_map_params)]
    )

    if gtx_dace_split.never_intersecting(first_map_sorted_range, second_map_sorted_range):
        return None
    if (first_map_sorted_range == second_map_sorted_range) == True:  # noqa: E712 [true-false-comparison]  # SymPy fuzzy bools.
        return None

    first_map_splitted_dict = {}
    second_map_splitted_dict = {}
    for param in first_map_params:
        first_map_range = first_map_dict[param]
        second_map_range = second_map_dict[param]
        if (step := first_map_range[2]) != second_map_range[2]:
            # we do not support splitting of map range when the range step is different
            return None
        elif first_map_range == second_map_range:
            first_map_splitted_dict[param] = [first_map_range]
            second_map_splitted_dict[param] = [second_map_range]
        else:
            try:
                overlap_range_start = max(first_map_range[0], second_map_range[0])
                overlap_range_stop = min(first_map_range[1], second_map_range[1])
            except TypeError:
                # cannot determine truth value of Relational
                # in case the ranges are defined with symbols we cannot determine the intersection
                return None

            def _split_range(
                range_: dace_subsets.Range, start: int, stop: int, step: int
            ) -> list[tuple[int, int, int]]:
                """Splits a range into sub-ranges based on the given start and stop.

                Given a range (range_start, range_stop, range_step), and an overlap defined by
                [start, stop], this function returns a list of sub-ranges that partition the original
                range into: the part before the overlap, the overlap itself, and the part after.

                Example:
                    If range_ = (0, 9, 1), start = 3, stop = 6, step = 1,
                    the result will be:
                        [(0, 2, 1), (3, 6, 1), (7, 9, 1)]
                """
                splitted_ranges = []
                if range_[0] < start:
                    splitted_ranges.append((range_[0], start - step, step))
                splitted_ranges.append((start, stop, step))
                if stop < range_[1]:
                    splitted_ranges.append((stop + step, range_[1], step))
                return splitted_ranges

            # split the ranges into sub-ranges based on the overlapping range
            first_map_splitted_dict[param] = _split_range(
                first_map_range, overlap_range_start, overlap_range_stop, step
            )
            second_map_splitted_dict[param] = _split_range(
                second_map_range, overlap_range_start, overlap_range_stop, step
            )

    first_map_combined_ranges = (first_map_splitted_dict[param] for param in first_map.params)
    second_map_combined_ranges = (second_map_splitted_dict[param] for param in second_map.params)

    first_map_range_combinations = [
        dace_subsets.Range(r) for r in itertools.product(*first_map_combined_ranges)
    ]
    second_map_range_combinations = [
        dace_subsets.Range(r) for r in itertools.product(*second_map_combined_ranges)
    ]

    return first_map_range_combinations, second_map_range_combinations
