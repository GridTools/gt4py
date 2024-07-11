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

"""Common functionality for the transformations."""

from typing import Iterable

import dace
from dace.sdfg import graph as dace_graph, nodes


def is_nested_sdfg(sdfg: dace.SDFG) -> bool:
    """Tests if `sdfg` is a neseted sdfg."""
    if isinstance(sdfg, dace.SDFGState):
        sdfg = sdfg.parent
    if isinstance(sdfg, dace.nodes.NestedSDFG):
        return True
    elif isinstance(sdfg, dace.SDFG):
        if sdfg.parent_nsdfg_node is not None:
            return True
        return False
    else:
        raise TypeError(f"Does not know how to handle '{type(sdfg).__name__}'.")


def all_nodes_between(
    graph: dace.SDFG | dace.SDFGState,
    begin: nodes.Node,
    end: nodes.Node,
    reverse: bool = False,
) -> set[nodes.Node] | None:
    """Returns all nodes that are reachable from `begin` but bound by `end`.

    What the function does is, that it starts a DFS starting at `begin`, which is
    not part of the returned set, every edge that goes to `end` will be considered
    to not exists.
    In case `end` is never found the function will return `None`.

    If `reverse` is set to `True` the function will start exploring at `end` and
    follows the outgoing edges, i.e. the meaning of `end` and `begin` are swapped.

    Args:
        graph: The graph to operate on.
        begin: The start of the DFS.
        end: The terminator node of the DFS.
        reverse: Perform a backward DFS.

    Notes:
        - The returned set will never contain the node `begin`.
        - The returned set will also contain the nodes of path that starts at
            `begin` and ends at a node that is not `end`.
    """

    def next_nodes(node: nodes.Node) -> Iterable[nodes.Node]:
        if reverse:
            return (edge.src for edge in graph.in_edges(node))
        return (edge.dst for edge in graph.out_edges(node))

    if reverse:
        begin, end = end, begin

    to_visit: list[nodes.Node] = [begin]
    seen: set[nodes.Node] = set()
    found_end: bool = False

    while len(to_visit) > 0:
        n: nodes.Node = to_visit.pop()
        if n == end:
            found_end = True
            continue
        elif n in seen:
            continue
        seen.add(n)
        to_visit.extend(next_nodes(n))

    if not found_end:
        return None

    seen.discard(begin)
    return seen


def find_downstream_consumers(
    state: dace.SDFGState,
    begin: nodes.Node | dace_graph.MultiConnectorEdge[dace.Memlet],
    only_tasklets: bool = False,
    reverse: bool = False,
) -> set[tuple[nodes.Node, dace_graph.MultiConnectorEdge[dace.Memlet]]]:
    """Find all downstream connectors of `begin`.

    A consumer, in this sense, is any node that is neither an entry nor an exit
    node. The function returns a set storing the pairs, the first element is the
    node that acts as consumer and the second is the edge that leads to it.
    By setting `only_tasklets` the nodes the function finds are only Tasklets.

    To find this set the function starts a search at `begin`, however, it is also
    possible to pass an edge as `begin`.
    If `reverse` is `True` the function essentially finds the producers that are
    upstream.

    Args:
        state: The state in which to look for the consumers.
        begin: The initial node that from which the search starts.
        only_tasklets: Return only Tasklets.
        reverse: Follow the reverse direction.
    """
    if isinstance(begin, dace_graph.MultiConnectorEdge):
        to_visit: list[dace_graph.MultiConnectorEdge[dace.Memlet]] = [begin]
    elif reverse:
        to_visit = list(state.in_edges(begin))
    else:
        to_visit = list(state.out_edges(begin))
    seen: set[dace_graph.MultiConnectorEdge[dace.Memlet]] = set()
    found: set[tuple[nodes.Node, dace_graph.MultiConnectorEdge[dace.Memlet]]] = set()

    while len(to_visit) != 0:
        curr_edge: dace_graph.MultiConnectorEdge[dace.Memlet] = to_visit.pop()
        next_node: nodes.Node = curr_edge.src if reverse else curr_edge.dst

        if curr_edge in seen:
            continue
        seen.add(curr_edge)

        if isinstance(next_node, (nodes.MapEntry, nodes.MapExit)):
            if reverse:
                target_conn = curr_edge.src_conn[4:]
                new_edges = state.in_edges_by_connector(curr_edge.src, "IN_" + target_conn)
            else:
                # In forward mode a Map entry could also mean the definition of a
                #  dynamic map range.
                if (not curr_edge.dst_conn.startswith("IN_")) and isinstance(
                    next_node, nodes.MapEntry
                ):
                    # This edge defines a dynamic map range, which is a consumer
                    if not only_tasklets:
                        found.add((next_node, curr_edge))
                    continue
                target_conn = curr_edge.dst_conn[3:]
                new_edges = state.out_edges_by_connector(curr_edge.dst, "OUT_" + target_conn)
            to_visit.extend(new_edges)
        else:
            if only_tasklets and (not isinstance(next_node, nodes.Tasklet)):
                continue
            found.add((next_node, curr_edge))

    return found


def find_upstream_producers(
    state: dace.SDFGState,
    begin: nodes.Node | dace_graph.MultiConnectorEdge[dace.Memlet],
    only_tasklets: bool = False,
) -> set[tuple[nodes.Node, dace_graph.MultiConnectorEdge[dace.Memlet]]]:
    """Same as `find_downstream_consumers()` but with `reverse` set to `True`."""
    return find_downstream_consumers(
        state=state,
        begin=begin,
        only_tasklets=only_tasklets,
        reverse=True,
    )
