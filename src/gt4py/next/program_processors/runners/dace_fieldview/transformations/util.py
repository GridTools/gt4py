# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Common functionality for the transformations/optimization pipeline."""

from typing import Iterable, Union

import dace
from dace.sdfg import graph as dace_graph, nodes as dace_nodes


def is_nested_sdfg(
    sdfg: Union[dace.SDFG, dace.SDFGState, dace_nodes.NestedSDFG],
) -> bool:
    """Tests if `sdfg` is a NestedSDFG."""
    if isinstance(sdfg, dace.SDFGState):
        sdfg = sdfg.parent
    if isinstance(sdfg, dace_nodes.NestedSDFG):
        return True
    elif isinstance(sdfg, dace.SDFG):
        return sdfg.parent_nsdfg_node is not None
    raise TypeError(f"Does not know how to handle '{type(sdfg).__name__}'.")


def all_nodes_between(
    graph: dace.SDFG | dace.SDFGState,
    begin: dace_nodes.Node,
    end: dace_nodes.Node,
    reverse: bool = False,
) -> set[dace_nodes.Node] | None:
    """Find all nodes that are reachable from `begin` but bound by `end`.

    Essentially the function starts a DFS at `begin`. If an edge is found that lead
    to `end`, this edge is ignored. It will thus found any node that is reachable
    from `begin` by a path that does not involve `end`. The returned set will
    never contain `end` nor `begin`. In case `end` is never found the function
    will return `None`.

    If `reverse` is set to `True` the function will start exploring at `end` and
    follows the outgoing edges, i.e. the meaning of `end` and `begin` are swapped.

    Args:
        graph: The graph to operate on.
        begin: The start of the DFS.
        end: The terminator node of the DFS.
        reverse: Perform a backward DFS.

    Notes:
        - The returned set will also contain the nodes of path that starts at
            `begin` and ends at a node that is not `end`.
    """

    def next_nodes(node: dace_nodes.Node) -> Iterable[dace_nodes.Node]:
        return (
            (edge.src for edge in graph.in_edges(node))
            if reverse
            else (edge.dst for edge in graph.out_edges(node))
        )

    if reverse:
        begin, end = end, begin

    to_visit: list[dace_nodes.Node] = [begin]
    seen: set[dace_nodes.Node] = set()

    while len(to_visit) > 0:
        node: dace_nodes.Node = to_visit.pop()
        if node != end and node not in seen:
            to_visit.extend(next_nodes(node))
        seen.add(node)

    # If `end` was not found we have to return `None`  to indicate this.
    if end not in seen:
        return None

    # `begin` and `end` are not included in the output set.
    return seen - {begin, end}


def find_downstream_consumers(
    state: dace.SDFGState,
    begin: dace_nodes.Node | dace_graph.MultiConnectorEdge[dace.Memlet],
    only_tasklets: bool = False,
    reverse: bool = False,
) -> set[tuple[dace_nodes.Node, dace_graph.MultiConnectorEdge[dace.Memlet]]]:
    """Find all downstream connectors of `begin`.

    A consumer, in for this function, is any node that is neither an entry nor
    an exit node. The function returns a set of pairs, the first element is the
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
    else:
        to_visit = state.in_edges(begin) if reverse else state.out_edges(begin)

    seen: set[dace_graph.MultiConnectorEdge[dace.Memlet]] = set()
    found: set[tuple[dace_nodes.Node, dace_graph.MultiConnectorEdge[dace.Memlet]]] = set()

    while len(to_visit) > 0:
        curr_edge: dace_graph.MultiConnectorEdge[dace.Memlet] = to_visit.pop()
        next_node: dace_nodes.Node = curr_edge.src if reverse else curr_edge.dst

        if curr_edge in seen:
            continue
        seen.add(curr_edge)

        if isinstance(next_node, (dace_nodes.MapEntry, dace_nodes.MapExit)):
            if not reverse:
                # In forward mode a Map entry could also mean the definition of a
                #  dynamic map range.
                if isinstance(next_node, dace_nodes.MapEntry) and (
                    not curr_edge.dst_conn.startswith("IN_")
                ):
                    if not only_tasklets:
                        found.add((next_node, curr_edge))
                    continue
                target_conn = curr_edge.dst_conn[3:]
                new_edges = state.out_edges_by_connector(curr_edge.dst, "OUT_" + target_conn)
            else:
                target_conn = curr_edge.src_conn[4:]
                new_edges = state.in_edges_by_connector(curr_edge.src, "IN_" + target_conn)
            to_visit.extend(new_edges)

        elif isinstance(next_node, dace_nodes.Tasklet) or not only_tasklets:
            # We have found a consumer.
            found.add((next_node, curr_edge))

    return found


def find_upstream_producers(
    state: dace.SDFGState,
    begin: dace_nodes.Node | dace_graph.MultiConnectorEdge[dace.Memlet],
    only_tasklets: bool = False,
) -> set[tuple[dace_nodes.Node, dace_graph.MultiConnectorEdge[dace.Memlet]]]:
    """Same as `find_downstream_consumers()` but with `reverse` set to `True`."""
    return find_downstream_consumers(
        state=state,
        begin=begin,
        only_tasklets=only_tasklets,
        reverse=True,
    )
