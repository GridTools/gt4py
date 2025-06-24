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
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import dace
from dace import subsets as dace_subsets, symbolic
from dace.sdfg import graph, nodes as dace_nodes, validation
from dace.transformation import helpers

from gt4py.next.program_processors.runners.dace.transformations import spliting_tools as gtx_st

def find_parameter_remapping(
    first_map: dace_nodes.Map, second_map: dace_nodes.Map
) -> Optional[Dict[str, str]]:
    """Computes the parameter remapping for the parameters of the _second_ map.

    The returned `dict` maps the parameters of the second map (keys) to parameter
    names of the first map (values). Because of how the replace function works
    the `dict` describes how to replace the parameters of the second map
    with parameters of the first map.
    Parameters that already have the correct name and compatible range, are not
    included in the return value, thus the keys and values are always different.
    If no renaming at is _needed_, i.e. all parameter have the same name and range,
    then the function returns an empty `dict`.
    If no remapping exists, then the function will return `None`.

    :param first_map: The first map (these parameters will be replaced).
    :param second_map: The second map, these parameters acts as source.

    :note: This function currently fails if the renaming is not unique. Consider the
        case were the first map has the structure `for i, j in map[0:20, 0:20]` and it
        writes `T[i, j]`, while the second map is equivalent to
        `for l, k in map[0:20, 0:20]` which reads `T[l, k]`. For this case we have
        the following valid remappings `{l: i, k: j}` and `{l: j, k: i}` but
        only the first one allows to fuse the map. This is because if the second
        one is used the second map will read `T[j, i]` which leads to a data
        dependency that can not be satisfied.
        To avoid this issue the renaming algorithm will process them in order, i.e.
        assuming that the order of the parameters in the map matches. But this is
        not perfect, the only way to really solve this is by trying possible
        remappings. At least the algorithm used here is deterministic.
    """

    # The parameter names
    first_params: List[str] = first_map.params
    second_params: List[str] = second_map.params

    if len(first_params) != len(second_params):
        return None

    # The ranges, however, we apply some post processing to them.
    simp = lambda e: symbolic.simplify_ext(symbolic.simplify(e))  # noqa: E731 [lambda-assignment]
    first_rngs: Dict[str, Tuple[Any, Any, Any]] = {
        param: tuple(simp(r) for r in rng) for param, rng in zip(first_params, first_map.range)
    }
    second_rngs: Dict[str, Tuple[Any, Any, Any]] = {
        param: tuple(simp(r) for r in rng)
        for param, rng in zip(second_params, second_map.range)
    }

    # Parameters of the second map that have not yet been matched to a parameter
    #  of the first map and the parameters of the first map that are still free.
    #  That we use a `list` instead of a `set` is intentional, because it counter
    #  acts the issue that is described in the doc string. Using a list ensures
    #  that they indexes are matched in order. This assume that in real world
    #  code the order of the loop is not arbitrary but kind of matches.
    unmapped_second_params: List[str] = list(second_params)
    unused_first_params: List[str] = list(first_params)

    # This is the result (`second_param -> first_param`), note that if no renaming
    #  is needed then the parameter is not present in the mapping.
    final_mapping: Dict[str, str] = {}

    # First we identify the parameters that already have the correct name.
    for param in set(first_params).intersection(second_params):
        first_rng = first_rngs[param]
        second_rng = second_rngs[param]

        if first_rng == second_rng:
            # They have the same name and the same range, this is already a match.
            #  Because the names are already the same, we do not have to enter them
            #  in the `final_mapping`
            unmapped_second_params.remove(param)
            unused_first_params.remove(param)

    # Check if no remapping is needed.
    if len(unmapped_second_params) == 0:
        return {}

    # Now we go through all the parameters that we have not mapped yet.
    #  All of them will result in a remapping.
    for unmapped_second_param in unmapped_second_params:
        second_rng = second_rngs[unmapped_second_param]
        assert unmapped_second_param not in final_mapping

        # Now look in all not yet used parameters of the first map which to use.
        for candidate_param in list(unused_first_params):
            candidate_rng = first_rngs[candidate_param]
            if candidate_rng == second_rng:
                final_mapping[unmapped_second_param] = candidate_param
                unused_first_params.remove(candidate_param)
                break
        else:
            # We did not find a candidate, so the remapping does not exist
            return None

    assert len(unused_first_params) == 0
    assert len(final_mapping) == len(unmapped_second_params)
    return final_mapping


def rename_map_parameters(
    first_map: dace_nodes.Map,
    second_map: dace_nodes.Map,
    second_map_entry: dace_nodes.MapEntry,
    state: dace.SDFGState,
    ) -> None:
    """Replaces the map parameters of the second map with names from the first.

    The replacement is done in a safe way, thus `{'i': 'j', 'j': 'i'}` is
    handled correct. The function assumes that a proper replacement exists.
    The replacement is computed by calling `find_parameter_remapping()`.

    :param first_map:  The first map (these are the final parameter).
    :param second_map: The second map, this map will be replaced.
    :param second_map_entry: The entry node of the second map.
    :param state: The SDFGState on which we operate.
    """
    # Compute the replacement dict.
    repl_dict: Dict[str, str] = find_parameter_remapping(  # type: ignore[assignment]  # Guaranteed to be not `None`.
        first_map=first_map, second_map=second_map
    )

    if repl_dict is None:
        raise RuntimeError("The replacement does not exist")
    if len(repl_dict) == 0:
        return

    second_map_scope = state.scope_subgraph(entry_node=second_map_entry)
    # Why is this thing is symbolic and not in replace?
    symbolic.safe_replace(
        mapping=repl_dict,
        replace_callback=second_map_scope.replace_dict,
    )

    # For some odd reason the replace function does not modify the range and
    #  parameter of the map, so we will do it the hard way.
    second_map.params = copy.deepcopy(first_map.params)
    second_map.range = copy.deepcopy(first_map.range)

def get_new_conn_name(
    edge_to_move: graph.MultiConnectorEdge[dace.Memlet],
    to_node: Union[dace_nodes.MapExit, dace_nodes.MapEntry],
    state: dace.SDFGState,
    scope_dict: Dict,
    never_consolidate_edges: bool = False,
    consolidate_edges_only_if_not_extending: bool = True,
) -> Tuple[str, bool]:
    """Determine the new connector name that should be used.

    The function returns a pair. The first element is the name of the connector
    name that should be used. The second element is a boolean that indicates if
    the connector name is already present on `to_node`, `True`, or if a new
    connector was created.

    The function honors the `self.never_consolidate_edges`, in which case
    a new connector is generated every time, leading to minimal subset but
    many connections. Furthermore, it will also consider
    `self.consolidate_edges_only_if_not_extending`. If it is set it will only
    create a new connection if this would lead to an increased subset.

    :note: In case `to_node` a MapExit or a nested map, the function will always
        generate a new connector.
    """
    assert edge_to_move.dst_conn.startswith("IN_")
    old_conn = edge_to_move.dst_conn[3:]

    # If we have a MapExit or have a nested Map we never consolidate or if
    #  especially requested.
    if (
        isinstance(to_node, dace_nodes.MapExit)
        or scope_dict[to_node] is not None
        or never_consolidate_edges
    ):
        return to_node.next_connector(old_conn), False

    # Now look for an edge that already referees to the data of the edge.
    edge_that_is_already_present = None
    for iedge in state.in_edges(to_node):
        if iedge.data.is_empty() or iedge.dst_conn is None:
            continue
        if not iedge.dst_conn.startswith("IN_"):
            continue
        if iedge.data.data == edge_to_move.data.data:
            # The same data is used so we reuse that connection.
            edge_that_is_already_present = iedge

    # No edge is there that is using the data, so create a new connector.
    #  TODO(phimuell): Probably should reuse the connector at `from_node`?
    if edge_that_is_already_present is None:
        return to_node.next_connector(old_conn), False

    # We also do not care if the consolidation leads to the extension of the
    #  subsets, thus we are done.
    if not consolidate_edges_only_if_not_extending:
        return edge_that_is_already_present.dst_conn[3:], True

    # We can only do the check for extension if both have a valid subset.
    edge_to_move_subset = edge_to_move.data.src_subset
    edge_that_is_already_present_subset = edge_that_is_already_present.data.src_subset
    if edge_to_move_subset is None or edge_that_is_already_present_subset is None:
        return to_node.next_connector(old_conn), False

    # The consolidation will not lead to an extension if either the edge that is
    #  there or the new edge covers each other.
    # NOTE: One could also say that we should only do that if `edge_that_is_already_there`
    #   covers the new one, but since the order, is kind of arbitrary, we test if
    #   either one covers.
    return (
        (edge_that_is_already_present.dst_conn[3:], True)
        if edge_that_is_already_present_subset.covers(edge_to_move_subset)
        or edge_to_move_subset.covers(edge_that_is_already_present_subset)
        else (to_node.next_connector(old_conn), False)
    )

def relocate_nodes(
    from_node: Union[dace_nodes.MapExit, dace_nodes.MapEntry],
    to_node: Union[dace_nodes.MapExit, dace_nodes.MapEntry],
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    scope_dict: Dict,
    never_consolidate_edges: bool = False,
    consolidate_edges_only_if_not_extending: bool = True,
) -> None:
    """Move the connectors and edges from `from_node` to `to_nodes` node.

    This function will only rewire the edges, it does not remove the nodes
    themselves. Furthermore, this function should be called twice per Map,
    once for the entry and then for the exit.
    While it does not remove the node themselves if guarantees that the
    `from_node` has degree zero.
    The function assumes that the parameter renaming was already done.

    :param from_node: Node from which the edges should be removed.
    :param to_node: Node to which the edges should reconnect.
    :param state: The state in which the operation happens.
    :param sdfg: The SDFG that is modified.

    :note: After the relocation Memlet propagation should be run.
    """

    # Now we relocate empty Memlets, from the `from_node` to the `to_node`
    for empty_edge in list(filter(lambda e: e.data.is_empty(), state.out_edges(from_node))):
        helpers.redirect_edge(state, empty_edge, new_src=to_node)
    for empty_edge in list(filter(lambda e: e.data.is_empty(), state.in_edges(from_node))):
        helpers.redirect_edge(state, empty_edge, new_dst=to_node)

    # We now ensure that there is only one empty Memlet from the `to_node` to any other node.
    #  Although it is allowed, we try to prevent it.
    empty_targets: Set[dace_nodes.Node] = set()
    for empty_edge in list(filter(lambda e: e.data.is_empty(), state.all_edges(to_node))):
        if empty_edge.dst in empty_targets:
            state.remove_edge(empty_edge)
        empty_targets.add(empty_edge.dst)

    # Relocation of the edges that carry data.
    for edge_to_move in list(state.in_edges(from_node)):
        assert isinstance(edge_to_move.dst_conn, str)

        if not edge_to_move.dst_conn.startswith("IN_"):
            # Dynamic Map Range
            #  The connector name simply defines a variable name that is used,
            #  inside the Map scope to define a variable. We handle it directly.
            dmr_symbol = edge_to_move.dst_conn

            # TODO(phimuell): Check if the symbol is really unused in the target scope.
            if dmr_symbol in to_node.in_connectors:
                raise NotImplementedError(
                    f"Tried to move the dynamic map range '{dmr_symbol}' from {from_node}'"
                    f" to '{to_node}', but the symbol is already known there, but the"
                    " renaming is not implemented."
                )
            if not to_node.add_in_connector(dmr_symbol, force=False):
                raise RuntimeError(  # Might fail because of out connectors.
                    f"Failed to add the dynamic map range symbol '{dmr_symbol}' to '{to_node}'."
                )
            helpers.redirect_edge(state=state, edge=edge_to_move, new_dst=to_node)
            from_node.remove_in_connector(dmr_symbol)

        else:
            # We have a Passthrough connection, i.e. there exists a matching `OUT_`.
            old_conn = edge_to_move.dst_conn[3:]  # The connection name without prefix
            new_conn, conn_was_reused = get_new_conn_name(
                edge_to_move=edge_to_move,
                to_node=to_node,
                state=state,
                scope_dict=scope_dict,
                never_consolidate_edges=never_consolidate_edges,
                consolidate_edges_only_if_not_extending=consolidate_edges_only_if_not_extending,
            )

            # Now move the incoming edges of `to_node` to `from_node`. However,
            #  we only move `edge_to_move` if we have a new connector, if we
            #  reuse the connector we will simply remove it.
            dst_in_conn = "IN_" + new_conn
            for e in list(state.in_edges_by_connector(from_node, f"IN_{old_conn}")):
                if conn_was_reused and e is edge_to_move:
                    state.remove_edge(edge_to_move)
                    if state.degree(edge_to_move.src) == 0:
                        state.remove_node(edge_to_move.src)
                else:
                    helpers.redirect_edge(state, e, new_dst=to_node, new_dst_conn=dst_in_conn)

            # Now move the outgoing edges of `to_node` to `from_node`.
            dst_out_conn = "OUT_" + new_conn
            for e in list(state.out_edges_by_connector(from_node, f"OUT_{old_conn}")):
                helpers.redirect_edge(state, e, new_src=to_node, new_src_conn=dst_out_conn)

            # If we have used new connectors we must add the new connector names.
            if not conn_was_reused:
                to_node.add_in_connector(dst_in_conn)
                to_node.add_out_connector(dst_out_conn)

            # In any case remove the old connector name from the `from_node`.
            from_node.remove_in_connector("IN_" + old_conn)
            from_node.remove_out_connector("OUT_" + old_conn)

    # Check if we succeeded.
    if state.out_degree(from_node) != 0:
        raise validation.InvalidSDFGError(
            f"Failed to relocate the outgoing edges from `{from_node}`, there are still `{state.out_edges(from_node)}`",
            sdfg,
            sdfg.node_id(state),
        )
    if state.in_degree(from_node) != 0:
        raise validation.InvalidSDFGError(
            f"Failed to relocate the incoming edges from `{from_node}`, there are still `{state.in_edges(from_node)}`",
            sdfg,
            sdfg.node_id(state),
        )
    assert len(from_node.in_connectors) == 0
    assert len(from_node.out_connectors) == 0

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

    if gtx_st.never_intersecting(first_map_sorted_range, second_map_sorted_range):
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
