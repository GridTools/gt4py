# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Mapping, Optional

import dace
from dace import properties as dace_properties, transformation as dace_transformation
from dace.sdfg import nodes as dace_nodes
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


def gt_eliminate_dead_dataflow(
    sdfg: dace.SDFG,
    run_simplify: bool,
    validate: bool = True,
    validate_all: bool = False,
) -> int:
    """Performs dead dataflow elimination on the `sdfg`.

    The function will run GT4Py's dead datflow elimination transformation and if
    requested simplify afterwards. It is recommended to use this function instead
    as it will also run `FindSingleUseData` before.
    The function will return the number maps and memlets that have been removed,
    the output of `gt_simplify()`, if any, is ignored.

    Args:
        sdfg: The SDFG to process.
        run_simplify: Run `gt_simplify()` after the dead datflow elimination.
        validate: Perform validation.
        validate_all: Perform extensive validation.

    Todo:
        Implement a better way of applying the `DeadMemletElimination` transformation.
    """
    find_single_use_data = dace_analysis.FindSingleUseData()
    single_use_data = find_single_use_data.apply_pass(sdfg, None)

    ret = sdfg.apply_transformations_once_everywhere(
        DeadMapElimination(single_use_data=single_use_data),
        validate=validate,
        validate_all=validate_all,
    )
    ret += gt_dead_memlet_elimination(
        sdfg=sdfg,
        single_use_data=single_use_data,
    )

    if run_simplify:
        gtx_transformations.gt_simplify(
            sdfg=sdfg,
            validate=validate,
            validate_all=validate_all,
        )

    return ret


@dace_properties.make_properties
class DeadMapElimination(dace_transformation.SingleStateTransformation):
    """The transformation removes all Maps that are no-ops.

    The transformation matches any MapEntry, at the top level and evaluates the ranges,
    if it concludes that the map never execute, because at least one dimension as
    zero or less iteration steps, it will remove the entire Map.

    The function will also remove the nodes that have become isolated. If the optional
    `single_use_data` was passed at construction, the transformation will also remove
    the data from the registry, i.e. `SDFG.arrays`.
    """

    map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)
    _single_use_data: Optional[dict[dace.SDFG, set[str]]]

    def __init__(
        self,
        single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._single_use_data = None
        if single_use_data is not None:
            self._single_use_data = single_use_data
        super().__init__(*args, **kwargs)

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.map_entry)]

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        map_entry: dace_nodes.MapEntry = self.map_entry
        map_range: dace.subsets.Range = map_entry.map.range
        scope_dict = graph.scope_dict()

        # The map must be on the top.
        if scope_dict[map_entry] is not None:
            return False

        # Look if there is a dimension with zero or "less" iteration steps.
        if not any((mr <= 0) == True for mr in map_range.size()):  # noqa: E712 [true-false-comparison]  # SymPy fuzzy bools.
            return False

        # For ease of implementation we also require that the Map scope is only
        #  adjacent to AccessNodes.
        if not all(
            isinstance(iedge.src, dace_nodes.AccessNode) for iedge in graph.in_edges(map_entry)
        ):
            return False

        map_exit: dace_nodes.MapExit = graph.exit_node(map_entry)
        if not all(
            isinstance(oedge.dst, dace_nodes.AccessNode) for oedge in graph.out_edges(map_exit)
        ):
            return False

        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        map_entry: dace_nodes.MapEntry = self.map_entry
        map_exit: dace_nodes.MapExit = graph.exit_node(map_entry)
        map_scope = graph.scope_subgraph(map_entry, include_exit=True, include_entry=True)

        # Now find all notes that are producer or consumer of the Map, after we removed
        #  the nodes of the Maps we need to check if they have become isolated.
        adjacent_nodes: list[dace_nodes.AccessNode] = [
            iedge.src for iedge in graph.in_edges(map_entry)
        ]
        adjacent_nodes.extend(oedge.dst for oedge in graph.out_edges(map_exit))

        # Now we delete all nodes that constitute the Map scope.
        graph.remove_nodes_from(map_scope.nodes())

        # Remove the nodes that have become isolated.
        removed_data_names: set[str] = set()
        for adjacent_node in adjacent_nodes:
            if graph.degree(adjacent_node) == 0:
                graph.remove_node(adjacent_node)
                if adjacent_node.desc(sdfg).transient:
                    removed_data_names.add(adjacent_node.data)

        if self._single_use_data is not None:
            single_use_data = self._single_use_data[sdfg]
            for removed_data in removed_data_names:
                if removed_data in single_use_data:
                    sdfg.remove_data(removed_data, validate=False)


def gt_dead_memlet_elimination(
    sdfg: dace.SDFG,
    single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
) -> int:
    """Removes Memlets that do not carry any dataflow.

    The function will recursively examine the outgoing Memlets of all AccessNodes.
    If it finds a Memlet that does not carry any data, the Memlet will be removed.
    However, the function will only consider connections between two AccessNodes.
    If the removing of the Memlet leads to isolated nodes, these will be removed.
    Furthermore, if `single_use_data` was passed the function will also remove
    the data descriptors that have become obsolete from the registry.

    Args:
        sdfg: The SDFG that is processed.
        single_use_data: List of data that is only used at a single location.
            If passed the function will also clean the registry.

    Return:
        The number of Memlets that were removed.
    """
    ret = 0
    for rsdfg in sdfg.all_sdfgs_recursive():
        ret += _gt_dead_memlet_elimination_sdfg(sdfg=rsdfg, single_use_data=single_use_data)
    return ret


def _gt_dead_memlet_elimination_sdfg(
    sdfg: dace.SDFG,
    single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
) -> int:
    """Removes Memlets that do not carry any dataflow.

    The same as `gt_dead_memlet_elimination` but only operates on a single level.
    """

    def _find_candidates_for(
        access_node: dace_nodes.AccessNode,
        state: dace.SDFGState,
    ) -> list[dace.sdfg.state.MultiConnectorEdge]:
        candidates: list[dace.sdfg.MultiConnectorEdge] = []
        for oedge in state.out_edges(access_node):
            dst: dace_nodes.Node = oedge.dst
            memlet: dace.Memlet = oedge.data
            if not isinstance(dst, dace_nodes.AccessNode):
                continue
            if memlet.is_empty():
                continue
            subset = memlet.subset if memlet.subset is not None else memlet.other_subset
            assert subset is not None, "Failed to identify the subset."
            if not any((ss <= 0) == True for ss in subset.size()):  # noqa: E712 [true-false-comparison]  # SymPy fuzzy bools.
                continue
            candidates.append(oedge)
        return candidates

    def _process_access_node(
        access_node: dace_nodes.AccessNode,
        state: dace.SDFGState,
    ) -> tuple[int, set[dace_nodes.AccessNode]]:
        removed_access_nodes: set[dace_nodes.AccessNode] = set()
        removed_memlets: int = 0
        for oedge in _find_candidates_for(access_node, state):
            other_node: dace_nodes.AccessNode = oedge.dst
            state.remove_edge(oedge)
            removed_memlets += 1
            # TODO(phimuell): Handle the case if there are only empty Memlets.
            if state.degree(other_node) == 0:
                state.remove_node(other_node)
                removed_access_nodes.add(other_node)
        # TODO(phimuell): Handle the case if there are only empty Memlets.
        if state.degree(access_node) == 0:
            state.remove_node(access_node)
            removed_access_nodes.add(access_node)
        return removed_memlets, removed_access_nodes

    # Now go through all states and process the data nodes, also keep track of all
    removed_memlets: int = 0
    for state in sdfg.states():
        # Keep track of all AccessNodes that have been removed, for the later cleaning
        #  and to prevent `NondeNotFoundError`s in DaCe.
        removed_access_nodes: set[dace_nodes.AccessNode] = set()
        for dnode in list(state.data_nodes()):
            if dnode in removed_access_nodes:
                continue
            this_removed_memlets, this_removed_access_nodes = _process_access_node(
                access_node=dnode,
                state=state,
            )
            removed_memlets += this_removed_memlets
            removed_access_nodes.update(this_removed_access_nodes)

        # Now remove the data that is no longer needed from the registry.
        if single_use_data is not None and len(removed_access_nodes) != 0:
            sdfg_single_use_data = single_use_data[sdfg]
            removed_data_names: set[str] = {ac.data for ac in removed_access_nodes}
            arrays: Mapping[str, dace.data.Data] = sdfg.arrays
            for removed_data in removed_data_names:
                if not arrays[removed_data].transient:
                    continue
                if removed_data in sdfg_single_use_data:
                    sdfg.remove_data(removed_data, validate=False)

    return removed_memlets
