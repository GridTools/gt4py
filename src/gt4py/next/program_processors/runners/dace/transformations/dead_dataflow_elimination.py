# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional

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
    The function returns the number of times the transformation was applied or zero
    if it was never applied.

    Args:
        sdfg: The SDFG to process.
        run_simplify: Run `gt_simplify()` after the dead datflow elimination.
        validate: Perform validation.
        validate_all: Perform extensive validation.
    """
    find_single_use_data = dace_analysis.FindSingleUseData()
    single_use_data = find_single_use_data.apply_pass(sdfg, None)

    # TODO(phimuell): Figuring out if `apply_transformations_once_everywhere()` is enough.
    ret = sdfg.apply_transformations_repeated(
        [
            DeadMemletElimination(single_use_data=single_use_data),
            DeadMapElimination(single_use_data=single_use_data),
        ],
        validate=validate,
        validate_all=validate_all,
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
        if not any(mr <= 0 for mr in map_range.size()):
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


@dace_properties.make_properties
class DeadMemletElimination(dace_transformation.SingleStateTransformation):
    """Removes Melets that does not carry any dataflow.

    The transformation matches an AccessNode and then processes all of its outgoing
    edges. An edge is removed if:
    - The Memlet carries no data.
    - The Memlet is connected to another AccessNode.

    If this would lead to isolated nodes then they are removed. By default the function
    will not remove the data from the registry, i.e. `sdfg.arrays`, except the
    `single_use_data` was passed at construction time.
    """

    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
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
        return [dace.sdfg.utils.node_path_graph(cls.access_node)]

    def _find_candidates(
        self,
        state: dace.SDFGState,
    ) -> list[dace.sdfg.state.MultiConnectorEdge]:
        """Find all edges of `self.access_node` that can be removed."""

        access_node: dace_nodes.AccessNode = self.access_node
        candidates: list[dace.sdfg.MultiConnectorEdge] = []

        for oedge in state.out_edges(access_node):
            dst: dace_nodes.Node = oedge.dst
            if not isinstance(dst, dace_nodes.AccessNode):
                continue

            memlet: dace.Memlet = oedge.data
            subset = memlet.subset if memlet.subset is not None else memlet.other_subset
            assert subset is not None, "Failed to identify the subset."
            if not any((ss <= 0) for ss in subset.size()):
                continue

            candidates.append(oedge)

        return candidates

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        access_node: dace_nodes.AccessNode = self.access_node
        if graph.out_degree(access_node) == 0:
            return False
        if len(self._find_candidates(graph)) == 0:
            return False
        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        access_node: dace_nodes.AccessNode = self.access_node
        removed_data_names: set[str] = set()

        for oedge in self._find_candidates(graph):
            other_node: dace_nodes.AccessNode = oedge.dst
            graph.remove_edge(oedge)

            if graph.degree(other_node) == 0:
                graph.remove_node(other_node)
                # Ensures that we never remove global data from the registry.
                if other_node.desc(sdfg).transient:
                    removed_data_names.add(other_node.data)

        if graph.degree(access_node) == 0:
            graph.remove_node(access_node)
            if access_node.desc(sdfg).transient:
                removed_data_names.add(access_node.data)

        if len(removed_data_names) != 0 and self._single_use_data is not None:
            single_use_data: set[str] = self._single_use_data[sdfg]
            for removed_data in removed_data_names:
                if removed_data in single_use_data:
                    sdfg.remove_data(removed_data, validate=False)
