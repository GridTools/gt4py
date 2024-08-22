# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the parallel map fusing transformation."""

from typing import Any, Union

import dace
from dace import properties, transformation
from dace.sdfg import SDFG, SDFGState, graph as dace_graph, nodes

from gt4py.next.program_processors.runners.dace_fieldview.transformations import (
    map_fusion_helper,
    util,
)


@properties.make_properties
class ParallelMapFusion(map_fusion_helper.MapFusionHelper):
    """The `ParallelMapFusion` transformation allows to merge two parallel maps together.

    The `SerialMapFusion` transformation is only able to handle maps that are sequential,
    however, this transformation is able to fuse _any_ maps that are not sequential
    and are in the same scope.

    Args:
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.

    Todo:
        Add options to restrict the surrounding further, such as common input.
    """

    map_entry1 = transformation.transformation.PatternNode(nodes.MapEntry)
    map_entry2 = transformation.transformation.PatternNode(nodes.MapEntry)

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

    @classmethod
    def expressions(cls) -> Any:
        # This just matches _any_ two Maps inside a state.
        state = dace_graph.OrderedMultiDiConnectorGraph()
        state.add_nodes_from([cls.map_entry1, cls.map_entry2])
        return [state]

    def can_be_applied(
        self,
        graph: Union[SDFGState, SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """The transformation is applicable."""
        map_entry_1: nodes.MapEntry = self.map_entry1
        map_entry_2: nodes.MapEntry = self.map_entry2

        # Check the structural properties of the maps, this will also ensure that
        #  the two maps are in the same scope.
        if not self.can_be_fused(
            map_entry_1=map_entry_1,
            map_entry_2=map_entry_2,
            graph=graph,
            sdfg=sdfg,
            permissive=permissive,
        ):
            return False

        # Since the match expression matches any twp Maps, we have to ensure that
        #  the maps are parallel. The `can_be_fused()` function already verified
        #  if they are in the same scope.
        if not util.is_parallel(graph=graph, node1=map_entry_1, node2=map_entry_2):
            return False

        return True

    def apply(self, graph: Union[SDFGState, SDFG], sdfg: SDFG) -> None:
        """Performs the Map fusing.

        Essentially, the function relocate all edges from the nodes forming the second
        Map to the corresponding nodes of the first Map. Afterwards the nodes of the
        second Map are removed.
        """
        assert self.map_parameter_compatible(self.map_entry1.map, self.map_entry2.map, graph, sdfg)

        map_entry_1: nodes.MapEntry = self.map_entry1
        map_exit_1: nodes.MapExit = graph.exit_node(map_entry_1)
        map_entry_2: nodes.MapEntry = self.map_entry2
        map_exit_2: nodes.MapExit = graph.exit_node(map_entry_2)

        for to_node, from_node in zip((map_entry_1, map_exit_1), (map_entry_2, map_exit_2)):
            self.relocate_nodes(
                from_node=from_node,
                to_node=to_node,
                state=graph,
                sdfg=sdfg,
            )
            # The relocate function does not remove the node, so we must do it.
            graph.remove_node(from_node)
