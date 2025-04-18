# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Union

import dace
from dace import properties as dace_properties, transformation as dace_transformation
from dace.sdfg import nodes as dace_nodes, propagation as dace_propagation
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.transformations import map_fusion_utils


def gt_vertical_map_fusion(
    sdfg: dace.SDFG,
    run_simplify: bool,
    validate: bool = True,
    validate_all: bool = False,
) -> int:
    find_single_use_data = dace_analysis.FindSingleUseData()
    single_use_data = find_single_use_data.apply_pass(sdfg, None)

    ret = sdfg.apply_transformations_repeated(
        [
            MapRangeVerticalSplit(),
            gtx_transformations.SplitAccessNode(single_use_data=single_use_data),
            gtx_transformations.MapFusionSerial(),
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
class MapRangeVerticalSplit(dace_transformation.SingleStateTransformation):
    """
    Identify overlapping range between serial maps, and split the range in order
    to promote serial map fusion.
    """

    # Pattern Matching
    exit_first_map = dace_transformation.PatternNode(dace_nodes.MapExit)
    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    entry_second_map = dace_transformation.PatternNode(dace_nodes.MapEntry)

    @classmethod
    def expressions(cls) -> Any:
        """Get the match expressions.

        The function generates two match expressions. The first match describes
        the case where the top map must be promoted, while the second case is
        the second/lower map must be promoted.
        """
        return [
            dace.sdfg.utils.node_path_graph(
                cls.exit_first_map, cls.access_node, cls.entry_second_map
            ),
        ]

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Check non overlapping range in the first and second map."""
        assert self.expr_index == expr_index
        first_map = self.exit_first_map.map
        second_map = self.entry_second_map.map

        splitted_range = map_fusion_utils.split_overlapping_map_range(first_map, second_map)
        if splitted_range is None:
            return False

        # TODO(edopao): additional checks needed?
        return True

    def apply(self, graph: Union[dace.SDFGState, dace.SDFG], sdfg: dace.SDFG) -> None:
        """Split the map range in order to obtain an overlapping range between the first and second map."""
        splitted_range = map_fusion_utils.split_overlapping_map_range(
            self.exit_first_map.map, self.entry_second_map.map
        )
        assert splitted_range is not None

        first_map_entry: dace_nodes.MapEntry = graph.entry_node(self.exit_first_map)
        first_map_exit: dace_nodes.MapExit = self.exit_first_map
        second_map_entry: dace_nodes.MapEntry = self.entry_second_map
        second_map_exit: dace_nodes.MapExit = graph.exit_node(self.entry_second_map)

        first_map_splitted_range, second_map_splitted_range = splitted_range

        # make copies of the first map with splitted ranges
        for i, r in enumerate(first_map_splitted_range):
            map_fusion_utils.copy_map_graph_with_new_range(
                sdfg, graph, first_map_entry, first_map_exit, r, str(i)
            )

        # remove the original first map
        for node in graph.scope_subgraph(
            first_map_entry, include_entry=True, include_exit=True
        ).nodes():
            graph.remove_node(node)

        # make copies of the second map with splitted ranges
        for i, r in enumerate(second_map_splitted_range):
            map_fusion_utils.copy_map_graph_with_new_range(
                sdfg, graph, second_map_entry, second_map_exit, r, str(i)
            )

        # remove the original second map
        for node in graph.scope_subgraph(
            second_map_entry, include_entry=True, include_exit=True
        ).nodes():
            graph.remove_node(node)

        dace_propagation.propagate_memlets_state(sdfg, graph)

        # workaround to refresh `cfg_list` on the SDFG
        sdfg.hash_sdfg()
