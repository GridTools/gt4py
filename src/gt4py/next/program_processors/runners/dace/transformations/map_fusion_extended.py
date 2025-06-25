# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import dace
from dace import (
    properties as dace_properties,
    subsets as dace_subsets,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes, propagation as dace_propagation
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.transformations import map_fusion_utils
from gt4py.next.program_processors.runners.dace.transformations.map_fusion_utils import (
    relocate_nodes,
    rename_map_parameters,
)

def gt_horizontal_map_fusion(
    sdfg: dace.SDFG,
    run_simplify: bool,
    consolidate_edges_only_if_not_extending: bool,
    single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
    validate: bool = True,
    validate_all: bool = False,
) -> int:
    if single_use_data is None:
        find_single_use_data = dace_analysis.FindSingleUseData()
        single_use_data = find_single_use_data.apply_pass(sdfg, None)

    # This transformation is only useful to operate on Maps that translates to
    #  kernels (OpenMP loops or GPU kernels). Thus we have to restrict them
    #  such that these Maps are processed. Thus we set `only_toplevel_maps` to
    #  `True`.
    # TODO: Restrict MapFusion such that it only applies to the Maps that have
    #   been split and not some other random Maps.
    transformations = [
        HorizontalSplitMapRange(
            only_toplevel_maps=True,
            consolidate_edges_only_if_not_extending=consolidate_edges_only_if_not_extending,
        ),
        # gtx_transformations.SplitAccessNode(single_use_data=single_use_data),
        gtx_transformations.MapFusionParallel(
            only_if_common_ancestor=True,
            only_inner_maps=False,
            only_toplevel_maps=True,
            consolidate_edges_only_if_not_extending=consolidate_edges_only_if_not_extending,
        ),
    ]
    # TODO(phimuell): Remove that hack once [issue#1911](https://github.com/spcl/dace/issues/1911)
    #   has been solved.
    transformations[-1]._single_use_data = single_use_data  # type: ignore[attr-defined]

    ret = sdfg.apply_transformations_repeated(
        transformations,
        validate=validate,
        validate_all=validate_all,
    )

    if run_simplify:
        skip = gtx_transformations.simplify.GT_SIMPLIFY_DEFAULT_SKIP_SET
        if not consolidate_edges_only_if_not_extending:
            skip = skip.union(["ConsolidateEdges"])
        gtx_transformations.gt_simplify(
            sdfg=sdfg,
            validate=validate,
            validate_all=validate_all,
            skip=skip,
        )

    return ret


def gt_vertical_map_fusion(
    sdfg: dace.SDFG,
    run_simplify: bool,
    consolidate_edges_only_if_not_extending: bool,
    single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
    validate: bool = True,
    validate_all: bool = False,
) -> int:
    if single_use_data is None:
        find_single_use_data = dace_analysis.FindSingleUseData()
        single_use_data = find_single_use_data.apply_pass(sdfg, None)

    # TODO: Restrict MapFusion such that it only applies to the Maps that have
    #   been split and not some other random Maps.
    transformations = [
        VerticalSplitMapRange(
            only_toplevel_maps=True,
        ),
        gtx_transformations.SplitAccessNode(single_use_data=single_use_data),
        gtx_transformations.MapFusionSerial(
            only_inner_maps=False,
            only_toplevel_maps=True,
            consolidate_edges_only_if_not_extending=consolidate_edges_only_if_not_extending,
        ),
    ]
    # TODO(phimuell): Remove that hack once [issue#1911](https://github.com/spcl/dace/issues/1911)
    #   has been solved.
    transformations[-1]._single_use_data = single_use_data  # type: ignore[attr-defined]

    ret = sdfg.apply_transformations_repeated(
        transformations,
        validate=validate,
        validate_all=validate_all,
    )

    if run_simplify:
        skip = gtx_transformations.simplify.GT_SIMPLIFY_DEFAULT_SKIP_SET
        if not consolidate_edges_only_if_not_extending:
            skip = skip.union(["ConsolidateEdges"])
        gtx_transformations.gt_simplify(
            sdfg=sdfg,
            validate=validate,
            validate_all=validate_all,
            skip=skip,
        )

    return ret


@dace_properties.make_properties
class SplitMapRange(dace_transformation.SingleStateTransformation):
    """
    Identify overlapping range between maps, and split the maps into
    maps with smaller ranges in order to create two maps with common
    range that can be fused together using the `HorizontalSplitMapRange`
    or `VerticalSplitMapRange` transformations.
    For example, given a map with range [0:10, 14:80] and a second map
    with range [5:15, 0:80], this transformation will split the first map
    into two maps with ranges [0:5, 14:80] and [5:10, 14:80], and the second
    map into four maps with ranges [0:10, 0:14], [0:10, 14:80] [10:15, 0:14]
    and [10:15, 14:80]. This will allow the `HorizontalSplitMapRange` or
    `VerticalSplitMapRange` transformations to fuse the maps together.
    """

    only_toplevel_maps = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="Only perform fusing if the Maps are in the top level.",
    )

    def __init__(
        self,
        only_toplevel_maps: Optional[bool] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if only_toplevel_maps is not None:
            self.only_toplevel_maps = only_toplevel_maps

    def split_maps(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
        first_map_entry: dace_nodes.MapEntry,
        first_map_exit: dace_nodes.MapExit,
        second_map_entry: dace_nodes.MapEntry,
        second_map_exit: dace_nodes.MapExit,
    ) -> tuple[list[tuple[dace_nodes.MapEntry, dace_nodes.MapExit]], list[tuple[dace_nodes.MapEntry, dace_nodes.MapExit]]]:
        """Split the map range in order to obtain an overlapping range between the first and second map."""
        splitted_range = map_fusion_utils.split_overlapping_map_range(
            first_map_entry.map, second_map_entry.map
        )
        assert splitted_range is not None

        first_map_splitted_range, second_map_splitted_range = splitted_range

        def _replace_ranged_map(
            map_entry: dace_nodes.MapEntry,
            map_exit: dace_nodes.MapExit,
            new_ranges: list[dace_subsets.Range],
        ) -> list[tuple[dace_nodes.MapEntry, dace_nodes.MapExit]]:
            """Replace a map with multiple maps based on new_ranges."""
            new_maps = []
            # make copies of the map with splitted ranges
            for i, r in enumerate(new_ranges):
                new_maps.append(
                    map_fusion_utils.copy_map_graph_with_new_range(
                        sdfg, graph, map_entry, map_exit, r, str(i)
                    )
                )

            # remove the original first map
            map_fusion_utils.delete_map(graph, map_entry, map_exit)

            return new_maps


        new_maps_from_first = _replace_ranged_map(
            first_map_entry,
            first_map_exit,
            first_map_splitted_range,
        )
        new_maps_from_second = _replace_ranged_map(
            second_map_entry,
            second_map_exit,
            second_map_splitted_range,
        )

        dace_propagation.propagate_memlets_state(sdfg, graph)

        # workaround to refresh `cfg_list` on the SDFG
        sdfg.hash_sdfg()

        return new_maps_from_first, new_maps_from_second


@dace_properties.make_properties
class HorizontalSplitMapRange(SplitMapRange):
    """
    Identify overlapping range between parallel maps, and split the range in order
    to promote parallel map fusion.
    """

    first_map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)
    second_map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    never_consolidate_edges = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="If `True`, always create a new connector, instead of reusing one that referring to the same data.",
    )
    consolidate_edges_only_if_not_extending = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="Only consolidate if this does not lead to an extension of the subset.",
    )

    def __init__(
        self,
        consolidate_edges_only_if_not_extending: Optional[bool] = None,
        never_consolidate_edges: Optional[bool] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if consolidate_edges_only_if_not_extending is not None:
            self.consolidate_edges_only_if_not_extending = consolidate_edges_only_if_not_extending
        if never_consolidate_edges is not None:
            self.never_consolidate_edges = never_consolidate_edges

    @classmethod
    def expressions(cls) -> Any:
        map_fusion_parallel_match = dace_graph.OrderedMultiDiConnectorGraph()
        map_fusion_parallel_match.add_nodes_from([cls.first_map_entry, cls.second_map_entry])
        return [map_fusion_parallel_match]

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        first_map: dace_nodes.Map = self.first_map_entry.map
        second_map: dace_nodes.Map = self.second_map_entry.map

        # Ensure that the Maps are in the same scope.
        scope_dict = graph.scope_dict()
        if scope_dict[self.first_map_entry] is not scope_dict[self.second_map_entry]:
            return False

        # Test if the map is in the right scope.
        map_scope: Union[dace_nodes.Node, None] = scope_dict[self.first_map_entry]
        if self.only_toplevel_maps and (map_scope is not None):
            return False

        first_map_src_data = {
            iedge.src.label
            for iedge in graph.in_edges(self.first_map_entry)
            if isinstance(iedge.src, dace_nodes.AccessNode)
        }
        second_map_src_data = {
            iedge.src.label
            for iedge in graph.in_edges(self.second_map_entry)
            if isinstance(iedge.src, dace_nodes.AccessNode)
        }

        if len(first_map_src_data.intersection(second_map_src_data)) == 0:
            # no common source access node
            return False

        splitted_range = map_fusion_utils.split_overlapping_map_range(first_map, second_map)
        if splitted_range is None:
            return False

        # TODO: Ensure that the fusion can be performed.

        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        first_map_entry: dace_nodes.MapEntry = self.first_map_entry
        first_map_exit: dace_nodes.MapExit = graph.exit_node(first_map_entry)
        second_map_entry: dace_nodes.MapEntry = self.second_map_entry
        second_map_exit: dace_nodes.MapExit = graph.exit_node(second_map_entry)

        new_maps_from_first, new_maps_from_second = self.split_maps(
            graph, sdfg, first_map_entry, first_map_exit, second_map_entry, second_map_exit
        )

        matching_maps = []
        for first_map_entry, first_map_exit in new_maps_from_first:
            for second_map_entry, second_map_exit in new_maps_from_second:
                if first_map_entry.map.range == second_map_entry.map.range and first_map_exit.map.range == second_map_exit.map.range:
                    matching_maps.append(
                        (first_map_entry, first_map_exit, second_map_entry, second_map_exit)
                    )

        # We have to get the scope_dict before we start mutating the graph.
        scope_dict: Dict = graph.scope_dict().copy()

        for first_map_entry, first_map_exit, second_map_entry, second_map_exit in matching_maps:
            # Before we do anything we perform the renaming, i.e. we will rename the
            #  parameters of the second map such that they match the one of the first map.
            rename_map_parameters(
                first_map=first_map_entry.map,
                second_map=second_map_entry.map,
                second_map_entry=second_map_entry,
                state=graph,
            )

            # Now we relocate all connectors from the second to the first map and remove
            #  the respective node of the second map.
            for to_node, from_node in [
                (first_map_entry, second_map_entry),
                (first_map_exit, second_map_exit),
            ]:
                relocate_nodes(
                    from_node=from_node,
                    to_node=to_node,
                    state=graph,
                    sdfg=sdfg,
                    scope_dict=scope_dict,
                    never_consolidate_edges=self.never_consolidate_edges,
                    consolidate_edges_only_if_not_extending=self.consolidate_edges_only_if_not_extending,
                )
                # The relocate function does not remove the node, so we must do it.
                graph.remove_node(from_node)


@dace_properties.make_properties
class VerticalSplitMapRange(SplitMapRange):
    """
    Identify overlapping range between serial maps, and split the range in order
    to promote serial map fusion.
    """

    # Pattern Matching
    first_map_exit = dace_transformation.PatternNode(dace_nodes.MapExit)
    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    second_map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def expressions(cls) -> Any:
        return [
            dace.sdfg.utils.node_path_graph(
                cls.first_map_exit, cls.access_node, cls.second_map_entry
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
        first_map = self.first_map_exit.map
        first_map_entry: dace_nodes.MapEntry = graph.entry_node(self.first_map_exit)
        second_map = self.second_map_entry.map

        # Test if the map is in the right scope.
        map_scope: Union[dace_nodes.Node, None] = graph.scope_dict()[first_map_entry]
        if self.only_toplevel_maps and (map_scope is not None):
            return False

        if not self.access_node.desc(graph).transient:
            return False

        splitted_range = map_fusion_utils.split_overlapping_map_range(first_map, second_map)
        if splitted_range is None:
            return False

        # TODO: Ensure that the fusion can be performed.

        return True

    def apply(self, graph: Union[dace.SDFGState, dace.SDFG], sdfg: dace.SDFG) -> None:
        """Split the map range in order to obtain an overlapping range between the first and second map."""

        first_map_entry: dace_nodes.MapEntry = graph.entry_node(self.first_map_exit)
        first_map_exit: dace_nodes.MapExit = self.first_map_exit
        second_map_entry: dace_nodes.MapEntry = self.second_map_entry
        second_map_exit: dace_nodes.MapExit = graph.exit_node(self.second_map_entry)

        self.split_maps(
            graph, sdfg, first_map_entry, first_map_exit, second_map_entry, second_map_exit
        )
