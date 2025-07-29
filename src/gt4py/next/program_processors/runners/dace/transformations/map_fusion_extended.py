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
from dace.transformation.dataflow import map_fusion_helper as dace_mfhelper
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
    utils as gtx_dace_utils,
)
from gt4py.next.program_processors.runners.dace.transformations import (
    map_fusion_utils as gtx_mfutils,
)


def gt_horizontal_map_split_fusion(
    sdfg: dace.SDFG,
    run_simplify: bool,
    fuse_possible_maps: bool,
    consolidate_edges_only_if_not_extending: bool,
    skip: Optional[set[str]] = None,
    validate: bool = True,
    validate_all: bool = False,
) -> int:
    transformations = [
        HorizontalSplitMapRange(
            fuse_possible_maps=fuse_possible_maps,
            only_toplevel_maps=True,
            consolidate_edges_only_if_not_extending=consolidate_edges_only_if_not_extending,
        ),
        gtx_transformations.MapFusionHorizontal(
            only_if_common_ancestor=True,
            only_inner_maps=False,
            only_toplevel_maps=True,
            consolidate_edges_only_if_not_extending=consolidate_edges_only_if_not_extending,
        ),
    ]

    ret = sdfg.apply_transformations_repeated(
        transformations,
        validate=False,
        validate_all=validate_all,
    )

    if run_simplify:
        if skip is None:
            skip = gtx_transformations.constants.GT_SIMPLIFY_DEFAULT_SKIP_SET
        if not consolidate_edges_only_if_not_extending:
            skip = skip.union(["ConsolidateEdges"])
        gtx_transformations.gt_simplify(
            sdfg=sdfg,
            validate=False,
            validate_all=validate_all,
            skip=skip,
        )
    elif validate and (not validate_all):
        sdfg.validate()

    return ret


def gt_vertical_map_split_fusion(
    sdfg: dace.SDFG,
    run_simplify: bool,
    consolidate_edges_only_if_not_extending: bool,
    single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
    skip: Optional[set[str]] = None,
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
        gtx_transformations.MapFusionVertical(
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
        validate=False,
        validate_all=validate_all,
    )

    if run_simplify:
        if skip is None:
            skip = gtx_transformations.constants.GT_SIMPLIFY_DEFAULT_SKIP_SET
        if not consolidate_edges_only_if_not_extending:
            skip = skip.union(["ConsolidateEdges"])
        gtx_transformations.gt_simplify(
            sdfg=sdfg,
            validate=validate,
            validate_all=validate_all,
            skip=skip,
        )
    elif validate and (not validate_all):
        sdfg.validate()

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
    ) -> tuple[list[dace_nodes.MapEntry], list[dace_nodes.MapEntry]]:
        """Split the map range in order to obtain an overlapping range between the first and second map.

        The function returns a pair of lists. The first `list` contains the new `MapEntry`s that
        were generated by splitting the first Map and the second `list` contains the `MapEntry`s
        generated by splitting the second Map.
        """
        splitted_range = gtx_mfutils.split_overlapping_map_range(
            first_map_entry.map, second_map_entry.map
        )
        assert splitted_range is not None

        scope_dict = graph.scope_dict()
        assert scope_dict[first_map_entry] is scope_dict[second_map_entry]
        containing_scope_or_none = scope_dict[first_map_entry]

        first_map_splitted_range, second_map_splitted_range = splitted_range

        def _replace_ranged_map(
            map_entry: dace_nodes.MapEntry,
            map_exit: dace_nodes.MapExit,
            new_ranges: list[dace_subsets.Range],
        ) -> list[dace_nodes.MapEntry]:
            """Replace a map with multiple maps based on new_ranges."""
            new_map_entries: list[dace_nodes.MapEntry] = []
            for i, r in enumerate(new_ranges):
                new_map_entry, _ = gtx_mfutils.copy_map_graph_with_new_range(
                    sdfg, graph, map_entry, map_exit, r, str(i)
                )
                new_map_entries.append(new_map_entry)
            gtx_mfutils.delete_map(graph, map_entry, map_exit)  # remove the original first map
            return new_map_entries

        new_map_entries_from_first = _replace_ranged_map(
            first_map_entry,
            first_map_exit,
            first_map_splitted_range,
        )
        new_map_entries_from_second = _replace_ranged_map(
            second_map_entry,
            second_map_exit,
            second_map_splitted_range,
        )

        # Now we propagate the Memlets. If the original Maps where located at the global
        #  scope we have to propagate the maps from the new ones. However, if they are
        #  not at global scope we simply propagate the surrounding Map, since they are
        #  by assumption inside the same Map.
        if containing_scope_or_none is None:
            for new_map_entry in new_map_entries_from_first + new_map_entries_from_second:
                dace_propagation.propagate_memlets_map_scope(sdfg, graph, new_map_entry)
        else:
            dace_propagation.propagate_memlets_map_scope(sdfg, graph, containing_scope_or_none)

        # Workaround to ensure that some cache in DaCe has been cleared.
        # TODO(phimuell, iomaganaris): Before `hash_sdfg()` was used, but this was a
        #   very heavy operation. I think that this is enough.
        sdfg.reset_cfg_list()

        return new_map_entries_from_first, new_map_entries_from_second


@dace_properties.make_properties
class HorizontalSplitMapRange(SplitMapRange):
    """
    Identify overlapping range between parallel maps, and split the range in order
    to promote parallel map fusion.
    """

    first_map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)
    second_map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    fuse_possible_maps = dace_properties.Property(
        dtype=bool,
        default=True,
        desc="If `True`, the transformation will try to fuse maps that have the same range together.",
    )
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
        fuse_possible_maps: Optional[bool] = None,
        consolidate_edges_only_if_not_extending: Optional[bool] = None,
        never_consolidate_edges: Optional[bool] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if fuse_possible_maps is not None:
            self.fuse_possible_maps = fuse_possible_maps
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
        first_map_entry: dace_nodes.MapEntry = self.first_map_entry
        second_map_entry: dace_nodes.MapEntry = self.second_map_entry
        first_map: dace_nodes.Map = first_map_entry.map
        second_map: dace_nodes.Map = second_map_entry.map

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

        # Test if the Maps are parallel.
        if not dace_mfhelper.is_parallel(
            graph=graph, node1=first_map_entry, node2=second_map_entry
        ):
            return False

        if len(first_map_src_data.intersection(second_map_src_data)) == 0:
            # no common source access node
            return False

        splitted_range = gtx_mfutils.split_overlapping_map_range(first_map, second_map)
        if splitted_range is None:
            return False

        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        first_map_entry: dace_nodes.MapEntry = self.first_map_entry
        second_map_entry: dace_nodes.MapEntry = self.second_map_entry

        # Now split the first and second Map.
        first_map_fragments, second_map_fragments = self.split_maps(
            graph,
            sdfg,
            first_map_entry,
            graph.exit_node(first_map_entry),
            second_map_entry,
            graph.exit_node(second_map_entry),
        )

        # If we do not want to fuse the maps, we can stop here.
        if not self.fuse_possible_maps:
            return

        # Look which Maps can be fused together. It is important that a Map can be listed multiple
        #  times. However, by fusing we remove them from the state, which is a problem. We solve
        #  this by always using "first Maps" as `to_node`, thus they survive. Furthermore, we
        #  only assign a second Map once.
        # TODO(phimuell, iomaganaris): Check and improve that.
        matched_map_fragments: list[tuple[dace_nodes.MapEntry, dace_nodes.MapEntry]] = []
        unmatched_second_map_fragments = second_map_fragments.copy()
        for first_map_fragment in first_map_fragments:
            for unmatched_second_map_fragment in unmatched_second_map_fragments:
                if first_map_fragment.map.range == unmatched_second_map_fragment.map.range:
                    matched_map_fragments.append(
                        (first_map_fragment, unmatched_second_map_fragment)
                    )
                    unmatched_second_map_fragments.remove(unmatched_second_map_fragment)
                    break

        assert len(matched_map_fragments) > 0

        # We have to get the original scope dict before we start mutating the graph. Actually we
        #  would need to obtain the scope dict after every iteration again, which would require
        #  a rescan. But since the operations do not alter the scopes, at least not in a way that
        #  would affect us, we can be more efficient and get the thing at the beginning.
        scope_dict: Dict = graph.scope_dict()

        for first_map_fragment_entry, second_map_fragment_entry in matched_map_fragments:
            first_map_fragment_exit = graph.exit_node(first_map_fragment_entry)
            second_map_fragment_exit = graph.exit_node(second_map_fragment_entry)

            # Before we do anything we perform the renaming, i.e. we will rename the
            #  parameters of the second map such that they match the one of the first map.
            dace_mfhelper.rename_map_parameters(
                first_map=first_map_fragment_entry.map,
                second_map=second_map_fragment_entry.map,
                second_map_entry=second_map_fragment_entry,
                state=graph,
            )

            # Now we relocate all connectors from the second to the first map and remove
            #  the respective node of the second map.
            for to_node, from_node in [
                (first_map_fragment_entry, second_map_fragment_entry),
                (first_map_fragment_exit, second_map_fragment_exit),
            ]:
                dace_mfhelper.relocate_nodes(
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
    to promote serial map fusion. In contrast to `HorizontalSplitMapRange` this
    class is not able to automatically fuse.
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

        splitted_range = gtx_mfutils.split_overlapping_map_range(first_map, second_map)
        if splitted_range is None:
            return False

        # TODO(phimuell): Implement a check that the array is single use data?

        # Avoid spliting maps for fusion that if the second map has a nested map which is a neighbor reduction which
        # accesses the data that got generated by the first map. This is due to the fact that because of the neighbor
        # reduction, the second map might access a wider range of data that its range. This leads to a read-after-write
        # dependency between the first and second map.
        second_map_subgraph = graph.scope_subgraph(
            self.second_map_entry, include_entry=False, include_exit=False
        )
        for second_map_node in second_map_subgraph.nodes():
            if isinstance(second_map_node, dace_nodes.MapEntry):
                nested_map_input_edges_data = [
                    edge.data.data for edge in second_map_subgraph.in_edges(second_map_node)
                ]
                if self.access_node.data in nested_map_input_edges_data and any(
                    gtx_dace_utils.is_connectivity_identifier(data)
                    for data in nested_map_input_edges_data
                ):
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
