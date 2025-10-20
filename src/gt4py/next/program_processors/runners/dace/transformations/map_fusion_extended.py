# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TypeAlias, Union

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
    splitting_tools as gtx_dace_split,
)


VerticalMapSplitCallback: TypeAlias = Callable[
    ["VerticalSplitMapRange", dace_nodes.MapExit, dace_nodes.MapEntry, dace.SDFGState, dace.SDFG],
    bool,
]
"""Callback used to influence the behaviour of `VerticalSplitMapRange`.

The function is used by `can_be_applied()`, before any other check is performed, thus
it might be that the split would not apply. If the function returns `True` then the
transformation will continue and if `False` is returned then the transformation will
not continue. This is similar to the callback function offered by `MapFusionVertical`
or `HorizontalMapSplitCallback`.

It has the following arguments:
- The transformation object itself.
- The MaxExit node of the first Map.
- The MapEntry node of the second Map.
- The SDFGState in which the nodes where found.
- The SDFG itself.
"""


HorizontalMapSplitCallback: TypeAlias = Callable[
    [
        "HorizontalSplitMapRange",
        dace_nodes.MapEntry,
        dace_nodes.MapEntry,
        dace.SDFGState,
        dace.SDFG,
    ],
    bool,
]
"""Callback used to influence the behaviour of `HorizontalSplitMapRange`.

The function is used by `can_be_applied()`, before any other check is performed, thus
it might be that the split would not apply. If the function returns `True` then the
transformation will continue and if `False` is returned then the transformation will
not continue. This is similar to the callback function offered by `MapFusionHorizontal`
or `VerticalMapSplitCallback`.

It has the following arguments:
- The transformation object itself.
- The MapEntry node of the first Map.
- The MapEntry node of the second Map.
- The SDFGState in which the nodes where found.
- The SDFG itself.
"""


def gt_horizontal_map_split_fusion(
    sdfg: dace.SDFG,
    run_simplify: bool,
    fuse_map_fragments: bool,
    consolidate_edges_only_if_not_extending: bool,
    skip: Optional[set[str]] = None,
    run_map_fusion: bool = False,
    check_split_callback: Optional[HorizontalMapSplitCallback] = None,
    check_fusion_callback: Optional["gtx_transformations.HorizontalMapFusionCallback"] = None,
    only_if_common_ancestor: bool = True,
    validate: bool = True,
    validate_all: bool = False,
) -> int:
    """Performs horizontal map splitting on the provided SDFG.

    This function runs `HorizontalSplitMapRange` transformation until a fix point is
    reached. Due to a limitation that transformation might not fuse all possible
    Maps together. For this case the `run_map_fusion` flag is provided.
    If it is set, then `MapFusionHorizontal` is also called. It is important to
    note that Map fusion operates on all Maps in the SDFG.

    Args:
        sdfg: The SDFG on which we operate.
        run_simplify: Run `gt_simplify()` at the end.
        fuse_map_fragments: Directly fuse the Maps inside `HorizontalSplitMapRange`.
        consolidate_edges_only_if_not_extending: See `MapFusionHorizontal` for more.
        skip: Skip these transformation during simplification.
        run_map_fusion: If `True` also run `MapFusionHorizontal`. Note
            that this will fuse Maps globally.
        check_split_callback: Use this as callback for the split transformation, see
            `HorizontalMapSplitCallback` for more information.
        check_fusion_callback: Use this as callback for the `MapFusionHorizontal`
            transformation, it only has an effect if `run_map_fusion`
            has been set to `True`. Note that this callback is not passed to
            `HorizontalSplitMapRange` thus it can not be used to limit fusion of Maps
            that are created during the splitting, see `fuse_map_fragments`.
        only_if_common_ancestor: Passed to `MapFusionHorizontal` only has an effect
            if `run_map_fusion` is `True`.
        validate: Perform validation during the steps.
        validate_all: Perform extensive validation.
    """

    transformations = [
        HorizontalSplitMapRange(
            fuse_map_fragments=fuse_map_fragments,
            only_toplevel_maps=True,
            consolidate_edges_only_if_not_extending=consolidate_edges_only_if_not_extending,
            check_split_callback=check_split_callback,
        )
    ]

    if run_map_fusion:
        transformations.append(
            gtx_transformations.MapFusionHorizontal(
                check_fusion_callback=check_fusion_callback,
                only_toplevel_maps=True,
                consolidate_edges_only_if_not_extending=consolidate_edges_only_if_not_extending,
                only_if_common_ancestor=only_if_common_ancestor,
            )
        )

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
    run_map_fusion: bool,
    consolidate_edges_only_if_not_extending: bool,
    fuse_map_fragments: bool,
    single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
    skip: Optional[set[str]] = None,
    check_split_callback: Optional[VerticalMapSplitCallback] = None,
    check_fusion_callback: Optional["gtx_transformations.VerticalMapFusionCallback"] = None,
    validate: bool = True,
    validate_all: bool = False,
) -> int:
    """Performs vertical map splitting on the provided SDFG.

    The function essentially runs `VerticalSplitMapRange` until a fix point is reached.

    Note that if the function is called with `fuse_map_fragments` set to `True`,
    then the `VerticalSplitMapRange` will also run `MapFusionVertical` and
    `SplitAccessNode` but their activities are restricted to Maps that were created
    by the splitting.
    This is a different behaviour compared to `run_map_fusion` where `MapFusionVertical`
    is run as an independent transformation, which acts on the entire SDFG, it will
    however, respect `check_fusion_callback`.

    Args:
        sdfg: The SDFG on which we operate.
        run_simplify: Run `gt_simplify()` at the end.
        consolidate_edges_only_if_not_extending: See `MapFusionVertical` for more.
        fuse_map_fragments: Run `MapFusionVertical` directly inside the split transformation.
            Note that this is limited to the Maps that are were created. Furthermore,
            `check_fusion_callback` is not used.
        single_use_data: Precomputed single use data.
        skip: Skip these transformation during simplification.
        run_map_fusion: Also run `MapFusionVertical`. Note that it acts on the entire SDFG.
            This call uses `check_fusion_callback`.
        check_split_callback: Use this as callback for the split transformation, see
            `VerticalMapSplitCallback` for more information.
        check_fusion_callback: Use this as callback for `MapFusionVertical`.
        validate: Perform validation during the steps.
        validate_all: Perform extensive validation.

    Note:
        - In previous versions this function also called `MapFusionVertical` and
            `SplitAccessNode` without any restriction, which had the side effect
            that all Maps in the SDFG were subject to Map fusion.
        - Due to a bug in the transformation, not all Maps, that were created by
            the splitting were fused. Especially "chains" might still be present.
    """
    if single_use_data is None:
        find_single_use_data = dace_analysis.FindSingleUseData()
        single_use_data = find_single_use_data.apply_pass(sdfg, None)

    transformations = [
        VerticalSplitMapRange(
            only_toplevel_maps=True,
            check_split_callback=check_split_callback,
            consolidate_edges_only_if_not_extending=consolidate_edges_only_if_not_extending,
            fuse_map_fragments=fuse_map_fragments,
            single_use_data=single_use_data,
        )
    ]

    if run_map_fusion:
        transformations.append(
            gtx_transformations.MapFusionVertical(
                only_toplevel_maps=True,
                check_fusion_callback=check_fusion_callback,
                consolidate_edges_only_if_not_extending=consolidate_edges_only_if_not_extending,
            )
        )
        transformations[-1]._single_use_data = single_use_data

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
    to promote parallel map fusion. The transformation will merge the Maps that
    were created by the split among themselves.

    Note:
        The automatic fusion of Maps is only among the ones that were created, which
        is a bit different behaviour than for `VerticalSplitMapRange` where also
        possibilities between a newly created and an old map is considered.
    """

    first_map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)
    second_map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    fuse_map_fragments = dace_properties.Property(
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

    _check_split_callback: Optional[HorizontalMapSplitCallback]

    def __init__(
        self,
        fuse_map_fragments: Optional[bool] = None,
        consolidate_edges_only_if_not_extending: Optional[bool] = None,
        never_consolidate_edges: Optional[bool] = None,
        check_split_callback: Optional[HorizontalMapSplitCallback] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if fuse_map_fragments is not None:
            self.fuse_map_fragments = fuse_map_fragments
        if consolidate_edges_only_if_not_extending is not None:
            self.consolidate_edges_only_if_not_extending = consolidate_edges_only_if_not_extending
        if never_consolidate_edges is not None:
            self.never_consolidate_edges = never_consolidate_edges
        self._check_split_callback = check_split_callback
        super().__init__(*args, **kwargs)

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

        if self._check_split_callback is not None:
            if not self._check_split_callback(self, first_map_entry, second_map_entry, graph, sdfg):
                return False

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
        if not self.fuse_map_fragments:
            return

        # Look which Maps can be fused together. It is important that a Map can be listed multiple
        #  times. However, by fusing we remove them from the state, which is a problem. We solve
        #  this by always using "first Maps" as `to_node`, thus they survive. Furthermore, we
        #  only assign a second Map once.
        # TODO(phimuell, iomaganaris): Check and improve that.
        matched_map_fragments: list[tuple[dace_nodes.MapEntry, dace_nodes.MapEntry]] = []
        unmatched_second_map_fragments = second_map_fragments.copy()
        for first_map_fragment in first_map_fragments:
            for unmatched_second_map_fragment in unmatched_second_map_fragments.copy():
                if first_map_fragment.map.range == unmatched_second_map_fragment.map.range:
                    matched_map_fragments.append(
                        (first_map_fragment, unmatched_second_map_fragment)
                    )
                    unmatched_second_map_fragments.remove(unmatched_second_map_fragment)

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
    to promote serial map fusion. Furthermore, this transformation also performs
    `SplitAccessNode` on all intermediates of the first and second Map.

    If `fuse_map_fragments` is set to `True`, the default is `False`, then the
    transformation will also perform MapFusionVertical, but limited to the newly
    created Maps.

    Args:
        only_toplevel_maps: Only applies to top level maps.
        check_split_callback: Callback used to check if a split should be performed
            or not, see `VerticalMapSplitCallback` for more.
        fuse_map_fragments: Immediately apply fusion and node splitting on the
            generated Maps.
        single_use_data: Use this as single use data and do not compute it on the fly.

    Note:
        - Even if `fuse_map_fragments` is `True` it might be that some fusion involving
            the split Maps is still possible, this is a bug.
        - Currently all AccessNodes between the first and second Map are subjected to
            `SplitAccessNode`, not just the ones that defined the split. This is a bug.
    """

    # Pattern Matching
    first_map_exit = dace_transformation.PatternNode(dace_nodes.MapExit)
    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    second_map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    fuse_map_fragments = dace_properties.Property(
        dtype=bool,
        default=False,  # NOTE: For backwards compatibility.
        desc="If `True`, the transformation will try to fuse the maps that were generated together.",
    )
    consolidate_edges_only_if_not_extending = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="Only consolidate if this does not lead to an extension of the subset.",
    )

    _check_split_callback: Optional[VerticalMapSplitCallback]

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and be passed at construction time. Needed until
    #  [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: Optional[dict[dace.SDFG, set[str]]]

    def __init__(
        self,
        *args: Any,
        check_split_callback: Optional[VerticalMapSplitCallback] = None,
        fuse_map_fragments: Optional[bool] = None,
        single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
        consolidate_edges_only_if_not_extending: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        if fuse_map_fragments is not None:
            self.fuse_map_fragments = fuse_map_fragments
        if consolidate_edges_only_if_not_extending is not None:
            self.consolidate_edges_only_if_not_extending = consolidate_edges_only_if_not_extending
        self._check_split_callback = check_split_callback
        self._single_use_data = single_use_data
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

        # Check the callback
        if self._check_split_callback is not None:
            if not self._check_split_callback(self, first_map, second_map, graph, sdfg):
                return False

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

        # Check that the data read by the second map from the intermediate AccessNode
        # are not copied to another AccessNode.
        access_node_out_edges = list(graph.out_edges(self.access_node))
        second_map_in_edges = list(graph.in_edges(self.second_map_entry))
        access_node_subset_into_second_map = [
            second_map_edge.data.src_subset
            for second_map_edge in second_map_in_edges
            if second_map_edge.src.data == self.access_node.data
            and second_map_edge.dst_conn.startswith("IN_")
        ]
        if any(
            any(
                gtx_dace_split.are_intersecting(out_edge.data.src_subset, matching_subset)
                for matching_subset in access_node_subset_into_second_map
            )
            for out_edge in access_node_out_edges
            if out_edge.dst != self.second_map_entry
            and isinstance(out_edge.dst, dace_nodes.AccessNode)
            and not out_edge.dst.desc(
                graph
            ).transient  # limit to transient nodes to avoid unnecessary splits in "vertically_implicit_solver_at_{corrector, predictor}_step"
        ):
            return False

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

        intermediates_that_might_need_splitting: list[dace_nodes.AccessNode] = []
        for oedge in graph.out_edges(first_map_exit):
            if isinstance(oedge.dst, dace_nodes.AccessNode) and any(
                ooedge.dst is second_map_entry for ooedge in graph.out_edges(oedge.dst)
            ):
                intermediates_that_might_need_splitting.append(oedge.dst)

        new_first_map_entries, new_second_map_entries = self.split_maps(
            graph,
            sdfg,
            first_map_entry,
            first_map_exit,
            second_map_entry,
            second_map_exit,
        )

        # We always split the AccessNode that are involved, regardless of `fuse_map_fragments`.
        # TODO(phimuell, iomaganaris): Refine this such that only the AccessNodes that
        #   defined the split are actually split and not some other random nodes.
        # TODO(phimuell): Make it possible to pass `self.single_use_data` to transformation.
        has_performed_a_split = False
        for intermediate_to_split in intermediates_that_might_need_splitting:
            if gtx_transformations.SplitAccessNode.can_be_applied_to(
                sdfg=sdfg,
                options={},
                access_node=intermediate_to_split,
            ):
                gtx_transformations.SplitAccessNode.apply_to(
                    sdfg=sdfg, options={}, access_node=intermediate_to_split
                )
                has_performed_a_split = True

        if not has_performed_a_split:
            # TODO(phimuell, iomaganaris): Is this an error?
            return

        # If requested perform the fusion.
        if self.fuse_map_fragments:
            assert len(intermediates_that_might_need_splitting) >= 1

            new_second_map_exits = [
                graph.exit_node(new_second_map_entry)
                for new_second_map_entry in new_second_map_entries
            ]

            def _restrict_fusion_to_newly_created_maps(
                this: gtx_transformations.MapFusionVertical,
                first_map_exit: dace_nodes.MapExit,
                second_map_entry: dace_nodes.MapEntry,
                state: dace.SDFGState,
                _sdfg: dace.SDFG,
            ) -> bool:
                # NOTE: For the check we have to use the entry of the first and the
                #   exit of the second Map, although there were not matched. The
                #   reason is that `MapFusionVertical` will remove the matched nodes,
                #   thus we will "loose sight of them". However, the "opposite" nodes
                #   of the Maps should survive, except in strange situation, but this
                #   is why we do not guarantee that everything related to the split
                #   has been merged.
                # NOTE: We use `or` here such that every fusion which involves
                #   a Map is is generated is considered. This should even work over
                #   chains, I guess.
                if (
                    state.entry_node(first_map_exit) in new_first_map_entries
                    or state.exit_node(second_map_entry) in new_second_map_exits
                ):
                    return True
                return False

            # NOTE: After here it is dangerous to access `self.PATTERN_NODE`.
            sdfg.reset_cfg_list()

            trafo = gtx_transformations.MapFusionVertical(
                check_fusion_callback=_restrict_fusion_to_newly_created_maps,
                only_toplevel_maps=self.only_toplevel_maps,
                consolidate_edges_only_if_not_extending=self.consolidate_edges_only_if_not_extending,
            )
            trafo._single_use_data = self._single_use_data

            # This is not efficient, but it is currently the only way to run it
            sdfg.apply_transformations_repeated(trafo, validate=False, validate_all=False)
