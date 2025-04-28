# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Optional, Union

import dace
from dace import properties as dace_properties, transformation as dace_transformation
from dace.sdfg import graph as dace_graph, nodes as dace_nodes, propagation as dace_propagation
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.transformations import map_fusion_utils


def gt_horizontal_map_fusion(
    sdfg: dace.SDFG,
    run_simplify: bool,
    only_inner_maps: Optional[bool] = None,
    only_toplevel_maps: Optional[bool] = None,
    validate: bool = True,
    validate_all: bool = False,
) -> int:
    find_single_use_data = dace_analysis.FindSingleUseData()
    single_use_data = find_single_use_data.apply_pass(sdfg, None)

    ret = sdfg.apply_transformations_repeated(
        [
            HorizontalSplitMapRange(
                only_inner_maps=only_inner_maps,
                only_toplevel_maps=only_toplevel_maps,
            ),
            HorizontalCloneMapRange(),
            gtx_transformations.SplitAccessNode(single_use_data=single_use_data),
            gtx_transformations.MapFusionParallel(
                only_if_common_ancestor=True,
                only_inner_maps=only_inner_maps,
                only_toplevel_maps=only_toplevel_maps,
            ),
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


def gt_vertical_map_fusion(
    sdfg: dace.SDFG,
    run_simplify: bool,
    only_inner_maps: Optional[bool] = None,
    only_toplevel_maps: Optional[bool] = None,
    validate: bool = True,
    validate_all: bool = False,
) -> int:
    find_single_use_data = dace_analysis.FindSingleUseData()
    single_use_data = find_single_use_data.apply_pass(sdfg, None)

    ret = sdfg.apply_transformations_repeated(
        [
            VerticalSplitMapRange(
                only_inner_maps=only_inner_maps,
                only_toplevel_maps=only_toplevel_maps,
            ),
            VerticalCloneMapRange(),
            gtx_transformations.SplitAccessNode(single_use_data=single_use_data),
            gtx_transformations.MapFusionSerial(
                only_inner_maps=only_inner_maps,
                only_toplevel_maps=only_toplevel_maps,
            ),
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
class SplitMapRange(dace_transformation.SingleStateTransformation):
    """
    Identify overlapping range between maps, and split the range in order to
    promote map fusion.
    """

    only_toplevel_maps = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="Only perform fusing if the Maps are in the top level.",
    )
    only_inner_maps = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="Only perform fusing if the Maps are inner Maps, i.e. does not have top level scope.",
    )

    def __init__(
        self,
        only_toplevel_maps: Optional[bool] = None,
        only_inner_maps: Optional[bool] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if only_toplevel_maps is not None:
            self.only_toplevel_maps = only_toplevel_maps
        if only_inner_maps is not None:
            self.only_inner_maps = only_inner_maps
        assert not (self.only_inner_maps and self.only_toplevel_maps)

    def split_maps(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
        first_map_entry: dace_nodes.MapEntry,
        first_map_exit: dace_nodes.MapExit,
        second_map_entry: dace_nodes.MapEntry,
        second_map_exit: dace_nodes.MapExit,
    ) -> None:
        """Split the map range in order to obtain an overlapping range between the first and second map."""
        splitted_range = map_fusion_utils.split_overlapping_map_range(
            first_map_entry.map, second_map_entry.map
        )
        assert splitted_range is not None

        first_map_splitted_range, second_map_splitted_range = splitted_range

        # make copies of the first map with splitted ranges
        for i, r in enumerate(first_map_splitted_range):
            map_fusion_utils.copy_map_graph_with_new_range(
                sdfg, graph, first_map_entry, first_map_exit, r, str(i)
            )

        # remove the original first map
        map_fusion_utils.delete_map(graph, first_map_entry, first_map_exit)

        # make copies of the second map with splitted ranges
        for i, r in enumerate(second_map_splitted_range):
            map_fusion_utils.copy_map_graph_with_new_range(
                sdfg, graph, second_map_entry, second_map_exit, r, str(i)
            )

        # remove the original second map
        map_fusion_utils.delete_map(graph, second_map_entry, second_map_exit)

        dace_propagation.propagate_memlets_state(sdfg, graph)

        # workaround to refresh `cfg_list` on the SDFG
        sdfg.hash_sdfg()


@dace_properties.make_properties
class HorizontalSplitMapRange(SplitMapRange):
    """
    Identify overlapping range between parallel maps, and split the range in order
    to promote parallel map fusion.
    """

    first_map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)
    second_map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
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
        first_map: dace_nodes.Map = self.first_map_entry.map
        second_map: dace_nodes.Map = self.second_map_entry.map

        map_scope: Union[dace_nodes.Node, None] = graph.scope_dict()[self.first_map_entry]
        if self.only_inner_maps and (map_scope is None):
            return False
        elif self.only_toplevel_maps and (map_scope is not None):
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

        self.split_maps(
            graph, sdfg, first_map_entry, first_map_exit, second_map_entry, second_map_exit
        )


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
        """Get the match expressions.

        The function generates two match expressions. The first match describes
        the case where the top map must be promoted, while the second case is
        the second/lower map must be promoted.
        """
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
        second_map = self.second_map_entry.map

        map_scope: Union[dace_nodes.Node, None] = graph.scope_dict()[self.first_map_exit]
        if self.only_inner_maps and (map_scope is None):
            return False
        elif self.only_toplevel_maps and (map_scope is not None):
            return False

        if not self.access_node.desc(graph).transient:
            return False

        splitted_range = map_fusion_utils.split_overlapping_map_range(first_map, second_map)
        if splitted_range is None:
            return False

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


@dace_properties.make_properties
class CloneMapRange(dace_transformation.SingleStateTransformation):
    """
    Identify small maps that can be duplicated and merged into large maps with
    adjacent range, in order to promote map fusion.
    """

    def duplicate_map(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
        map_entry: dace_nodes.MapEntry,
        map_exit: dace_nodes.MapExit,
    ) -> tuple[
        dace_nodes.MapEntry, dace_nodes.MapExit, dict[dace_nodes.AccessNode, dace_nodes.AccessNode]
    ]:
        """Create a copy of the given map and duplicate the output data containers.

        Args:
            sdfg: The SDFG containing the map graph.
            graph: The SDFG state containing the map graph.
            map_entry: The entry node of the map graph.
            map_exit: The exit node of the map graph.

        Returns:
            A tuple of three elements:
            - The entry node of the cloned map.
            - The exit node of the cloned map.
            - Mapping from old access nodes to new ones for the data output of the cloned map.
        """
        suffix = "clone"
        new_map_entry, new_map_exit = map_fusion_utils.copy_map_graph(
            sdfg, graph, map_entry, map_exit, suffix
        )

        new_access_nodes = {}
        for oedge in graph.out_edges(new_map_exit):
            assert isinstance(oedge.data.dst, dace_nodes.AccessNode)
            old_access_node = oedge.data.dst
            data_name = old_access_node.data
            data_desc = old_access_node.desc(sdfg)
            new_data_desc = data_desc.clone()
            new_data_name = sdfg.add_datadesc(
                f"{data_name}_{suffix}", new_data_desc, find_new_name=True
            )
            new_access_node = graph.add_access(new_data_name)
            oedge.dst = new_access_node
            oedge.data.data = new_data_name
            new_access_nodes[old_access_node] = new_access_node
            if not data_desc.transient:
                # We let the original map write to the global data node.
                #  Therefore, we just replace the output node with a sink temporary,
                #  that will be removed by dead-dataflow elimination pass.
                new_data_desc.transient = True

        # workaround to refresh `cfg_list` on the SDFG
        sdfg.hash_sdfg()

        return new_map_entry, new_map_exit, new_access_nodes


@dace_properties.make_properties
class HorizontalCloneMapRange(CloneMapRange):
    """
    Implementation of `CloneMapRange` for horizontal map fusion.
    """

    @classmethod
    def expressions(cls) -> Any:
        return []

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        return False

    def apply(self, graph: Union[dace.SDFGState, dace.SDFG], sdfg: dace.SDFG) -> None:
        raise NotImplementedError()


@dace_properties.make_properties
class VerticalCloneMapRange(CloneMapRange):
    """
    Implementation of `CloneMapRange` for horizontal map fusion.
    """

    @classmethod
    def expressions(cls) -> Any:
        return []

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        return False

    def apply(self, graph: Union[dace.SDFGState, dace.SDFG], sdfg: dace.SDFG) -> None:
        raise NotImplementedError()
