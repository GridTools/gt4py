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
        ),
        gtx_transformations.SplitAccessNode(single_use_data=single_use_data),
        gtx_transformations.MapFusionParallel(
            only_if_common_ancestor=True,
            only_inner_maps=False,
            only_toplevel_maps=True,
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
        gtx_transformations.gt_simplify(
            sdfg=sdfg,
            validate=validate,
            validate_all=validate_all,
        )

    return ret


def gt_vertical_map_fusion(
    sdfg: dace.SDFG,
    run_simplify: bool,
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
