# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import Any, Optional

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    subsets as dace_sbs,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


@dace_properties.make_properties
class SplitConsumerMemlet(dace_transformation.SingleStateTransformation):
    """Split a consumer edge such that `SplitAccessNode` can be applied.

    The transformation matches an AccessNode and examines outgoing edges
    to other AccessNodes. The transformation will then check if these
    edges could be split such that the matched AccessNode can be handled
    by the `SplitAccessNode` transformation.

    This transformation is also related to pattern that are generated
    by `concat_where()`.

    Args:
        single_use_data: The result of the `FindSingleUseData` analysis
            pass. If not passed the transformation will rescan the
            SDFG every time.

    Note:
        For performance reasons the transformation matches two AccessNode,
        that are directly connected by an edge.
    """

    source_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    destination_node = dace_transformation.PatternNode(dace_nodes.AccessNode)

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and be passed at construction time. Needed until
    #  [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: Optional[dict[dace.SDFG, set[str]]]

    def __init__(
        self,
        *args: Any,
        single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._single_use_data = None
        if single_use_data is not None:
            self._single_use_data = single_use_data

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.source_node, cls.destination_node)]

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        src_node: dace_nodes.AccessNode = self.source_node
        dst_node: dace_nodes.AccessNode = self.destination_node
        src_desc: dace_data.Data = src_node.desc(sdfg)

        # Since, this transformation is to prepare for the `SplitAccessNode`
        #  transformation. Thus we have to meet its requirements. The test for
        #  single use data is performed later.
        if not src_desc.transient:
            return False
        if gtx_transformations.utils.is_view(src_desc, sdfg):
            return False
        if graph.in_degree(src_node) <= 1:
            return False

        # TODO(phimuell): For optimal result we should fuse these edges first.
        src_to_dest_edges = list(graph.edges_between(src_node, dst_node))
        if len(src_to_dest_edges) != 1:
            warnings.warn(
                f"Found multiple edges between '{src_node.data}' and '{dst_node.data}'",
                stacklevel=0,
            )

        # A temporary might make sense if the data is used multiple times. Thus in
        #  that case we have to keep it alive. By checking if a memory subset is read
        #  multiple times we can do that, but it is a very basic check.
        # TODO(phimuell): Improve this condition such that the split can be performed
        #   under the consideration of all consumer edges.
        split_description, copy_edges_to_split = self._get_copy_edges_and_split_description(
            src_node, graph, sdfg
        )
        known_read_subsets: list[dace_sbs.Subset] = []
        for oedge in copy_edges_to_split:
            current_read_subset = oedge.data.src_subset
            if current_read_subset is None:
                return False
            for known_read_subset in known_read_subsets:
                if dace_sbs.intersects(known_read_subset, current_read_subset) != False:  # noqa: E712 [true-false-comparison]  # Handle incomparable.
                    return False
            known_read_subsets.append(current_read_subset)
        assert len(known_read_subsets) > 0

        # We have to ensure that we can actually split something.
        found_edge_to_split = False
        for copy_edge_to_split in copy_edges_to_split:
            for split in split_description:
                if gtx_transformations.spliting_tools.decompose_subset(
                    producer=split, consumer=copy_edge_to_split.data.src_subset
                ):
                    found_edge_to_split = True
                    break
            if found_edge_to_split:
                break
        if not found_edge_to_split:
            return False

        # We do the `SingleUseData` scan here, to postpone it as long as possible.
        if self._single_use_data is None:
            find_single_use_data = dace_analysis.FindSingleUseData()
            single_use_data = find_single_use_data.apply_pass(sdfg, None)
        else:
            single_use_data = self._single_use_data
        if src_node.data not in single_use_data[sdfg]:
            return False

        # TODO(phimuell): Find out if these checks are enough.
        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        split_description, copy_edges_to_split = self._get_copy_edges_and_split_description(
            self.source_node, graph, sdfg
        )
        for copy_edge_to_split in copy_edges_to_split:
            gtx_transformations.spliting_tools.split_copy_edge(
                state=graph,
                sdfg=sdfg,
                edge_to_split=copy_edge_to_split,
                split_description=split_description,
            )

    def _get_copy_edges_and_split_description(
        self,
        node: dace_nodes.AccessNode,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> tuple[list[dace_sbs.Subset], list[dace_graph.MultiConnectorEdge]]:
        split_description = [
            desc.subset
            for desc in gtx_transformations.spliting_tools.describe_incoming_edges(state, node)
        ]
        copy_edges_to_split = [
            oedge
            for oedge in state.out_edges(node)
            if isinstance(oedge.dst, dace_nodes.AccessNode)
            and (not gtx_transformations.utils.is_view(oedge.dst, sdfg))
            and (oedge.data.dst_subset is not None)
        ]
        return split_description, copy_edges_to_split
