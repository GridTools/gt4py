# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# TODO(iomaganaris): This transformation is very specific to a pattern in the vertically_implicit_solver of ICON4Py.
# The transformation should be generalized or moved to ICON4Py.
# One idea to generalize this transformation is: in case we have a chain of 4 AccessNodes which are
# connected sequentially with memlets (first_node -> second_node -> third_node -> fourth_node),
# we can split the middle access nodes (second and third) based on their input and output memlets' subsets and those can later
# be removed because the same data are copied from the first all the way to fourth node.
# The split should be done using SplitAccessNode transformation.

import warnings
from typing import Any, Optional

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    transformation as dace_transformation,
)
from dace.sdfg import nodes as dace_nodes
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace.transformations import (
    splitting_tools as gtx_dace_split,
    strides as gtx_transformations_strides,
    utils as gtx_transformations_utils,
)


@dace_properties.make_properties
class RemoveAccessNodeCopies(dace_transformation.SingleStateTransformation):
    """
    Replace temporary AccessNodes' data with global AccessNode data in a pattern
    found in ICON4Py's dycore vertically_implicit_solver.
    The pattern is of the form:
        first_node (global) -> second_node (transient) -> third_node (transient) -> fourth_node (global)
    where first_node and fourth_node refer to the same data.
    The data of second_node and third_node are replaced with the data of first_node/fourth_node,
    and the subsets of the edges are updated accordingly.
    This transformation helps to reduce the number of data copies in the SDFG."""

    first_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    second_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    third_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    fourth_node = dace_transformation.PatternNode(dace_nodes.AccessNode)

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
        self._single_use_data = single_use_data

    @classmethod
    def expressions(cls) -> Any:
        return [
            dace.sdfg.utils.node_path_graph(
                cls.first_node, cls.second_node, cls.third_node, cls.fourth_node
            )
        ]

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        first_node: dace_nodes.AccessNode = self.first_node
        first_desc: dace_data.Data = first_node.desc(sdfg)
        second_node: dace_nodes.AccessNode = self.second_node
        second_desc: dace_data.Data = second_node.desc(sdfg)
        third_node: dace_nodes.AccessNode = self.third_node
        third_desc: dace_data.Data = third_node.desc(sdfg)
        fourth_node: dace_nodes.AccessNode = self.fourth_node
        fourth_desc: dace_data.Data = fourth_node.desc(sdfg)

        # Match the pattern found in "vertically_implicit_solver"
        if not (
            first_desc.transient is False
            and second_desc.transient is True
            and third_desc.transient is True
            and fourth_desc.transient is False
        ):
            return False

        scope_dict = graph.scope_dict()
        # Make sure that all the access nodes node are not in any scope
        if scope_dict[first_node] is not None:
            return False
        if scope_dict[second_node] is not None:
            return False
        if scope_dict[third_node] is not None:
            return False
        if scope_dict[fourth_node] is not None:
            return False

        # Make sure that first and fourth node refer to the same data
        if first_node.data != fourth_node.data:
            return False

        # Make sure that second and third node have the same shape
        if second_desc.shape != third_desc.shape:
            return False

        # Make sure that second and third node are not used anywhere else in the SDFG
        if self._single_use_data is None:
            find_single_use_data = dace_analysis.FindSingleUseData()
            single_use_data = find_single_use_data.apply_pass(sdfg, None)
        else:
            single_use_data = self._single_use_data
        if (
            second_node.data not in single_use_data[sdfg]
            and third_node.data not in single_use_data[sdfg]
        ):
            return False

        # Make sure that there is no other AccessNode with the same data as first and fourth nodes in the SDFG state
        for node in graph.data_nodes():
            if node not in [first_node, fourth_node] and node.data == first_node.data:
                return False

        # Make sure that the data written to the first node are not a subset of the data written to the fourth node
        ranges_written_to_fourth_node = []
        for edge in graph.in_edges(fourth_node):
            ranges_written_to_fourth_node.append(edge.data.dst_subset)

        ranges_written_to_first_node = []
        for edge in graph.in_edges(first_node):
            ranges_written_to_first_node.append(edge.data.dst_subset)
            if any(
                gtx_dace_split.are_intersecting(edge.data.dst_subset, fourth_node_range)
                for fourth_node_range in ranges_written_to_fourth_node
            ):
                return False

        # Make sure that the whole data of the first node is written
        first_node_range = first_desc.shape

        first_node_writes_subset = gtx_dace_split.subset_merger(ranges_written_to_first_node)
        if len(first_node_writes_subset) != 1:
            warnings.warn(
                "[RemoveAccessNodeCopies] The range of writes to the first node is not a single range.",
                stacklevel=0,
            )
            return False
        fourth_node_writes_subset = gtx_dace_split.subset_merger(ranges_written_to_fourth_node)
        if len(fourth_node_writes_subset) != 1:
            warnings.warn(
                "[RemoveAccessNodeCopies] The range of writes to the fourth node is not a single range.",
                stacklevel=0,
            )
            return False
        union_written_to_first_node_data = next(iter(first_node_writes_subset))
        union_written_to_fourth_node_data = next(iter(fourth_node_writes_subset))
        union_written_to_common_data = gtx_dace_split.subset_merger(
            [union_written_to_first_node_data, union_written_to_fourth_node_data]
        )
        if len(union_written_to_common_data) != 1:
            warnings.warn(
                "[RemoveAccessNodeCopies] The union of the ranges written to the first and fourth nodes is not a single range.",
                stacklevel=0,
            )
            return False
        if union_written_to_common_data != first_node_range:
            warnings.warn(
                "[RemoveAccessNodeCopies] The whole range of the first node is not written.",
                stacklevel=0,
            )
            # When https://github.com/GridTools/gt4py/pull/2173 is merged we can return False
            # At the moment we don't have the range of the global AccessNodes at compile time

        # Make sure that data written to first node are only copied from first to second node
        if any(edge.dst != second_node for edge in graph.out_edges(first_node)):
            return False

        # Make sure that there are no edges from second node that end up in any node apart from the third node
        for edge in graph.out_edges(second_node):
            if edge.dst != third_node and not gtx_transformations_utils.is_reachable(
                start=edge.dst,
                target=third_node,
                state=graph,
            ):
                return False
        source_nodes_of_third_node = gtx_transformations_utils.find_upstream_nodes(
            start=third_node,
            state=graph,
        )
        if second_node not in source_nodes_of_third_node:
            return False

        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        first_node: dace_nodes.AccessNode = self.first_node
        second_node: dace_nodes.AccessNode = self.second_node
        third_node: dace_nodes.AccessNode = self.third_node

        # Find the edge between first and second access nodes (there should be only one)
        edge_between_first_and_second = next(
            edge for edge in graph.edges_between(first_node, second_node)
        )

        # Compute the offset of the data between first and second node
        assert (
            edge_between_first_and_second.data.src_subset.dims()
            == edge_between_first_and_second.data.dst_subset.dims()
        )
        node_offset = [
            range_first_min - range_second_min
            for range_first_min, range_second_min in zip(
                edge_between_first_and_second.data.src_subset.min_element(),
                edge_between_first_and_second.data.dst_subset.min_element(),
            )
        ]

        # Make sure that the `subset` and `other_subset` of all memlets inside the map scopes
        # of the SDFG state are referring to the data of the access nodes which are outside the map scopes
        scope_dict = graph.scope_dict()
        for node in graph.nodes():
            if isinstance(node, (dace_nodes.EntryNode)) and scope_dict[node] is None:
                dace.sdfg.utils.canonicalize_memlet_trees_for_map(graph, node)
                dace.sdfg.utils.canonicalize_memlet_trees_for_map(graph, graph.exit_node(node))

        # Update all edges with the correct subsets that match the first node range
        for edge in graph.edges():
            # Update edges that refer to second or third AccessNodes
            if edge.data.data == second_node.data or edge.data.data == third_node.data:
                # Replace data with data of first node
                edge.data.data = first_node.data
                # Add offset to subset
                edge.data.subset.offset(node_offset, negative=False)
                # Add offset to other_subset if it's necessary (src is either first, second or third node and dst is either second or third node)
                if edge.dst in [second_node, third_node] and edge.data.other_subset is not None:
                    edge.data.other_subset.offset(node_offset, negative=False)
            elif edge.data.data == first_node.data and edge.dst in [second_node, third_node]:
                # Edge from first node to second or third node
                # Update other_subset to match the subset of the first node
                edge.data.other_subset.offset(node_offset, negative=False)

        # Replace data of second and third node with data of first and fourth node
        second_node.data = first_node.data
        third_node.data = first_node.data

        # Update strides after changing the data of access nodes
        for modified_node in [second_node, third_node]:
            gtx_transformations_strides.gt_propagate_strides_from_access_node(
                sdfg, graph, modified_node
            )
