# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    transformation as dace_transformation,
)
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace.transformations import (
    splitting_tools as gtx_dace_split,
    utils as gtx_transformations_utils,
)


@dace_properties.make_properties
class RemoveAccessNodeCopies(dace_transformation.SingleStateTransformation):
    """
    Replace temporary AccessNodes that are then copied to a global AccessNode
    to avoid copies.
    This pattern only exists in "vertically_implicit_solver" in ICON4Py dycore.
    """

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
        self._single_use_data = None
        if single_use_data is not None:
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

        if (
            first_desc.transient is False
            and second_desc.transient is True
            and third_desc.transient is True
            and fourth_desc.transient is False
        ):
            print(
                "[RemoveAccessNodeCopies] {} -> {} -> {} -> {}".format(
                    first_node.data, second_node.data, third_node.data, fourth_node.data
                )
            )
        else:
            return False

        if first_node.data != fourth_node.data:
            print("[RemoveAccessNodeCopies] First and fourth node do not have the same data")
            return False

        if second_desc.shape != third_desc.shape:
            print("[RemoveAccessNodeCopies] Second and third node do not have the same shape")
            return False

        # Make sure that second and fourth node have the same shape as the first node
        if second_desc.shape != first_desc.shape or third_desc.shape != first_desc.shape:
            print(
                f"[RemoveAccessNodeCopies] Shapes of second or third node do not match the shape of the first node: {second_desc.shape}, {third_desc.shape} vs {first_desc.shape}"
            )
            # needs https://github.com/GridTools/gt4py/pull/2173 # return False

        # Make sure that there is no other AccessNode with the same data in the SDFG state
        for node in graph.nodes():
            if (
                isinstance(node, dace_nodes.AccessNode)
                and node not in [first_node, second_node, third_node, fourth_node]
                and node.data == first_node.data
            ):
                print(
                    "[RemoveAccessNodeCopies] There is another AccessNode with the same data in the SDFG state"
                )
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
                print(
                    "[RemoveAccessNodeCopies] Data written to first node are a subset of the data written to fourth node"
                )
                return False

        # Make sure that the whole data of the first node is written
        first_node_range = first_desc.shape

        union_written_to_first_node_data = gtx_dace_split.union_of_subsets(
            ranges_written_to_first_node
        )
        union_written_to_fourth_node_data = gtx_dace_split.union_of_subsets(
            ranges_written_to_fourth_node
        )
        union_written_to_common_data = gtx_dace_split.union_of_subsets(
            [union_written_to_first_node_data, union_written_to_fourth_node_data]
        )
        if union_written_to_common_data != first_node_range:
            print("[RemoveAccessNodeCopies] Not all data of the first node are written")
            # needs https://github.com/GridTools/gt4py/pull/2173 # return False

        # Make sure that data written to first node are only copied from first to second and from second to third node (they don't have to be rewritten to the last node)
        first_node_out_edges = list(graph.out_edges(first_node))

        # Make sure that all the edges from the first node go to the second node
        if any(edge.dst != second_node for edge in first_node_out_edges):
            print(
                "[RemoveAccessNodeCopies] There are edges from first node that do not go to the second node"
            )
            return False

        # Make sure that there are no edges from second node that end up in any node apart from the third node (we have to potentially check that there are no other nodes reachable from the second node apart from the third node)
        second_node_out_edges = list(graph.out_edges(second_node))
        for edge in second_node_out_edges:
            if edge.dst != third_node and not gtx_transformations_utils.is_reachable(
                start=edge.dst,
                target=third_node,
                state=graph,
            ):
                print(
                    "[RemoveAccessNodeCopies] There are edges from second node that end up in nodes other than the third node"
                )
                return False
        source_nodes_of_third_node = gtx_transformations_utils.find_upstream_nodes(
            start=third_node,
            state=graph,
        )
        if second_node not in source_nodes_of_third_node:
            print(
                "[RemoveAccessNodeCopies] There are nodes that write to the third node other than the second node"
            )
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

        edge_between_first_and_second = next(
            edge for edge in graph.edges() if edge.src == first_node and edge.dst == second_node
        )

        node_offset = [
            range_first[0] - range_second[0]
            for range_first, range_second in zip(
                edge_between_first_and_second.data.src_subset,
                edge_between_first_and_second.data.dst_subset,
            )
        ]
        for edge in graph.edges():
            if edge.data.data == second_node.data:
                edge.data.data = first_node.data
                if edge.data.dst_subset != tuple([0]):
                    if edge.data.dst_subset is not None:
                        for i, dst_subset_range in enumerate(edge.data.dst_subset):
                            new_subset = []
                            new_subset.append(dst_subset_range[0] + node_offset[i])
                            new_subset.append(dst_subset_range[1] + node_offset[i])
                            new_subset.append(dst_subset_range[2])
                            edge.data.dst_subset[i] = tuple(new_subset)
                    else:
                        for i, src_subset_range in enumerate(edge.data.src_subset):
                            new_subset = []
                            new_subset.append(src_subset_range[0] + node_offset[i])
                            new_subset.append(src_subset_range[1] + node_offset[i])
                            new_subset.append(src_subset_range[2])
                            edge.data.src_subset[i] = tuple(new_subset)

            if edge.data.data == third_node.data:
                edge.data.data = first_node.data
                if edge.data.dst_subset != tuple([0]):
                    if edge.data.dst_subset is not None:
                        for i, dst_subset_range in enumerate(edge.data.dst_subset):
                            new_subset = []
                            new_subset.append(dst_subset_range[0] + node_offset[i])
                            new_subset.append(dst_subset_range[1] + node_offset[i])
                            new_subset.append(dst_subset_range[2])
                            edge.data.dst_subset[i] = tuple(new_subset)
                    else:
                        for i, src_subset_range in enumerate(edge.data.src_subset):
                            new_subset = []
                            new_subset.append(src_subset_range[0] + node_offset[i])
                            new_subset.append(src_subset_range[1] + node_offset[i])
                            new_subset.append(src_subset_range[2])
                            edge.data.src_subset[i] = tuple(new_subset)

        second_node.data = first_node.data
        third_node.data = first_node.data
