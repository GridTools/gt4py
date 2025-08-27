# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from typing import Any, Optional

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    transformation as dace_transformation,
)
from dace.sdfg import nodes as dace_nodes
from dace.transformation import helpers

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


@dace_properties.make_properties
class RemovePointwiseViews(dace_transformation.SingleStateTransformation):
    """
    Remove pointwise views from the SDFG.
    This transformation is used to remove views that are created for pointwise operations.
    It redirects the edges from the view to the original data and removes the view node.
    The view generation is non-deterministic and usually happens after reduction `Library` nodes.
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
        dst_desc: dace_data.Data = dst_node.desc(sdfg)

        scope = graph.scope_dict()

        # Currently handle only pointwise views inside a Map. If the view is not inside a Map,
        # it is probably not a pointwise view.
        if scope[src_node] is None or scope[dst_node] is None:
            return False  # Not inside a Map.

        if not src_desc.transient:
            return False
        if not gtx_transformations.utils.is_view(src_desc, sdfg):
            return False
        if graph.in_degree(src_node) > 1:
            return False
        if not dst_desc.transient:
            return False
        if graph.out_degree(src_node) != 1:
            return False
        if len(graph.in_edges(src_node)) != 1:
            return False
        if isinstance(src_desc, dace_data.ArrayView) and src_desc.total_size != 1:
            return False

        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        src_node: dace_nodes.AccessNode = self.source_node
        dst_node: dace_nodes.AccessNode = self.destination_node

        dst_in_edge = graph.in_edges(dst_node)[0]
        edge_to_redirect = graph.in_edges(src_node)[0]
        # Redirect the edge going into the view to the AccessNode that is the destination of the view.
        # new_dst: AccessNode that is the destination of the view.
        # new_dst_conn: Connection name of the destination AccessNode.
        # new_edge.data: The original Memlet of the edge_to_redirect.
        dst_in_edge_dst_subset = copy.deepcopy(dst_in_edge.data.get_dst_subset(dst_in_edge, graph))
        new_edge = helpers.redirect_edge(
            graph, edge_to_redirect, new_dst=dst_in_edge.dst, new_dst_conn=dst_in_edge.dst_conn
        )

        # If new_edge.data has the same data as the src_node, we need to update it to point to the dst_node
        # since the src_node is being removed. Else the new_edge.data can either be some other unrelated `data`
        # or dst_node.data. In this case we have to update the `other_subset` to match the dst_node.data other_subset
        # which is the subset which gets written to the dst of the edge.
        if new_edge.data.data == src_node.data:
            new_edge.data.data = dst_node.data
            new_edge.data.subset = dst_in_edge_dst_subset
        else:
            new_edge.data.other_subset = dst_in_edge_dst_subset

        graph.remove_edge(dst_in_edge)
        sdfg.remove_data(src_node.data, validate=False)
        graph.remove_node(src_node)
