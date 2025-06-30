# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional
from copy import deepcopy

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

        if not src_desc.transient:
            return False
        if not gtx_transformations.utils.is_view(src_desc, sdfg):
            return False
        if graph.in_degree(src_node) > 1:
            return False
        if not dst_desc.transient:
            return False
        first_node_out_edges = graph.out_edges(src_node)
        if len(first_node_out_edges) != 1 and first_node_out_edges[0].dst != dst_node:
            return False
        if len(graph.in_edges(src_node)) != 1:
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
        helpers.redirect_edge(graph, edge_to_redirect, new_dst=dst_in_edge.dst, new_dst_conn=dst_in_edge.dst_conn, new_memlet=dace.Memlet(dst_node.data))

        graph.remove_edge(dst_in_edge)
        graph.remove_node(src_node)
