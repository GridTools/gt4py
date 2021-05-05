# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
DaCe based horizontal executions merging.

Notes:
------
Merging is performed by merging the body of "right" into "left" within this module.
This is equivalent to merging the later into the earlier occurring horizontal execution
by order within the OIR. This is consistently reflected in variable and parameter names.
"""
from typing import Dict, List, Optional, Set, Tuple, Union

import dace
import dace.subsets
from dace import SDFGState
from dace.sdfg import graph
from dace.sdfg.utils import node_path_graph
from dace.transformation.transformation import PatternNode, Transformation

from gtc import oir
from gtc.dace.nodes import HorizontalExecutionLibraryNode
from gtc.passes.oir_optimizations.utils import AccessCollector


OFFSETS_T = Dict[str, Set[Tuple[int, int, int]]]
IJ_OFFSETS_T = Dict[str, Set[Tuple[int, int]]]


def masks_match(
    left: HorizontalExecutionLibraryNode, right: HorizontalExecutionLibraryNode
) -> bool:
    left_masks = left.as_oir().iter_tree().if_isinstance(oir.MaskStmt).to_set()
    right_masks = right.as_oir().iter_tree().if_isinstance(oir.MaskStmt).to_set()
    return left_masks == right_masks


def ij_offsets(offsets: OFFSETS_T) -> IJ_OFFSETS_T:
    return {field: {o[:2] for o in field_offsets} for field, field_offsets in offsets.items()}


def read_after_write_conflicts(left_writes: IJ_OFFSETS_T, right_reads: IJ_OFFSETS_T) -> Set[str]:
    return {
        field
        for field, offsets in right_reads.items()
        if field in left_writes and offsets ^ left_writes[field]
    }


def write_after_read_conflicts(left_reads: IJ_OFFSETS_T, right_writes: IJ_OFFSETS_T) -> Set[str]:
    return {
        field
        for field, offsets in right_writes.items()
        if field in left_reads and any(o != (0, 0) for o in offsets ^ left_reads[field])
    }


def offsets_match(
    left: HorizontalExecutionLibraryNode, right: HorizontalExecutionLibraryNode
) -> bool:
    left_accesses = AccessCollector.apply(left.as_oir())
    right_accesses = AccessCollector.apply(right.as_oir())
    conflicting = read_after_write_conflicts(
        ij_offsets(left_accesses.write_offsets()), ij_offsets(right_accesses.read_offsets())
    ) | write_after_read_conflicts(
        ij_offsets(left_accesses.read_offsets()), ij_offsets(right_accesses.write_offsets())
    )
    return not conflicting


def unwire_access_node(
    state: SDFGState,
    left: HorizontalExecutionLibraryNode,
    access: dace.nodes.AccessNode,
    right: HorizontalExecutionLibraryNode,
) -> None:
    out_removable = set(state.edges_between(access, right))
    for removable_edge in out_removable:
        state.remove_edge_and_connectors(removable_edge)


def rewire_edge(
    state: SDFGState,
    edge: graph.MultiConnectorEdge,
    **kwargs: Union[dace.nodes.AccessNode, HorizontalExecutionLibraryNode],
) -> None:
    src = kwargs.get("src", edge.src)
    src_conn = edge.src_conn
    dst = kwargs.get("dst", edge.dst)
    dst_conn = edge.dst_conn

    if src_conn and src_conn not in src.out_connectors:
        src.add_out_connector(src_conn)
    if dst_conn and dst_conn not in dst.in_connectors:
        dst.add_in_connector(dst_conn)

    state.remove_edge(edge)
    existing_edges = [
        e
        for e in state.edges_between(src, dst)
        if src_conn == e.src_conn and dst_conn == e.dst_conn
    ]
    if existing_edges:
        assert len(existing_edges) == 1
        existing_edges[0].data.subset = dace.subsets.union(
            edge.data.subset, existing_edges[0].data.subset
        )
    else:
        state.add_edge(
            src, src_conn, dst, dst_conn, dace.Memlet.simple(edge.data.data, edge.data.subset)
        )


def parallel_pattern(
    left: PatternNode, access: PatternNode, right: PatternNode
) -> graph.OrderedMultiDiGraph:
    pattern = graph.OrderedMultiDiGraph()
    pattern.add_node(access)
    pattern.add_node(left)
    pattern.add_edge(access, left, None)
    pattern.add_node(right)
    pattern.add_edge(access, right, None)
    return pattern


def optional_node(pattern_node: PatternNode, sdfg: dace.SDFG) -> Optional[dace.nodes.Node]:
    node = None
    try:
        node = pattern_node(sdfg)
    except KeyError:
        pass
    return node


@dace.registry.autoregister_params(singlestate=True)
class GraphMerging(Transformation):
    left = PatternNode(HorizontalExecutionLibraryNode)
    right = PatternNode(HorizontalExecutionLibraryNode)
    access = PatternNode(dace.nodes.AccessNode)
    access_thru = PatternNode(dace.nodes.AccessNode)

    @classmethod
    def expressions(cls) -> List[graph.OrderedMultiDiGraph]:
        return [
            node_path_graph(cls.left, cls.right),
            node_path_graph(cls.left, cls.access, cls.right, cls.access_thru),
            node_path_graph(cls.left, cls.access, cls.right),
            parallel_pattern(cls.left, cls.access, cls.right),
        ]

    def can_be_applied(
        self,
        graph: SDFGState,
        candidate: Dict[str, dace.nodes.Node],
        expr_index: int,
        sdfg: Union[dace.SDFG, SDFGState],
        strict: bool = False,
    ) -> bool:
        left = self.left(sdfg)
        access = optional_node(self.access, sdfg)
        right = self.right(sdfg)
        access_thru = optional_node(self.access_thru, sdfg)
        if access and access_thru and access.label != access_thru.label:
            return False
        return masks_match(left, right) and offsets_match(left, right)

    def apply(self, sdfg: dace.SDFG) -> None:
        state = sdfg.node(self.state_id)
        left = self.left(sdfg)
        access = optional_node(self.access, sdfg)
        right = self.right(sdfg)
        access_thru = optional_node(self.access_thru, sdfg)

        # rewire access chains
        if access and access_thru:
            rewire_edge(state, state.edges_between(left, access)[0], dst=access_thru)
            state.remove_node(access)
            access = None
            state.remove_edge(state.edges_between(right, access_thru)[0])

        # Disconnect the intermediate access node if it exists.
        # If it becomes an island, remove the access node.
        if access:
            unwire_access_node(state, left, access, right)
            if not state.all_edges(access):
                state.remove_node(access)

        # merge oir nodes
        res = HorizontalExecutionLibraryNode(
            oir_node=oir.HorizontalExecution(
                body=left.as_oir().body + right.as_oir().body,
                declarations=left.as_oir().declarations + right.as_oir().declarations,
            ),
            iteration_space=left.iteration_space,
        )
        state.add_node(res)
        # rewire edges and connectors to left and delete right
        for edge in state.edges_between(left, right):
            state.remove_edge_and_connectors(edge)
        for edge in state.in_edges(left):
            rewire_edge(state, edge, dst=res)
        for edge in state.in_edges(right):
            rewire_edge(state, edge, dst=res)
        for edge in state.out_edges(left):
            rewire_edge(state, edge, src=res)
        for edge in state.out_edges(right):
            rewire_edge(state, edge, src=res)
        state.remove_node(left)
        state.remove_node(right)
