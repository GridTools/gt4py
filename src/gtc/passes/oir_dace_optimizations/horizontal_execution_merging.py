"""
DaCe based horizontal executions merging.

Notes:
------
Merging is performed by merging the body of "right" into "left" within this module.
This is equivalent to merging the later into the earlier occurring horizontal execution
by order within the OIR. This is consistently reflected in variable and parameter names.
"""
from typing import Dict, Set, Tuple

import dace
from dace import SDFGState
from dace.sdfg import graph
from dace.sdfg.utils import node_path_graph
from dace.transformation.optimizer import Optimizer
from dace.transformation.transformation import PatternNode, Transformation

from gtc.dace.nodes import HorizontalExecutionLibraryNode
from gtc.passes.oir_optimizations.utils import AccessCollector


OFFSETS_T = Dict[str, Set[Tuple[int, int, int]]]
IJ_OFFSETS_T = Dict[str, Set[Tuple[int, int]]]


def masks_match(
    left: HorizontalExecutionLibraryNode, right: HorizontalExecutionLibraryNode
) -> bool:
    return left.oir_node.mask == right.oir_node.mask


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
    left_accesses = AccessCollector.apply(left.oir_node)
    right_accesses = AccessCollector.apply(right.oir_node)
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


def rewire_edge(state: SDFGState, edge: graph.Edge, **kwargs) -> None:
    src = kwargs.get("src", edge.src)
    src_conn = kwargs.get("src_conn", edge.src_conn)
    dst = kwargs.get("dst", edge.dst)
    dst_conn = kwargs.get("dst_conn", edge.dst_conn)
    if src_conn and src_conn not in src.out_connectors:
        src.add_out_connector(src_conn)
    if dst_conn and dst_conn not in dst.in_connectors:
        dst.add_in_connector(dst_conn)
    state.remove_edge(edge)
    if [
        e
        for e in state.edges_between(src, dst)
        if src_conn == e.src_conn and dst_conn == e.dst_conn
    ]:
        return None
    state.add_edge(src, src_conn, dst, dst_conn, dace.Memlet())


def chained_access_pattern(left, access, right, access_chained):
    pattern = graph.OrderedMultiDiGraph()
    pattern.add_node(left)
    pattern.add_node(access)
    pattern.add_node(right)
    pattern.add_node(access_chained)
    pattern.add_edge(left, access, None)
    pattern.add_edge(access, right, None)
    pattern.add_edge(right, access_chained, None)
    return pattern


def multi_access_pattern(left, access, right, other):
    pattern = graph.OrderedMultiDiGraph()
    pattern.add_node(left)
    pattern.add_node(access)
    pattern.add_edge(left, access, None)
    pattern.add_node(right)
    pattern.add_edge(access, right, None)
    pattern.add_node(other)
    pattern.add_edge(access, other, None)
    return pattern


def parallel_pattern(left, access, right):
    pattern = graph.OrderedMultiDiGraph()
    pattern.add_node(access)
    pattern.add_node(left)
    pattern.add_edge(access, left, None)
    pattern.add_node(right)
    pattern.add_edge(access, right, None)
    return pattern


@dace.registry.autoregister_params(singlestate=True)
class _IntermediateAccessChained(Transformation):
    left = PatternNode(HorizontalExecutionLibraryNode)
    access = PatternNode(dace.nodes.AccessNode)
    right = PatternNode(HorizontalExecutionLibraryNode)
    access_chained = PatternNode(dace.nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [chained_access_pattern(cls.left, cls.access, cls.right, cls.access_chained)]

    @classmethod
    def can_be_applied(cls, graph, candidate, expr_index, sdfg, strict=False):
        access = graph.node(candidate[cls.access])
        access_chained = graph.node(candidate[cls.access_chained])
        if access.label != access_chained.label:
            return False
        return True

    def apply(self, sdfg):
        state = sdfg.node(self.state_id)
        left = self.left(sdfg)
        access = self.access(sdfg)
        right = self.right(sdfg)
        access_chained = self.access_chained(sdfg)

        rewire_edge(state, state.edges_between(left, access)[0], dst=access_chained)
        state.remove_node(access)
        state.remove_edge(state.edges_between(right, access_chained)[0])


@dace.registry.autoregister_params(singlestate=True)
class GraphMerging(Transformation):
    left = PatternNode(HorizontalExecutionLibraryNode)
    right = PatternNode(HorizontalExecutionLibraryNode)
    access = PatternNode(dace.nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [
            node_path_graph(cls.left, cls.right),
            node_path_graph(cls.left, cls.access, cls.right),
            parallel_pattern(cls.left, cls.access, cls.right),
        ]

    @classmethod
    def can_be_applied(cls, graph, candidate, expr_index, sdfg, strict=False):
        left = graph.node(candidate[cls.left])
        right = graph.node(candidate[cls.right])
        return masks_match(left, right) and offsets_match(left, right)

    def apply(self, sdfg):
        state = sdfg.node(self.state_id)
        left = self.left(sdfg)
        right = self.right(sdfg)

        # rewire access chains
        for match in Optimizer(sdfg).get_pattern_matches(patterns=[_IntermediateAccessChained]):
            if match.left(sdfg) == left and match.right(sdfg) == right:
                _IntermediateAccessChained.apply_to(
                    sdfg,
                    left=left,
                    access=match.access(sdfg),
                    right=right,
                    access_chained=match.access_chained(sdfg),
                    verify=True,
                    save=False,
                )

        # Disconnect the intermediate access node if it exists.
        # If it becomes an island, remove the access node.
        access_node = None
        try:
            access_node = self.access(sdfg)
            unwire_access_node(state, left, access_node, right)
            if not state.all_edges(access_node):
                state.remove_node(access_node)
        except KeyError:
            pass

        # merge oir nodes
        left.oir_node.body += right.oir_node.body

        # rewire edges and connectors to left and delete right
        for edge in state.edges_between(left, right):
            state.remove_edge_and_connectors(edge)
        for edge in state.in_edges(right):
            rewire_edge(state, edge, dst=left)
        for edge in state.out_edges(right):
            rewire_edge(state, edge, src=left)
        state.remove_node(right)
