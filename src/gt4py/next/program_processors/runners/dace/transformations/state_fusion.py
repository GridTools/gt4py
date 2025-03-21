# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import dace
from dace import transformation as dace_transformation
from dace.sdfg import nodes as dace_nodes, utils as dace_sdutils


# @dace_properties.make_properties
@dace_transformation.explicit_cf_compatible
class GT4PyStateFusion(dace_transformation.MultiStateTransformation):
    """Implements a state fusion transformation that is specific to GT4Py.

    Because there are, by construction, no write conflicts in transient this
    transformation can be much simpler, than the one in DaCe.
    It should not be seen as a replacement but as an extension.

    The transformation can only be applied in the following cases:
    - The first state has only one outgoing edge, that connects it with the second state.
    - The second only has one incoming edge, that connects is with the first state.
    - The connecting edge is unconditionally and does not contains any assignments.
    - If global data is written to in the first state and there is an AccessNode that
        reads and writes to the same data in the second state the transformation
        does not apply.

    If the conditions above are met, then the transformation will copy the nodes from
    the second state into the first state and if needed create connections as needed.

    Todo:
        Improve robustness if there are multiple global AccessNodes.
    """

    first_state = dace_transformation.PatternNode(dace.SDFGState)
    second_state = dace_transformation.PatternNode(dace.SDFGState)

    @classmethod
    def expressions(cls) -> Any:
        return [dace_sdutils.node_path_graph(cls.first_state, cls.second_state)]

    @staticmethod
    def annotates_memlets() -> bool:
        return False

    def can_be_applied(
        self, _: Any, expr_index: int, sdfg: dace.SDFG, permissive: bool = False
    ) -> bool:
        first_state: dace.SDFGState = self.first_state
        second_state: dace.SDFGState = self.second_state
        graph: dace.sdfg.state.AbstractControlFlowRegion = first_state.parent_graph

        if graph.out_degree(first_state) != 1:
            return False
        if graph.in_degree(second_state) != 1:
            return False

        conn_edge = next(iter(graph.out_edges(first_state)))
        if not conn_edge.data.is_unconditional():
            return False

        # TODO(phimuell): Lift this limitation.
        if len(conn_edge.data.assignments) != 0:
            return False

        # If the first state writes to global memory and the second state contains
        #  an AccessNode that reads and writes to the same global memory then we
        #  do not apply. This is a very obscure case.
        first_global_memory_write: set[str] = {
            dnode.data for dnode in first_state.data_nodes() if not dnode.desc(sdfg).transient
        }
        if any(
            dnode.name in first_global_memory_write
            for dnode in second_state.data_nodes()
            if second_state.in_degree(dnode) != 0 and second_state.out_degree(dnode) != 0
        ):
            return False

        return True

    def _move_nodes(self, sdfg: dace.SDFG) -> None:
        """Moves the nodes from the second state to the first state.

        The function will assign all nodes from the second state into the first state.
        It will also modify the data flow to preserve the order of execution. This
        means if a data container was written to in the first state and read in the
        second state, then the reads to the data will not use the AccessNode that was
        in the second state, instead they will be rerouted such that they use the
        one from the first state.
        It is important that this function will not mutate the second state, thus
        this function will produce an SDFG in an invalid state.
        """
        first_state: dace.SDFGState = self.first_state
        second_state: dace.SDFGState = self.second_state
        assert isinstance(first_state, dace.SDFGState)
        assert isinstance(second_state, dace.SDFGState)

        first_scope_dict = first_state.scope_dict()
        second_scope_dict = second_state.scope_dict()

        # We have to preserve the order this means that every source node of the
        #  second state, must merge the nodes from the first node with the
        #  corresponding nodes in the second state.
        #  By definition and ADR-18, these are all AccessNodes with a non zero
        #  input degree. Although not strictly needed, we also add globals that
        #  are read, but not written to in the first state, the effect is that
        #  there is only one reading AccessNode, which is closer to ADR-18.
        data_producers: dict[str, dace_nodes.AccessNode] = {
            dnode.data: dnode
            for dnode in first_state.data_nodes()
            if (first_scope_dict[dnode] is None and first_state.in_degree(dnode) != 0)
        }

        # Adding globals that that only read.
        for dnode in first_state.data_nodes():
            if dnode.desc(sdfg).transient:
                continue
            if not (first_state.in_degree(dnode) == 0 and first_state.out_degree(dnode) != 0):
                continue
            if dnode.data in data_producers:
                continue
            data_producers[dnode.data] = dnode

        # Now we will look for all data consumers, i.e. nodes that are reading data from
        #  the first state. This are all AccessNodes not have a zero in degree
        #  and are listed in `data_producers`. Note that these nodes might have to be
        #  replaced with the nodes from the source.
        #  Note we can not use a `dict` here because it is possible, not fully legal
        #  though, that there are multiple AccessNodes that refers to the same data.
        data_consumers: list[dace_nodes.AccessNode] = [
            dnode
            for dnode in second_state.data_nodes()
            if (
                second_scope_dict[dnode] is None
                and second_state.in_degree(dnode) == 0
                and dnode.data in data_producers
            )
        ]
        assert all(second_state.in_degree(consumer) == 0 for consumer in data_consumers)

        # Move the nodes and edges into from the second into the first state. However
        #  they are still part of the second state, this will be handled afterwards.
        for node in second_state.nodes():
            if isinstance(node, dace_nodes.NestedSDFG):
                node.sdfg.parent = first_state
            first_state.add_node(node)
        for src, src_conn, dst, dst_conn, data in second_state.edges():
            first_state.add_edge(src, src_conn, dst, dst_conn, data)

        # The first state generated data that the second state consumed, since they
        #  are now in one single state, we have to ensure an order. For this we simply
        #  merge the two nodes, which means that instead of reading from the original
        #  AccessNode (part of the second state) it should now read from the AccessNode
        #  from the first state.
        for data_consumer in data_consumers:
            data_producer: dace_nodes.AccessNode = data_producers[data_consumer.data]

            # Now modify the edge, there is no other modification needed.
            for old_edge in first_state.out_edges(data_consumer):
                first_state.add_edge(
                    data_producer, old_edge.src_conn, old_edge.dst, old_edge.dst_conn, old_edge.data
                )
            first_state.remove_node(data_consumer)

        # For ADR-18 compatibility we now have to ensure that there is only one sink
        #  node for any global data in the combined state. The transients are handled
        #  naturally because we maintain the SSA invariant, furthermore, the inputs
        #  are already merged.
        # TODO(phimuell): Improve this merging.
        global_sink_nodes: dict[str, set[dace_nodes.AccessNode]] = {}
        for sink_node in first_state.sink_nodes():
            if not isinstance(sink_node, dace_nodes.AccessNode):
                continue
            if sink_node.desc(sdfg).transient:
                continue
            if sink_node.data not in global_sink_nodes:
                global_sink_nodes[sink_node.data] = set()
            global_sink_nodes[sink_node.data].add(sink_node)

        for sink_nodes in global_sink_nodes.values():
            if len(sink_nodes) <= 1:
                continue
            # We now select one node and redirect all writes to it.
            final_sink_node = sink_nodes.pop()
            for sink_node in sink_nodes:
                for iedge in first_state.in_edges(sink_node):
                    first_state.add_edge(
                        iedge.src, iedge.src_conn, final_sink_node, iedge.dst_conn, iedge.data
                    )
                first_state.remove_node(sink_node)

    def apply(self, _: Any, sdfg: dace.SDFG) -> None:
        first_state: dace.SDFGState = self.first_state
        second_state: dace.SDFGState = self.second_state
        graph: dace.sdfg.state.AbstractControlFlowRegion = first_state.parent_graph

        # Move the nodes from the second into the first state.
        self._move_nodes(sdfg)

        # Now redirect all state edges and remove the second state.
        # NOTE: It is important that we do not remove the first state. If we would
        #  remove it, then we would have to modify the start block property of the
        #  graph.
        dace_sdutils.change_edge_src(graph, second_state, first_state)
        graph.remove_node(second_state)
