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

    If the conditions above are met, then the transformation will copy the nodes from
    the second state into the first state and if needed create connections as needed.
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
        #  second state, must be connected to the respective node in the first state.
        #  We will now look for all nodes that are potentially read in the second
        #  state, these are all AccessNodes that have a non zero in degree.
        data_sources: dict[str, dace_nodes.AccessNode] = {
            dnode.data: dnode
            for dnode in first_state.data_nodes()
            if first_scope_dict[dnode] is None and first_state.in_degree(dnode) != 0
        }

        # Now we will look for all data sinks, i.e. nodes that are reading data from
        #  the first state. This are all AccessNodes not have a non zero out degree
        #  and are listed in `data_source`. Note that these nodes might have to be
        #  replaced with the nodes from the source.
        #  Note we can not use a `dict` here because it is possible, not fully legal
        #  though, that there are multiple access nodes that refers to the same data.
        data_sinks: list[dace_nodes.AccessNode] = [
            dnode
            for dnode in second_state.data_nodes()
            if second_scope_dict[dnode] is None
            and second_state.out_degree(dnode) != 0
            and dnode.data in data_sources
        ]
        assert all(second_state.in_degree(data_sink) == 0 for data_sink in data_sinks)

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
        for data_sink in data_sinks:
            data_source: dace_nodes.AccessNode = data_sources[data_sink.data]

            # Now modify the edge, there is no other modification needed.
            for old_edge in first_state.out_edges(data_sink):
                first_state.add_edge(
                    data_source, old_edge.src_conn, old_edge.dst, old_edge.dst_conn, old_edge.data
                )
            first_state.remove_node(data_sink)

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
