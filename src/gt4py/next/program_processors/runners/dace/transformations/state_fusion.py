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


@dace_transformation.explicit_cf_compatible
class GT4PyStateFusion(dace_transformation.MultiStateTransformation):
    """Implements a state fusion transformation that is specific to GT4Py.

    Because there are, by construction, no write conflicts in transient this
    transformation can be much simpler, than the one in DaCe.
    It should not be seen as a replacement but as an extension.

    The transformation can only be applied in the following cases:
    - The first state has only one outgoing edge, that connects it with the second state.
    - The second only has one incoming edge, that connects it with the first state.
    - The connecting edge is unconditional and does not contain any assignments.
    - If global data is written to in the first state and there is an AccessNode that
        reads and writes to the same data in the second state the transformation
        does not apply.
    - No read writes conflicts are created.

    If the conditions above are met, then the transformation will copy the nodes from
    the second state into the first state and if needed create connections as needed.

    Todo:
        - Add a step that merges global AccessNodes together, this is borderline
            ADR-18 compatibility.
        - If both states read from the same transient data, that was not defined in
            either, then the AccessNodes are not merged.
        - Refactor this transformation such that it pre compute more.
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
        #  an AccessNode that reads and writes to the same global memory. In this case
        #  we do not know how to merge the nodes together. Thus we reject these cases
        # NOTE: This is a very obscure case.
        first_global_memory_write: set[str] = {
            dnode.data
            for dnode in first_state.data_nodes()
            if not dnode.desc(sdfg).transient and first_state.in_degree(dnode) != 0
        }
        if any(
            dnode.data in first_global_memory_write
            for dnode in second_state.data_nodes()
            if second_state.in_degree(dnode) != 0 and second_state.out_degree(dnode) != 0
        ):
            return False

        if self._check_for_read_write_conflicts(sdfg=sdfg):
            return False

        return True

    def _check_for_read_write_conflicts(
        self,
        sdfg: dace.SDFG,
    ) -> bool:
        """Return `True` if there are conflicts."""

        if self._check_for_transient_conflicts(sdfg=sdfg):
            return True
        if self._check_for_global_read_write_conflicts(sdfg=sdfg):
            return True
        if self._check_for_wcr_conflicts(sdfg=sdfg):
            return True
        return False

    def _check_for_transient_conflicts(
        self,
        sdfg: dace.SDFG,
    ) -> bool:
        """Tests if there are conflicts involving transients.

        The function checks if there is a write to data in the second state, that
        was used inside in the first state.

        Note that the pattern this function looks for violates ADR-18, but it is
        required to support scan operation. It appears inside scan loops, where
        the state variable or carry, is written to in multiple locations.
        """
        first_state: dace.SDFGState = self.first_state
        first_scope_dict = first_state.scope_dict()
        second_state: dace.SDFGState = self.second_state
        second_scope_dict = second_state.scope_dict()

        # Find the set of transients that are written to in the first state.
        first_transient_reads: set[str] = {
            dnode.data
            for dnode in first_state.data_nodes()
            if (
                first_scope_dict[dnode] is None
                and dnode.desc(sdfg).transient
                and first_state.out_degree(dnode) != 0
            )
        }
        if any(
            dnode.data in first_transient_reads
            for dnode in second_state.data_nodes()
            if (second_scope_dict[dnode] is None and second_state.in_degree(dnode) != 0)
        ):
            return True
        return False

    def _check_for_wcr_conflicts(
        self,
        sdfg: dace.SDFG,
    ) -> bool:
        """Checks if wcr edges prevent two consecutive states to be fused.

        If both states contains a write to the same data, and one of them has a set
        `wcr` then we can not merge the states. Note that we allow the case that
        the first state contains a `wcr` write to a data and the second state contains
        a read to it.
        """
        first_state: dace.SDFGState = self.first_state
        first_scope_dict = first_state.scope_dict()
        second_state: dace.SDFGState = self.second_state
        second_scope_dict = second_state.scope_dict()

        # Collect the AccessNodes that are written to in the first state.
        first_state_writes: list[dace_nodes.AccessNode] = [
            dnode
            for dnode in first_state.data_nodes()
            if first_scope_dict[dnode] is None and first_state.in_degree(dnode) != 0
        ]
        first_state_writes_names: set[str] = {ac.data for ac in first_state_writes}

        # Now collect all the AccessNodes that are written to in the second state.
        second_state_writes: list[dace_nodes.AccessNode] = [
            dnode
            for dnode in second_state.data_nodes()
            if second_scope_dict[dnode] is None and second_state.in_degree(dnode) != 0
        ]
        second_state_writes_names: set[str] = {ac.data for ac in second_state_writes}

        # If there is no data that is written to in both state then there is no conflict.
        common_write_data: set[str] = first_state_writes_names.intersection(
            second_state_writes_names
        )
        if second_state_writes_names.isdisjoint(first_state_writes_names):
            return False

        for state, ac_nodes in [
            (first_state, first_state_writes),
            (second_state, second_state_writes),
        ]:
            for ac in ac_nodes:
                if ac.data not in common_write_data:
                    continue
                if any(iedge.data.wcr is not None for iedge in state.in_edges(ac)):
                    return True

        return False

    def _check_for_global_read_write_conflicts(
        self,
        sdfg: dace.SDFG,
    ) -> bool:
        """Checks for read write conflicts in the global memory.

        The function checks if by the elimination of states creates read write
        conflicts. Because of the structure outlined by ADR-18 it only checks this
        for the global data, since transients are by definition written only once.
        """
        first_state: dace.SDFGState = self.first_state
        first_scope_dict = first_state.scope_dict()
        second_state: dace.SDFGState = self.second_state
        second_scope_dict = second_state.scope_dict()

        # First we find all "messenger data" these are the data that is defined, i.e.
        #  written to, in the first state and read in the second state. It is important
        #  that this is not limited to transients.
        data_producers: list[dace_nodes.AccessNode] = [
            dnode
            for dnode in first_state.data_nodes()
            if first_scope_dict[dnode] is None and first_state.in_degree(dnode) != 0
        ]
        data_producers_names: set[str] = {ac.data for ac in data_producers}

        data_consumers: list[dace_nodes.AccessNode] = [
            dnode
            for dnode in second_state.data_nodes()
            if second_scope_dict[dnode] is None and dnode.data in data_producers_names
        ]

        # There are no data exchange between the two states.
        if len(data_consumers) == 0:
            return False

        # For every message data find the global data that it influences, i.e. all
        #  global data that are downstream reachable from that node.
        consumer_destinations: dict[str, set[str]] = {}
        for data_consumer in data_consumers:
            consumer_destination = self._find_global_destination_data_for(
                dnode=data_consumer,
                state=second_state,
                sdfg=sdfg,
            )
            if len(consumer_destination) != 0:
                if data_consumer.data not in consumer_destinations:
                    consumer_destinations[data_consumer.data] = set()
                consumer_destinations[data_consumer.data].update(consumer_destination)

        # The second state does not write into any global data, so there are no
        #  read-write conflicts to check.
        if len(consumer_destinations) == 0:
            return False

        # Now we check if there are read write conflicts. The process is quite simple,
        #  for every messenger data we determine the set of global data it depends on.
        #  Then we look at the set of global data the other _other_ coupling transients
        #  write to. If there is an intersection, i.e. one coupling transient depends on
        #  a certain global data and another coupling transient writes to it, then
        #  we consider this as read write conflict.
        # TODO(phimuell): This is a very simple solution, that does not takes into
        #   account Maps. We should actually focus on concurrent dataflow, i.e. data
        #   flow that is not connected.
        for source_node in data_producers:
            coupling_data = source_node.data
            if coupling_data not in consumer_destinations:
                continue
            global_sources: set[str] = self._find_global_source_data_of(
                dnode=source_node,
                state=first_state,
                sdfg=sdfg,
            )
            if any(
                not global_sources.isdisjoint(global_destination)
                for dname, global_destination in consumer_destinations.items()
                if dname != coupling_data
            ):
                return True

        return False

    def _find_global_source_data_of(
        self,
        dnode: dace_nodes.AccessNode,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> set[str]:
        """Finds the global data `dnode` depends on in `state`.

        Essentially traverse the dataflow graph in reverse order. After all nodes
        have been visited the set of global data that was encountered is returned.
        """
        global_data: set[str] = set()
        to_visit: list[dace_nodes.Node] = [dnode]
        seen: set[dace_nodes.Node] = set()

        while len(to_visit) > 0:
            node = to_visit.pop()
            if node in seen:
                continue
            elif isinstance(node, dace_nodes.AccessNode) and not node.desc(sdfg).transient:
                global_data.add(node.data)
            to_visit.extend(iedge.src for iedge in state.in_edges(node))
            seen.add(node)

        return global_data

    def _find_global_destination_data_for(
        self,
        dnode: dace_nodes.AccessNode,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> set[str]:
        """Find the global data in to which `dnode` is written to.

        Essentially this function traverse the graph starting from `dnode`. It
        will record all AccessNodes that refers to the data it encounters on the way.
        """
        global_data: set[str] = set()
        to_visit: list[dace_nodes.Node] = [dnode]
        seen: set[dace_nodes.Node] = set()

        while len(to_visit) > 0:
            node = to_visit.pop()
            if node in seen:
                continue
            elif isinstance(node, dace_nodes.AccessNode) and not node.desc(sdfg).transient:
                global_data.add(node.data)
            to_visit.extend(oedge.dst for oedge in state.out_edges(node))
            seen.add(node)

        return global_data

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
        # TODO(phimuell): In case of global data it might be possible that there are
        #   multiple AccessNodes that writes to the data. We currently ignore that case.
        data_producers: dict[str, dace_nodes.AccessNode] = {
            dnode.data: dnode
            for dnode in first_state.data_nodes()
            if (first_scope_dict[dnode] is None and first_state.in_degree(dnode) != 0)
        }

        # Add the AccessNodes from the first state that read from global memory.
        #  However, if there is an AccessNodes that writes to it use that one.
        for dnode in first_state.data_nodes():
            if dnode.desc(sdfg).transient:
                continue
            if not (first_state.in_degree(dnode) == 0 and first_state.out_degree(dnode) != 0):
                continue
            if dnode.data in data_producers:
                continue
            data_producers[dnode.data] = dnode

        # Now we will look for all data consumers, i.e. nodes that are reading data
        #  from the first state. These are all AccessNodes that have a zero indegree
        #  and that are listed in `data_producers`. Note that these nodes might have
        #  to be replaced with the nodes from the source.
        #  Note we can not use a `dict` here because it is possible, although not
        #  fully compliant to ADR-18 though, that there are multiple AccessNodes
        #  referring to the same data.
        data_consumers: list[dace_nodes.AccessNode] = [
            dnode
            for dnode in second_state.data_nodes()
            if (
                second_scope_dict[dnode] is None
                and second_state.in_degree(dnode) == 0
                and dnode.data in data_producers
            )
        ]

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
        #  AccessNode (that was part of the second state) it should now read from the
        #  AccessNode that was defined in the first state. If there were multiple
        #  AccessNodes in the second state, that refer to the same data, they are
        #  all merged.
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
