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


# Conditional import because `gt4py.cartesian` uses an older DaCe version without
#  `explicit_cf_compatible`.
# TODO(phimuell): Remove once `gt4py.cartesian` has been updated.
try:
    explicit_cf_compatible = dace_transformation.explicit_cf_compatible
except AttributeError:
    explicit_cf_compatible = lambda x: x  # noqa: E731 [lambda-assignment]


@explicit_cf_compatible
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

        # If one of the states is empty, then we can accept it because it will not
        #  create any data access issues.
        if first_state.number_of_nodes() == 0:
            return True
        if second_state.number_of_nodes() == 0:
            return True

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

        if self._check_for_read_write_conflicts(sdfg):
            return False

        return True

    def _check_for_read_write_conflicts(
        self,
        sdfg: dace.SDFG,
    ) -> bool:
        """Return `True` if there are conflicts."""

        if self._check_for_read_write_dependencies(sdfg):
            return True
        if self._check_for_wcr_conflicts(sdfg):
            return True
        return False

    def _check_for_wcr_conflicts(
        self,
        sdfg: dace.SDFG,
    ) -> bool:
        """Checks if wcr edges prevent two consecutive states to be fused.

        If both states contains a write to the same data, and one of them has a set
        `wcr` then we can not merge the states.

        Note:
            In case all writes are `wcr` it might be possible to merge them, but
            this is a rather obscure case that we ignore.
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
        if len(common_write_data) == 0:
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

    def _check_for_read_write_dependencies(
        self,
        sdfg: dace.SDFG,
    ) -> bool:
        """Checks for read write conflicts in the global memory.

        The function checks if by the elimination of states creates read write
        conflicts. Because of the structure outlined by ADR-18 it only checks this
        for the global data, since transients are by definition written only once.

        Todo:
            Refine this function.
        """
        first_state: dace.SDFGState = self.first_state
        first_scope_dict = first_state.scope_dict()
        second_state: dace.SDFGState = self.second_state
        second_scope_dict = second_state.scope_dict()

        # Determine the concurrent subgraphs, i.e. the data flow that can, in theory
        #  execute in parallel. Note that this is not the same as connected components.
        #  As subgraphs might share nodes, although this should happen only for source
        #  nodes. For each of them we determine the set of data they write and on
        #  which data they depend on.
        first_subgraphs = dace.sdfg.utils.concurrent_subgraphs(first_state)
        data_producers: list[set[str]] = []
        data_producers_dependencies: list[set[str]] = []
        all_data_producers: set[str] = set()
        for first_subgraph in first_subgraphs:
            data_producers.append(
                {
                    dnode.data
                    for dnode in first_subgraph.data_nodes()
                    if first_scope_dict[dnode] is None and first_subgraph.in_degree(dnode) != 0
                }
            )
            data_producers_dependencies.append(
                {
                    dnode.data
                    for dnode in first_subgraph.data_nodes()
                    if first_scope_dict[dnode] is None and first_subgraph.out_degree(dnode) != 0
                }
            )
            assert all_data_producers.isdisjoint(
                data_producers[-1]
            ), "Found multiple AccessNodes that writes to data in one state."
            all_data_producers.update(data_producers[-1])

        # Now determine the concurrent subgraphs of the second state, i.e. the parts
        #  that in theory could execute in parallel. For every subgraph we determine
        #  which data it reads, that were written to in the first state, the so called
        #  messenger data. Components that share a messenger data will be merged
        #  together, if the states are fused. Furthermore, we determine the set of data
        #  a component writes to. Because if a component, from the first state, depends
        #  on data a component from the second state writes to and the two components
        #  are not related through a messenger data, then we end up with two
        #  independent subgraphs that read and write from the same global data.
        #  This is indeterministic behaviour, thus we should reject it.
        second_subgraphs = dace.sdfg.utils.concurrent_subgraphs(second_state)
        data_consumers: list[set[str]] = []
        data_consumers_influences: list[set[str]] = []
        messanger_data: set[str] = set()
        for second_subgraph in second_subgraphs:
            data_consumers.append(
                {
                    dnode.data
                    for dnode in second_subgraph.data_nodes()
                    if second_scope_dict[dnode] is None and dnode.data in all_data_producers
                }
            )
            data_consumers_influences.append(
                {
                    dnode.data
                    for dnode in second_subgraph.data_nodes()
                    if second_scope_dict[dnode] is None and second_subgraph.in_degree(dnode) != 0
                }
            )
            messanger_data.update(data_consumers[-1].intersection(all_data_producers))

        for data_producer, producer_dependency in zip(data_producers, data_producers_dependencies):
            produced_messenger = data_producer.intersection(messanger_data)

            if len(produced_messenger) == 0:
                # The component does not generate any messenger data. In that case
                #  we must ensure that it does not depend on any data that is
                #  modified by the second state.
                if not all(
                    producer_dependency.isdisjoint(consumer_influece)
                    for consumer_influece in data_consumers_influences
                ):
                    return True
                continue

            # The ID of the consumers that depend on the current producer. Note that
            #  we do not consider consumers here that fully depends on the producer.
            depending_consumers: list[int] = []

            for consumer_id, (data_consumer, consumer_influece) in enumerate(
                zip(data_consumers, data_consumers_influences)
            ):
                consumed_messenger = data_consumer.intersection(messanger_data)
                exchanged_messenger = produced_messenger.intersection(consumed_messenger)
                if len(exchanged_messenger) == 0:
                    # This component does not consume anything the first state
                    #  generates, this means it is an isolated component and remains
                    #  an isolated component. There is technically nothing to do.
                    #  However, there is the special case that it writes a variable
                    #  some component of the first state depends on. Strictly speaking
                    #  this violates ADR-18, but it happens in some scan expressions.
                    # We do not add the consumer to `depending_consumer` because there
                    #  is no connection.
                    if not producer_dependency.isdisjoint(consumer_influece):
                        # The consumer modifies something the producer needs. So we
                        #  can not fuse it.
                        return True

                else:
                    # There is some exchange between the producer and the consumer.
                    #  We assume that despite the merge the inner dependencies are
                    #  meet, i.e. all messenger materializes before the dependent
                    #  dataflow is executed. We thus have to ensure that the new
                    #  component will not affect any component that is not involved
                    #  inside the merge.
                    unaffected_producers: list[int] = [
                        producer_id
                        for producer_id, data_producer in enumerate(data_producers)
                        if data_producer.isdisjoint(data_consumer)
                    ]
                    dependencies_of_unaffected_producers: set[str] = set()
                    for unaffected_producer in unaffected_producers:
                        dependencies_of_unaffected_producers.update(
                            data_producers_dependencies[unaffected_producer]
                        )
                    if not dependencies_of_unaffected_producers.isdisjoint(consumer_influece):
                        return True

                    # In case the consumer did not consume everything inside the
                    #  producer add it to the list of dependent consumers. See
                    #  below for more.
                    # TODO(phimuell): Lift this or refine this.
                    if exchanged_messenger != produced_messenger:
                        depending_consumers.append(consumer_id)

            if len(depending_consumers) > 1:
                # If we have more than one depending consumer, then this means that
                #  several consumers would partially read the producer. This
                #  essentially creates concurrent dataflow within the component, which
                #  would lead to non-deterministic results, so we have to reject the
                #  merge.
                return True

        return False

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

        # We have to preserve the order this means that every node from the second
        #  state, must merge with the corresponding node from the first state.
        #  These are all nodes that have a non zero input degree. For convenience,
        #  we also add all other data that is read, except if it is not written
        # TODO(phimuell): In case of global data it might be possible that there are
        #   multiple AccessNodes that writes to the data. We currently ignore that case.
        data_producers: dict[str, dace_nodes.AccessNode] = {
            dnode.data: dnode
            for dnode in first_state.data_nodes()
            if (first_scope_dict[dnode] is None and first_state.in_degree(dnode) != 0)
        }

        # Now add everything that is read too, if it is not already present.
        for dnode in first_state.data_nodes():
            if not (first_state.in_degree(dnode) == 0 and first_state.out_degree(dnode) != 0):
                continue
            if dnode.data in data_producers:
                assert not dnode.desc(sdfg).transient
                continue
            data_producers[dnode.data] = dnode

        # Now we will look for all data consumers, i.e. nodes that are reading data
        #  from the first state. These are all AccessNodes that have a zero indegree
        #  and that are listed in `data_producers`. Note that these nodes might have
        #  to be replaced with the nodes from the first state.
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
