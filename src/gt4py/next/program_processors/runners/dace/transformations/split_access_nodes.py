# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Iterable, Optional

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    subsets as dace_sbs,
    symbolic as dace_sym,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


def gt_split_access_nodes(
    sdfg: dace.SDFG,
    validate: bool = False,
    validate_all: bool = False,
    single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
) -> Optional[int]:
    """Applies the `SplitAccessNode` transformation to the SDFG.

    The transformation returns the number of AccessNodes that have been split.

    Args:
        sdfg: The SDFG to process.
        validate: Perform validation after the pass has run.
        validate_all: Perform extensive validation.
        single_use_data: Which data descriptors are used only once.
            If not passed the function will run `FindSingleUseData`.
    """

    # To ensures that the `{src,dst}_subset` are properly set, run initialization.
    #  See [issue 1703](https://github.com/spcl/dace/issues/1703)
    for state in sdfg.states():
        for edge in state.edges():
            edge.data.try_initialize(sdfg, state, edge)

    if single_use_data is None:
        find_single_use_data = dace_analysis.FindSingleUseData()
        single_use_data = find_single_use_data.apply_pass(sdfg, None)

    return sdfg.apply_transformations_repeated(
        SplitAccessNode(single_use_data=single_use_data),
        validate=validate,
        validate_all=validate_all,
    )


@dace_properties.make_properties
class SplitAccessNode(dace_transformation.SingleStateTransformation):
    """The transformation will split an AccessNode into multiple ones.

    If there is no interesection between a write and different reads,
    i.e. if every read to the AccessNode can be satisfied by a single write
    to the AccessNode and the AccessNode is only used at one location,
    then the node is split.
    This means that the reads will be satisfied directly and the node
    does not have to materialize.

    Args:
        single_use_data: The list of data that is used only once.

    Todo:
        Currently for every consumer there can only be one producer. In case the
        consumer is a Map, this makes sense, but in case the consumer is an
        AccessNode one could split the consumer.
    """

    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and be passed at construction time. Needed until
    #  [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: dict[dace.SDFG, set[str]]

    def __init__(
        self,
        *args: Any,
        single_use_data: dict[dace.SDFG, set[str]],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._single_use_data = single_use_data

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.access_node)]

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        access_node: dace_nodes.AccessNode = self.access_node
        desc = access_node.desc(sdfg)

        # The intermediate access node must be a single use data, because we will
        #  get rid of it, and it must be a transient and a non-view element.
        if access_node.data not in self._single_use_data[sdfg]:
            return False
        if not desc.transient:
            return False
        if gtx_transformations.utils.is_view(desc, sdfg):
            return False

        # There must be multiple producers, otherwise this transformation
        #  does not make sense.
        number_of_producers = graph.in_degree(access_node)
        if number_of_producers <= 1:
            return False

        # We also require, that every producer is distinct.
        # NOTE: That this is for simplifying the implementation.
        if number_of_producers != len({producer for producer in graph.in_edges(access_node)}):
            return False

        # To make sense there must also be different consumers.
        number_of_consumers = graph.out_degree(access_node)
        if number_of_consumers <= 1:
            return False

        # Furthermore, they must also be different distinct consumers.
        # NOTE: This is for simplifying the implementation.
        if number_of_consumers != len({consumer for consumer in graph.out_edges(access_node)}):
            return False

        # Now check if a decomposition exist.
        assignment = self._match_consumers_to_producers(graph)
        if assignment is None:
            return False
        if not self._check_spliting_constraints(
            state=graph,
            sdfg=sdfg,
            assignment=assignment,
        ):
            return False

        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        access_node: dace_nodes.AccessNode = self.access_node

        assignment = self._match_consumers_to_producers(graph)
        assert assignment is not None

        for producer_edge, consumer_edges in assignment.items():
            self._reroute_consumer(
                producer_edge=producer_edge,
                consumer_edges=consumer_edges,
                sdfg=sdfg,
                state=graph,
            )

        # Remove the old intermediate that is no longer needed.
        # TODO(phimuell): Make this smarter.
        assert graph.degree(access_node) == 0
        graph.remove_node(access_node)
        sdfg.remove_data(access_node.data, validate=False)

    def _match_consumers_to_producers(
        self,
        state: dace.SDFGState,
    ) -> dict[dace_graph.MultiConnectorEdge, set[dace_graph.MultiConnectorEdge]] | None:
        """For each incoming (writing) edge, find the edges that read the data.

        The function will go through each outgoing edge and determine which
        incoming edge write that data. If it is not possible to clearly assign
        each consumer to a producer then `None` is returned.

        No additional feasibility checks are performed.
        """
        access_node: dace_nodes.AccessNode = self.access_node

        # Which input edge can cover which output edge assignment
        assignment: dict[dace_graph.MultiConnectorEdge, set[dace_graph.MultiConnectorEdge]] = {}
        for iedge in state.in_edges(access_node):
            # TODO(phimuell): Lift this.
            if iedge.data.dst_subset is None:
                return None
            assignment[iedge] = set()

        # Now match the outgoing edges to their incoming producers.
        for oedge in state.out_edges(access_node):
            possible_producer = self._find_producer(oedge, assignment.keys())
            if possible_producer is None:
                return None
            assignment[possible_producer].add(oedge)

        # At least every producer should have at least one consumer. If this is not the
        #  case then we compute something that is not needed.
        # TODO(phimuell): Figuring out what we should actually do.
        assert not any(len(assigned_consumers) == 0 for assigned_consumers in assignment.values())

        return assignment

    def _find_producer(
        self,
        consumer_edge: dace_graph.MultiConnectorEdge,
        producer_edges: Iterable[dace_graph.MultiConnectorEdge],
    ) -> dace_graph.MultiConnectorEdge | None:
        """Find the producer edge that generates what the consumer reads.

        The function checks which producer covers what the consumer reads.
        If there is not producer that does this, this function returns `None`.
        This function does not perform any additional tests.

        Args:
            consumer_edge: The edge that reads from `self.access_node`.
            producer_edges: List of all edges that writes to `self.access_node`.
        """
        consumer_subset = consumer_edge.data.src_subset

        # The consumer subset does not exist, so we can not do the decomposition.
        # TODO(phimuell): Fix this.
        if consumer_subset is None:
            return None

        # This check only checks if that the producer really generates the data that
        #  is consumed later. However, we also have to ensure that nothing is computed
        #  what is not consumed later. Thus.
        possible_producers = [
            producer_edge
            for producer_edge in producer_edges
            if producer_edge.data.dst_subset.covers(consumer_subset)
        ]

        # We only allow the case that one producer covers the consumer. If we found
        #  multiple candidates then we have an invalid SDFG, because multiple
        #  producer writes to the same memory location.
        if len(possible_producers) == 0:
            return None
        elif len(possible_producers) != 1:
            raise ValueError(
                f"Found an invalid SDFG, there are multiple producer for '{self.access_node.data}"
            )
        return possible_producers[0]

    def _check_spliting_constraints(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
        assignment: dict[dace_graph.MultiConnectorEdge, set[dace_graph.MultiConnectorEdge]],
    ) -> bool:
        """Checks if the decomposition can be handled.

        Perform the feasibility tests that were not performed by
        `_match_consumers_to_producers()`. The function returns `True` if the node
        can be split.
        """

        for producer_edge, consumer_edges in assignment.items():
            data_source = producer_edge.src

            if len(consumer_edges) == 0:
                continue

            # If the producer edge or any consumer edge has an active WCR we
            #  do not apply.
            if producer_edge.data.wcr is not None:
                return False
            if any(consumer_edge.data.wcr is not None for consumer_edge in consumer_edges):
                return False

            if isinstance(data_source, dace_nodes.AccessNode):
                # TODO(phimuell): Should we also ensure that the domains are tight?
                if gtx_transformations.utils.is_view(data_source, sdfg):
                    return False

                # If the source is a global data, then we do not impose any other
                #  constraints.
                if not data_source.desc(sdfg).transient:
                    continue

                # If the source is a transient then we distinguish between the cases
                #  that there is only one consumer, in which case we require that what
                #  produced must be read everything and multiple consumer, in which
                #  case we do not impose further restrictions. We do this to ensure
                #  the tightness of the temporaries, i.e. what is computed is also
                #  read, which is core assumption of the `CopyChainRemover`.
                # TODO(phimuell): Lift this limitation.
                if len(consumer_edges) == 1:
                    if not next(iter(consumer_edges)).data.src_subset.covers(
                        producer_edge.data.dst_subset
                    ):
                        return False

            elif isinstance(data_source, dace_nodes.MapExit):
                # The source is a Map, in this case we just generate a new transient
                #  output and then perform some reconnection. However, we require that
                #  all consumer read exactly what is is written by the map. This
                #  is to ensure some tightness of the domains.
                if not all(
                    consumer_edge.data.src_subset.covers(producer_edge.data.dst_subset)
                    for consumer_edge in consumer_edges
                ):
                    return False

            else:
                return False
        return True

    def _reroute_consumer(
        self,
        producer_edge: dace_graph.MultiConnectorEdge,
        consumer_edges: set[dace_graph.MultiConnectorEdge],
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        """Perform the rerouting for `producer_edge`.

        Essentially, instead of writing first into `self.access_node`, reconnect
        the data flow such that the writing happens directly into `consumer_edges`.

        The function will remove `producer_edge` but not `self.access_node` from
        the state.

        Args:
            producer_edge: The edge generating the data.
            consumer_edges: List of all consumer edges that read from data
                generated by `producer_edge`.
            state: The state in which we operate.
            sdfg: The SDFG in which we operate.
        """
        data_producer: dace_nodes.Node = producer_edge.src

        if isinstance(data_producer, dace_nodes.MapExit):
            self._reroute_consumer_map_producer(
                producer_edge=producer_edge,
                consumer_edges=consumer_edges,
                sdfg=sdfg,
                state=state,
            )

        elif isinstance(data_producer, dace_nodes.AccessNode):
            self._reroute_consumer_access_node_producer(
                producer_edge=producer_edge,
                consumer_edges=consumer_edges,
                sdfg=sdfg,
                state=state,
            )
        else:
            raise NotImplementedError(
                f"Can not handle a '{data_producer.__class__.__name__}' producer."
            )

    def _reroute_consumer_access_node_producer(
        self,
        producer_edge: dace_graph.MultiConnectorEdge,
        consumer_edges: set[dace_graph.MultiConnectorEdge],
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        """Perform the rerouting if the producer is an AccessNode.

        Essentially, this function will the reads from `self.access_node`, of the
        data that is generated by `producer_edge`, such that the reads now directly
        go to `producer_edge.src`.
        The function will delete the old producer edge and the replaced consumer
        edges.

        Args:
            producer_edge: The edge generating the data.
            consumer_edges: List of all consumer edges that read from data
                generated by `producer_edge`.
            state: The state in which we operate.
            sdfg: The SDFG in which we operate.
        """
        access_node: dace_nodes.AccessNode = producer_edge.dst
        data_producer: dace_nodes.AccessNode = producer_edge.src

        old_producer_read = producer_edge.data.src_subset.min_element()
        old_producer_write = producer_edge.data.dst_subset.min_element()

        reconfigured_consumer: set[tuple[dace_nodes.Node, str]] = set()
        new_consumer_edges: list[dace_graph.MultiConnectorEdge] = []
        for consumer_edge in consumer_edges:
            consumer_node: dace_nodes.Node = consumer_edge.dst
            consumer_conn = consumer_edge.dst_conn

            # Index from where the consumer should start reading from the producer
            #  directly. But since it has gone through the intermediate AccessNode,
            #  the indexes are different and we have to compute them. At the end it
            #  is some kind of projection starting at what was read from the
            #  intermediate to the indexes where it has been written to, to the
            #  intermediate and finally from where it is originally coming from.
            #  We only do this projection for the start index, the end index
            #  is computed by adding the length to the start index.
            old_consumer_read = consumer_edge.data.src_subset.min_element()
            consumer_read_size = consumer_edge.data.src_subset.size()
            consumer_direct_read: list[
                tuple[dace_sym.SymbolicType, dace_sym.SymbolicType, int]
            ] = []
            for i in range(len(old_producer_read)):
                old_producer_read_start = old_producer_read[i]
                old_consumer_read_start = old_consumer_read[i]
                old_producer_write_start = old_producer_write[i]
                transfer_size = consumer_read_size[i]

                consumer_direct_read_start = dace.symbolic.pystr_to_symbolic(
                    f"({old_producer_read_start}) + (({old_consumer_read_start}) - ({old_producer_write_start}))",
                    simplify=True,
                )
                # The `-1` is because the end is considered inclusive in DaCe.
                consumer_direct_read_end = dace.symbolic.pystr_to_symbolic(
                    f"({consumer_direct_read_start}) + ({transfer_size}) - 1", simplify=True
                )
                consumer_direct_read.append(
                    (consumer_direct_read_start, consumer_direct_read_end, 1)
                )

            new_consumer_direct_read_subset = dace_sbs.Range(consumer_direct_read)

            # Create a new edge that reads from the producer directly and remove the
            #  old edge.
            new_consumer_edge = state.add_edge(
                producer_edge.src,
                producer_edge.src_conn,
                consumer_edge.dst,
                consumer_edge.dst_conn,
                dace.Memlet(
                    data=data_producer.data,
                    subset=new_consumer_direct_read_subset,
                    other_subset=consumer_edge.data.dst_subset,
                    dynamic=consumer_edge.data.dynamic or producer_edge.data.dynamic,
                ),
            )
            state.remove_edge(consumer_edge)
            new_consumer_edges.append(new_consumer_edge)

            # If needed reconfigure the consumers since it now involves the
            #  original producer.
            #  The stride propagation is done after all edges have been updated.
            if (consumer_node, consumer_conn) not in reconfigured_consumer:
                reconfigured_consumer.add((consumer_node, consumer_conn))

                # The subset correct we have to apply to the consumer depends on the
                #  type of the consumer.
                if isinstance(data_producer.desc(sdfg), dace_data.Scalar):
                    # The producer is a scalar so we just set the subset to `0`,
                    #  which is indicated by `None`.
                    consumer_subset_correction = None

                elif isinstance(consumer_node, dace_nodes.AccessNode):
                    # Here we only have to consider the offset between the reading
                    #  from the intermediate and where we write into it. The minus
                    #  is because we have to subtract this value.
                    consumer_subset_correction = [
                        dace.symbolic.pystr_to_symbolic(
                            f"-(({old_consumer_read_start}) - ({old_producer_write_start}))",
                            simplify=True,
                        )
                        for old_consumer_read_start, old_producer_write_start in zip(
                            old_consumer_read,
                            old_producer_write,
                            strict=True,
                        )
                    ]
                elif isinstance(consumer_node, dace_nodes.MapEntry):
                    # Things are different here, because the map ranges (and the
                    #  offsets in the inner Memlets handle everything, Thus, in
                    #  this case we have to correct the case that they now read
                    #  from the producer instead of the intermediate.
                    consumer_subset_correction = [
                        dace.symbolic.pystr_to_symbolic(
                            f"-(({old_producer_write_start}) - ({old_producer_read_start}))",
                            simplify=True,
                        )
                        for old_producer_read_start, old_producer_write_start in zip(
                            old_producer_read,
                            old_producer_write,
                            strict=True,
                        )
                    ]

                elif isinstance(consumer_node, dace_nodes.NestedSDFG):
                    # Since a NestedSDFG can only read from an AccessNode there is
                    #  nothing to do, except for the stride propagation, which is
                    #  done later.
                    continue

                else:
                    raise TypeError(
                        f"Can not correct a consumer of type '{type(consumer_node).__name__}'"
                    )

                gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                    is_producer_edge=False,
                    new_edge=new_consumer_edge,
                    ss_offset=consumer_subset_correction,
                    state=state,
                    sdfg=sdfg,
                    old_node=access_node,
                    new_node=data_producer,
                )
                reconfigured_consumer.add((consumer_node, consumer_conn))

        # All data has been rerouted, so remove the write to the old intermediate
        #  and propagate the now strides.
        state.remove_edge(producer_edge)
        processed_nsdfgs: set = set()
        for new_consumer_edge in new_consumer_edges:
            gtx_transformations.gt_map_strides_to_dst_nested_sdfg(
                outer_node=data_producer,
                edge=new_consumer_edge,
                sdfg=sdfg,
                state=state,
                processed_nsdfgs=processed_nsdfgs,
            )

    def _reroute_consumer_map_producer(
        self,
        producer_edge: dace_graph.MultiConnectorEdge,
        consumer_edges: set[dace_graph.MultiConnectorEdge],
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        """Perform the rerouting in case where the data producer is a Map.

        The function will create a new intermediate node, into which the map will
        write to. The corresponding reads will be redirected such that they will
        go to this new intermediate node.

        Args:
            producer_edge: The edge generating the data.
            consumer_edges: List of all consumer edges that read from data
                generated by `producer_edge`.
            state: The state in which we operate.
            sdfg: The SDFG in which we operate.


        Todo:
            Try to merge this function with the function for handling AccessNode
            producers.
        """
        access_node: dace_nodes.AccessNode = producer_edge.dst
        access_node_desc = access_node.desc(sdfg)
        producer_subset = producer_edge.data.dst_subset

        # Create the intermediate storage.
        tmp_name, tmp_desc = sdfg.add_temp_transient(
            shape=producer_subset.size(),
            dtype=access_node_desc.dtype,
            storage=access_node_desc.storage,
        )
        tmp_node: dace_nodes.AccessNode = state.add_access(tmp_name)

        # Subset for the modification :
        #  Before, the Map was writing _somewhere_ into `access_nodes`, but now it
        #  writes into `tmp_node`. Thus we have to modify the Memlets inside the Map.
        #  The minus is because the correction is added, but we need a subtraction.
        ss_correction = [f"- ({old_ss_start})" for old_ss_start in producer_subset.min_element()]

        # Now perform the rerouting.
        new_producer_edge = gtx_transformations.utils.reroute_edge(
            is_producer_edge=True,
            current_edge=producer_edge,
            ss_offset=ss_correction,
            state=state,
            sdfg=sdfg,
            old_node=access_node,
            new_node=tmp_node,
        )
        gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
            is_producer_edge=True,
            new_edge=new_producer_edge,
            ss_offset=ss_correction,
            state=state,
            sdfg=sdfg,
            old_node=access_node,
            new_node=tmp_node,
        )
        state.remove_edge(producer_edge)

        # Now we reroute the consumer, also here we have to correct the subsets.
        #  The correction is the same as on the producer side.
        reconfigured_consumer: set[tuple[dace_nodes.Node, str]] = set()
        for consumer_edge in consumer_edges:
            consumer: dace_nodes.Node = consumer_edge.dst
            consumer_conn = consumer_edge.dst_conn

            new_consumer_edge = gtx_transformations.utils.reroute_edge(
                is_producer_edge=False,
                current_edge=consumer_edge,
                ss_offset=ss_correction,
                state=state,
                sdfg=sdfg,
                old_node=access_node,
                new_node=tmp_node,
            )
            state.remove_edge(consumer_edge)
            if (consumer, consumer_conn) not in reconfigured_consumer:
                gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                    is_producer_edge=False,
                    new_edge=new_consumer_edge,
                    sdfg=sdfg,
                    state=state,
                    ss_offset=ss_correction,
                    old_node=access_node,
                    new_node=tmp_node,
                )
                reconfigured_consumer.add((consumer, consumer_conn))

        gtx_transformations.gt_propagate_strides_from_access_node(
            sdfg=sdfg,
            state=state,
            outer_node=tmp_node,
        )
