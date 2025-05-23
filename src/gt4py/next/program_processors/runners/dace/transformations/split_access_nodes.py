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

        # TODO(phimuell): Make it more general that it can take the full advantage
        #  of the splitter functionality.
        split_description = [e.data.dst_subset for e in assignment.keys()]

        gtx_transformations.spliting_tools.split_node(
                state=graph,
                sdfg=sdfg,
                node_to_split=access_node,
                split_description=split_description,
                allow_to_bypass_nodes=True,
        )
        # Remove the old intermediate that is no longer needed.
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
