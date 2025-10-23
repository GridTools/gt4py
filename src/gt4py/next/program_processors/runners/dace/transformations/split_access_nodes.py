# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import Any, Iterable, Optional

import dace
from dace import properties as dace_properties, transformation as dace_transformation
from dace.sdfg import graph as dace_graph, nodes as dace_nodes
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.transformations import (
    splitting_tools as gtx_dace_split,
)


def gt_split_access_nodes(
    sdfg: dace.SDFG,
    validate: bool = True,
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
    i.e. if every read to the AccessNode can be satisfied by a single
    write to the AccessNode and the AccessNode is only used at one
    location, then the node is split.
    This means that the reads will be satisfied directly and the node
    does not have to materialize.

    Before this transformation is run the `SplitConsumerMemlet` should
    be run.

    Args:
        single_use_data: The list of data that is used only once.
        assume_single_use_data: Assume that `access_node` is single use data.
            Note this flag should _only_ be used if the transformation is used
            through the `apply_to()` interface and the caller has ensured that
            `self.access_node` really is single use data.

    Todo:
        - Made it possible to merge splits, i.e. such that a fragment can
            be described by two producers.
        - Create a version that is able to split over multiple states. This is
            mostly useful to enable more state fusion.
    """

    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)

    assume_single_use_data = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="Always assume that `self.access_node` is single use data. Only useful if used through `SplitAccessNode.apply_to()`.",
    )

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and be passed at construction time. Needed until
    #  [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: Optional[dict[dace.SDFG, set[str]]]

    def __init__(
        self,
        *args: Any,
        single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
        assume_single_use_data: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._single_use_data = single_use_data
        if assume_single_use_data is not None:
            self.assume_single_use_data = assume_single_use_data

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

        # To get rid of the intermediate it must be a single use transient.
        #  We postpone the single use check.
        if not desc.transient:
            return False
        if gtx_transformations.utils.is_view(desc, sdfg):
            return False

        # There must be multiple producers, otherwise this transformation
        #  does not make sense.
        number_of_producers = graph.in_degree(access_node)
        if number_of_producers <= 1:
            return False

        # Since this transformation can not handle splitting over multiple state
        #  it must be consumed directly, although single use data implies this
        #  we do it here explicitly to avoid a scan.
        #  It is also important that we explicitly allow one consumer. This case
        #  might imply that we have dead data.
        number_of_consumers = graph.out_degree(access_node)
        if number_of_consumers == 0:
            return False

        # Now check if a decomposition exist.
        edge_reassignments = self._find_edge_reassignment(graph)
        if edge_reassignments is None:
            return False
        if not self._check_split_constraints(
            state=graph,
            sdfg=sdfg,
            edge_reassignments=edge_reassignments,
        ):
            return False

        if self.assume_single_use_data:
            single_use_data = {sdfg: {access_node.data}}
        if self._single_use_data is None:
            find_single_use_data = dace_analysis.FindSingleUseData()
            single_use_data = find_single_use_data.apply_pass(sdfg, None)
        else:
            single_use_data = self._single_use_data
        if access_node.data not in single_use_data[sdfg]:
            return False

        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        access_node: dace_nodes.AccessNode = self.access_node

        edge_reassignments = self._find_edge_reassignment(graph)
        assert edge_reassignments is not None

        # TODO(phimuell): Make it more general that it can take the full advantage
        #   of the splitter functionality.
        split_description = [e.data.dst_subset for e in edge_reassignments.keys()]

        fragment_access_nodes = gtx_dace_split.split_node(
            state=graph,
            sdfg=sdfg,
            node_to_split=access_node,
            split_description=split_description,
            allow_to_bypass_nodes=True,
        )

        # Split node will remove the AccessNode but does not remove the data.
        sdfg.remove_data(access_node.data, validate=False)

        # We have to clean up the isolated fragments. This is because we specified
        #  `allow_to_bypass_nodes` in the call above.
        for ac in fragment_access_nodes.values():
            if graph.degree(ac) == 0:
                graph.remove_node(ac)
                sdfg.remove_data(ac.data, validate=False)

        # NOTE: In some situation it happens that when a producer writes
        #   something inside `access_node` and the data is never read. This is
        #   not an error, but can be a side effect of MapFusion or similar
        #   transformations. This will lead to dead data flow, that we will
        #   not remove. Instead DDE should be run.

    def _find_edge_reassignment(
        self,
        state: dace.SDFGState,
    ) -> dict[dace_graph.MultiConnectorEdge, set[dace_graph.MultiConnectorEdge]] | None:
        """Determine how the edges should be distributed to the fragments.

        The current implementation defines the fragments, i.e. the pieces into
        which `self.access_node` should be split into, through the incoming edges.
        This means that every incoming edge defines one fragment.
        Therefor, the function returns a `dict` that maps each incoming edge to the
        set of out going edges that are associated to it, i.e. depend on the producer
        edge or `None` if such a distribution does not exist.

        The function does not perform any checks if the split lead to a valid
        SDFG, for that reason the result should be checked by
        `_check_split_constraints()`.

        Todo:
            Extend the function such that a fragment is not limited to a single
            incoming edge, but can also be defined through multiple edges.
        """
        access_node: dace_nodes.AccessNode = self.access_node

        # NOTE: Build the assignments based on the producers. Basing the split on
        #  the producers has the advantages that it naturally takes "double use"
        #  into account, i.e. one producer computes something and two different
        #  reads it. However, it is also quite unnatural, because the consumer
        #  should define the split (i.e. more than one producer are needed to
        #  generate the data for a consumer). This is hard to handle, but should
        #  be implemented at some point.
        edge_reassignments: dict[
            dace_graph.MultiConnectorEdge, set[dace_graph.MultiConnectorEdge]
        ] = {}
        for iedge in state.in_edges(access_node):
            if iedge.data.dst_subset is None:
                return None  # TODO(phimuell): Lift this.
            if iedge.data.wcr is not None:
                return None
            edge_reassignments[iedge] = set()

        # Now match the outgoing edges to their incoming producers.
        for oedge in state.out_edges(access_node):
            if oedge.data.wcr is not None:
                return None
            possible_producer = self._find_producer(oedge, edge_reassignments.keys())
            if possible_producer is None:
                return None
            edge_reassignments[possible_producer].add(oedge)

        unused_producers = [
            producer
            for producer, assigned_consumers in edge_reassignments.items()
            if len(assigned_consumers) == 0
        ]
        if unused_producers:
            # This situation is generated by MapFusion, if the intermediate
            #  AccessNode has to be kept alive.
            warnings.warn(
                "'SplitAccessNode': found producers "
                + ", ".join((str(p) for p in unused_producers))
                + " that generates data but that is never read.",
                stacklevel=0,
            )

        return edge_reassignments

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
            # This might indicate an error (the same memory location is written by
            #  multiple producer. However, there are some cases where it is not an
            #  error. For example a Map, with two Tasklets, writes to the node,
            #  one Tasklet writes `T[__i, 0]` the other `T[__i, 10]`, where `__i`
            #  is the iteration index. Then Memlet propagation will set the subset
            #  to something like `T[:, 0:10]`. So it is not an error in that case.
            warnings.warn(
                f"Found transient '{self.access_node.data}' that has multiple overlapping"
                " incoming edges. Might indicate an error.",
                stacklevel=0,
            )
            return None

        return possible_producers[0]

    def _check_split_constraints(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
        edge_reassignments: dict[dace_graph.MultiConnectorEdge, set[dace_graph.MultiConnectorEdge]],
    ) -> bool:
        """Checks if the decomposition results in a valid SDFG.

        This function is used to validate the decomposition computed by
        `self._find_edge_reassignment()`.
        """

        for producer_edge, consumer_edges in edge_reassignments.items():
            data_source = producer_edge.src

            if len(consumer_edges) == 0:
                continue
            elif isinstance(data_source, dace_nodes.AccessNode):
                # TODO(phimuell): Should we also ensure that the domains are tight?
                if gtx_transformations.utils.is_view(data_source, sdfg):
                    return False

                # If the source is a global data, then we do not impose any other
                #  constraints.
                if not data_source.desc(sdfg).transient:
                    continue

                # If the source is a transient then we distinguish between two cases.
                #  In the first case there is only one consumer, in that case we
                #  require that everything is read. In the second case, more than
                #  one consumer, we do not impose any constraints.
                #  We do this to ensure the tightness of the temporaries, i.e. what
                #  is computed is also read, which is core assumption of the
                #  `CopyChainRemover`.
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
