# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Iterable, Literal, Optional, Union, overload

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

    It is tuned towards the following situations:
    ```python
    tmp = concat_where(condition, foo(...), bar(...))
    tmp2 = concat_where(condition, foo2(tmp, ...), bar2(tmp, ...))
    ```
    I.e. two consecutive `concat_where()` with the same condition.
    The transformation will essentially rewrite it to:
    ```python
    tmp2 = concat_where(condition, foo2(foo(...), ...), bar2(bar(...), ...))
    ```

    The transformation matches AccessNodes and checks if every read from the
    matched node can be satisfied by a single write to it. In that case the
    transformation will split the nodes and each producer will write into its
    own independent transients, the consumer nodes will be modified accordingly.

    Args:
        single_use_data: The list of data that is used only once.

    Todo:
        Currently for every consumer there can only have one producer. In case the
        consumer is a Map, this makes sense, but in case the consumer is an
        AccessNode one could split the consumer.
    """

    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)

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
        if not desc.transient:
            return False
        if isinstance(desc, dace_data.Scalar):
            return False
        if gtx_transformations.utils.is_view(desc, sdfg):
            return False

        if self._single_use_data is None:
            find_single_use_data = dace_analysis.FindSingleUseData()
            single_use_data = find_single_use_data.apply_pass(sdfg, None)
        else:
            single_use_data = self._single_use_data
        if access_node.data not in single_use_data[sdfg]:
            return False

        # There must be multiple producers, otherwise this transformation
        #  does not make sense.
        number_of_producers = graph.in_degree(access_node)
        if number_of_producers <= 1:
            return False

        # We also require, that every producer is distinct.
        #  We assume this to only simplify the implementation, because we do not
        #  need to keep track of which producer have been reconfigured. Furthermore,
        #  it does not make much sense to allow this.
        if number_of_producers != len({iedge.src for iedge in graph.in_edges(access_node)}):
            return False

        # Note we explicitly allow that there are less than one consumer. This _can_
        #  indicate that something is computed that is not used? However, because of
        #  MapFusion's ability to handle intermediates that have multiple producers
        #  this can happen. For that reason alone, we allow this. We really here on
        #  DaCe's DeadDataflow elimination to remove them. This is possible because
        #  we have made sure that the data is not used anywhere else.

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

        reconfigured_consumer: set[tuple[dace_nodes.Node, str]] = set()
        for producer_edge, consumer_edges in assignment.items():
            self._reroute_consumer(
                producer_edge=producer_edge,
                consumer_edges=consumer_edges,
                sdfg=sdfg,
                state=graph,
                reconfigured_consumer=reconfigured_consumer,
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
        Note that is possible that no consumer is assigned to a producer.
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
                #  output and then perform some reconnection. Similar to case of
                #  AccessNodes we require that if there is only one consumer that
                #  the consumer really consumes everything, this is done to ensure
                #  the tightness of the bounds, i.e. that everything that is consumed
                #  is used somewhere. If there are multiple consumer, then we assume
                #  that this is the case.
                # TODO(phimuell): Fuse that with the AccessNode check.
                if len(consumer_edges) == 1:
                    if not next(iter(consumer_edges)).data.src_subset.covers(
                        producer_edge.data.dst_subset
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
        reconfigured_consumer: set[tuple[dace_nodes.Node, str]],
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
            reconfigured_consumer: The set of consumer that has been reconfigured.
        """
        data_producer: dace_nodes.Node = producer_edge.src

        if isinstance(data_producer, dace_nodes.MapExit):
            self._reroute_consumer_map_producer(
                producer_edge=producer_edge,
                consumer_edges=consumer_edges,
                sdfg=sdfg,
                state=state,
                reconfigured_consumer=reconfigured_consumer,
            )

        elif isinstance(data_producer, dace_nodes.AccessNode):
            self._reroute_consumer_access_node_producer(
                producer_edge=producer_edge,
                consumer_edges=consumer_edges,
                sdfg=sdfg,
                state=state,
                reconfigured_consumer=reconfigured_consumer,
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
        reconfigured_consumer: set[tuple[dace_nodes.Node, str]],
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
            reconfigured_consumer: The set of consumer that has been reconfigured.
        """
        access_node: dace_nodes.AccessNode = producer_edge.dst
        data_producer: dace_nodes.AccessNode = producer_edge.src

        old_producer_read = producer_edge.data.src_subset.min_element()
        old_producer_write = producer_edge.data.dst_subset.min_element()

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
        reconfigured_consumer: set[tuple[dace_nodes.Node, str]],
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
            reconfigured_consumer: The set of consumer that has been reconfigured.

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


@dace_properties.make_properties
class SplitMemlet(dace_transformation.SingleStateTransformation):
    """Preparation stage for the `SplitAccessNode`.

    Essentially splits consumer edges, such that `SplitAccessNode` become applicable.
    The function matches the following situations: `(S) -> (i) -> (D)`.
    Where `i` is the node that would be split by the `SplitAccessNode` transformation.

    The transformation essentially targets the following situation:
    ```python
    tmp = concat_where(cond1, foo(...), a)
    tmp2 = concat_where(cond1 & cond2, foo2(tmp, ...), tmp)
    ```
    It essentially rewrites it into:
    tmp = concat_where(cond1 & cond2, foo(...), a)
    tmp2_1 = concat_where(cond1 & cond2, foo2(tmp, ...), tmp)
    tmp2 = concat_where(cond1 & !cond2, foo(...), tmp2_1)
    ```
    """

    source_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    intermediate_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    target_node = dace_transformation.PatternNode(dace_nodes.AccessNode)

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def expressions(cls) -> Any:
        return [
            dace.sdfg.utils.node_path_graph(cls.source_node, cls.intermediate_node, cls.target_node)
        ]

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        src_node: dace_nodes.AccessNode = self.source_node
        tmp_node: dace_nodes.AccessNode = self.intermediate_node
        tmp_desc: dace_data.Data = tmp_node.desc(sdfg)

        # If there is less than one incoming connection then it is useless to
        #  split the edges. Furthermore, `SplitAccessNode` must be able to get
        #  rid of `tmp_node`.
        if graph.in_degree(tmp_node) <= 1:
            return False
        if not tmp_desc.transient:
            return False
        if gtx_transformations.utils.is_view(tmp_desc, sdfg):
            return False

        # There can only be one connection between the source and the intermediate.
        #  This is to simplify implementation and also a restriction from the
        #  actual `SplitAccessNode`.
        src_tmp_edges = [oedge for oedge in graph.out_edges(src_node) if oedge.dst is tmp_node]
        if len(src_tmp_edges) != 1:
            return False
        src_tmp_edge: dace_graph.MultiConnectorEdge = src_tmp_edges[0]

        # We require that the producer of `tmp_node` are all distinct, these is a
        #  requirement from the splitter.
        if graph.in_degree(tmp_node) != len({iedge.src for iedge in graph.in_edges(tmp_node)}):
            return False

        tmp_subset: dace_sbs.Subset = src_tmp_edge.data.dst_subset
        if tmp_subset is None:
            return False

        # The splitting is only possible if the data, that comes from `src_node` can
        #  really be separated. For that we have to make sure that no map consumes
        #  what we write. However it is fully allowed that the Map consumes everything.
        found_edge_to_split = False
        for oedge in graph.out_edges(tmp_node):
            consumer = oedge.dst
            consumer_read = oedge.data.src_subset
            if consumer_read is None:
                return False
            elif isinstance(consumer, dace_nodes.AccessNode):
                # This transformation only makes sense if we can split some reads,
                #  thus there must be an intersection.
                if any((rs == 1) == False for _, _, rs in consumer_read):  # noqa: E712 [true-false-comparison]  # SymPy comparison
                    continue
                elif self._split_consumer_subset(
                    producer=tmp_subset,
                    consumer=consumer_read,
                    for_check=True,
                ):
                    # TODO: extend this to see that all edges could be split, also see note
                    #   At the end of this function.
                    found_edge_to_split = True
                continue
            elif isinstance(consumer, dace_nodes.MapEntry):
                if tmp_subset.intersects(consumer_read) and (not tmp_subset.covers(consumer_read)):
                    return False
                continue
            else:
                # TODO(phimuell): Implement these case.
                return False

        if not found_edge_to_split:
            return False

        # TODO(phimuell): These tests might not be enough, meaning this transformation
        #   might apply, but there are other things that prevent `SplitAccessNode`
        #   from applying. I guess the best thing would be to apply some collapsing
        #   pass that merges the edges together.
        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        src_node: dace_nodes.AccessNode = self.source_node
        tmp_node: dace_nodes.AccessNode = self.intermediate_node
        src_tmp_edge: dace_graph.MultiConnectorEdge = next(
            oedge for oedge in graph.out_edges(src_node) if oedge.dst is tmp_node
        )

        edges_to_split = self._find_edges_to_split(state=graph, src_tmp_edge=src_tmp_edge)
        self._split_consumer_edges(
            sdfg=sdfg,
            state=graph,
            src_tmp_edge=src_tmp_edge,
            edges_to_split=edges_to_split,
        )

    def _find_edges_to_split(
        self,
        state: dace.SDFGState,
        src_tmp_edge: dace_graph.MultiConnectorEdge,
    ) -> list[dace_graph.MultiConnectorEdge]:
        tmp_subset: dace_sbs.Subset = src_tmp_edge.data.dst_subset
        edges_to_split: list[dace_graph.OrderedMultiDiGraph] = []
        for oedge in state.out_edges(src_tmp_edge.dst):
            consumer = oedge.dst
            consumer_read = oedge.data.src_subset
            if isinstance(consumer, dace_nodes.AccessNode) and consumer_read is not None:
                if self._split_consumer_subset(
                    producer=tmp_subset,
                    consumer=consumer_read,
                    for_check=True,
                ):
                    edges_to_split.append(oedge)
        return edges_to_split

    def _split_consumer_edges(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        src_tmp_edge: dace_graph.MultiConnectorEdge,
        edges_to_split: list[dace_graph.MultiConnectorEdge],
    ) -> None:
        """Split all edges in `edges_to_split` into multiple edges.

        The edges will be split such that the source subset of the new edges
        are either fully convered by the destination subset of `src_tmp_edge`
        edge or have no intersection with it at all.
        The old edges will also be removed.

        Args:
            sdfg: The SDFG on which we operate.
            state: The state in which we operate.
            src_tmp_edges: The producing source edge.
            edges_to_split: The list of edges that should be split.
        """
        for edge_to_split in edges_to_split:
            self._split_consumer_edge(
                sdfg=sdfg,
                state=state,
                src_tmp_edge=src_tmp_edge,
                edge_to_split=edge_to_split,
            )
            state.remove_edge(edge_to_split)

    def _split_consumer_edge(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        src_tmp_edge: dace_graph.MultiConnectorEdge,
        edge_to_split: dace_graph.MultiConnectorEdge,
    ) -> None:
        """Split a single edge, see `_split_consumer_edges()`  for more."""
        producer_subset = src_tmp_edge.data.dst_subset
        consumer_subset = edge_to_split.data.src_subset
        assert isinstance(edge_to_split.dst, dace_nodes.AccessNode)

        # If the subset is not given assume that it starts at zero.
        consumer_destination_subset = edge_to_split.data.dst_subset
        if consumer_destination_subset is None:
            consumer_destination_subset = dace_sbs.Range.from_array(edge_to_split.dst.desc(sdfg))

        # Perform the actual splitting.
        new_consumer_subsets = self._split_consumer_subset(
            producer=producer_subset,
            consumer=consumer_subset,
            for_check=False,
        )

        old_consumer_start = consumer_subset.min_element()
        consumer_dest_start = consumer_destination_subset.min_element()

        new_edges = []
        for new_consumer_subset in new_consumer_subsets:
            new_subset_size = new_consumer_subset.size()
            new_consumer_start = new_consumer_subset.min_element()

            # The subset at the source was computed by `_split_consumer_subset()`,
            #  but we also need the subset at the destination, i.e. where do we
            #  write to. For this we assume that we always write into some hypercube.
            #  We then compute the offset of the now source subset, compared to the
            #  original origin of the source subset and apply the same shift also
            #  to the original destination subset.
            new_consumer_dest_start = [
                dace_sym.pystr_to_symbolic(f"({dstart}) + (({ncstart}) - ({ocstart}))")
                for dstart, ocstart, ncstart in zip(
                    consumer_dest_start, old_consumer_start, new_consumer_start
                )
            ]
            new_consumer_dest_end = [
                dace_sym.pystr_to_symbolic(f"({ncdstart}) + ({ss} - 1)")
                for ncdstart, ss in zip(new_consumer_dest_start, new_subset_size)
            ]
            new_consumer_dest_subset = dace_sbs.Range(
                [
                    (start, end, 1)
                    for start, end in zip(new_consumer_dest_start, new_consumer_dest_end)
                ]
            )

            # Create the new edge, and copy the Memlet, afterwards set the subsets
            #  accordingly.
            # NOTE: The volume is not updated, but we do not care about that.
            # NOTE: Because the consumer are only AccessNodes, and the data has not
            #   changed, there is no need to propagate or update anything.
            new_edges.append(
                state.add_edge(
                    edge_to_split.src,
                    edge_to_split.src_conn,
                    edge_to_split.dst,
                    edge_to_split.dst_conn,
                    dace.Memlet.from_memlet(edge_to_split.data),
                )
            )
            new_edges[-1].data.src_subset = new_consumer_subset
            new_edges[-1].data.dst_subset = new_consumer_dest_subset

    @overload
    def _split_consumer_subset(
        self,
        producer: dace_sbs.Range,
        consumer: dace_sbs.Range,
        for_check: Literal[True],
    ) -> bool: ...

    @overload
    def _split_consumer_subset(
        self,
        producer: dace_sbs.Range,
        consumer: dace_sbs.Range,
        for_check: Literal[False],
    ) -> list[dace_sbs.Range]: ...

    def _split_consumer_subset(
        self,
        producer: dace_sbs.Range,
        consumer: dace_sbs.Range,
        for_check: bool,
    ) -> Union[list[dace_sbs.Range], bool]:
        """Splits the `consumer` subset.

        The resulting subsets are either fully covered by `producer` or have no
        intersection with it. If `for_check` is `True` the function will return
        a boolean to indicate if the subset can be split or not (for any reason).
        It is an error to call the function with `for_check` set to `False`
        but the subset can not be split.

        Args:
            producer: The subset describing the producer.
            consumer: The subset describing what the consumer reads.
            for_check: Only check if the subset can be split.

        Todo:
            The current implementation is only able to handle the case where
            the consumer subset must only be split in one dimension. This
            restriction must be solved.
        """
        assert producer.dims() == consumer.dims()

        # Currently we require that we have to split only along one dimension.
        dimension_in_which_to_split: Optional[int] = None
        splitted_subsets_in_dim: list[tuple[Any, ...]] = []
        for dim in range(producer.dims()):
            prod_low = producer[dim][0]
            prod_high = producer[dim][1]
            consu_low = consumer[dim][0]
            consu_high = consumer[dim][1]

            # In this dimension the consumer consumes everything the producer
            #  generates. Therefore no splitting is needed.
            embedded_cond1 = (prod_low <= consu_low) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
            embedded_cond2 = (consu_high <= prod_high) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
            if embedded_cond1 and embedded_cond2:
                continue

            # Check if there is an intersection at all.
            #  I am pretty sure that there is no strange `-1` correction needed.
            intersec_cond1 = consu_low <= prod_high
            intersec_cond2 = prod_low <= consu_high
            if intersec_cond1 == False or intersec_cond2 == False:  # noqa: E712 [true-false-comparison]  # SymPy comparison
                assert for_check
                return False
            if not (intersec_cond1 == True and intersec_cond2 == True):  # noqa: E712 [true-false-comparison]  # SymPy comparison
                assert for_check
                return False

            # The consumer is not fully embedded in the producer, so this dimension
            #  we must split. If we found before ignore it.
            # TODO(phimuell): By ignoring this case here, i.e. "pretend that no split
            #   was needed", we could handle that and then recursively handle the
            #   rest.
            if dimension_in_which_to_split is not None:
                assert for_check
                return False
            dimension_in_which_to_split = dim

            # Determine the splitting case that we have.
            #  I am pretty sure about the `<` here.
            read_right = (prod_high < consu_high) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
            read_left = (consu_low < prod_low) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
            assert read_right or read_left

            # If we only want to check then we do not need the exact splitting.
            if for_check:
                continue

            # Now we determine the split mode. There are three cases.
            if read_right and read_left:
                # The consumer starts reading before the producer starts to write
                #  and also reads more than the producer writes to, so it is split
                #  into three parts.
                splitted_subsets_in_dim = [
                    (consu_low, prod_low - 1, 1),
                    (prod_low, prod_high, 1),
                    (prod_high + 1, consu_high, 1),
                ]
            elif read_left:
                # The consumer starts reading before the producer starts writing.
                #  Thus there are two parts.
                splitted_subsets_in_dim = [(consu_low, prod_low - 1, 1), (prod_low, consu_high, 1)]
            elif read_right:
                # The consumer starts reading inside the range the producer writes to
                #  but reads more, so again two splits.
                splitted_subsets_in_dim = [
                    (consu_low, prod_high, 1),
                    (prod_high + 1, consu_high, 1),
                ]

        # In check mode we are done.
        if for_check:
            return dimension_in_which_to_split is not None

        assert dimension_in_which_to_split is not None
        assert len(splitted_subsets_in_dim) > 0
        assert all(((e - s) >= 0) == True for s, e, _ in splitted_subsets_in_dim)  # noqa: E712 [true-false-comparison]  # SymPy comparison

        splitted_subsets: list[dace_sbs.Range] = []
        for splitted_subset_in_dim in splitted_subsets_in_dim:
            splitted_subsets.append(
                dace_sbs.Range(
                    [
                        (
                            splitted_subset_in_dim
                            if dim == dimension_in_which_to_split
                            else org_consumer_sbs
                        )
                        for dim, org_consumer_sbs in enumerate(consumer)
                    ]
                )
            )

        return splitted_subsets
