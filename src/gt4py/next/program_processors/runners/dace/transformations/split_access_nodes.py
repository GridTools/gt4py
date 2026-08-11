# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Optional

import dace
from dace import (
    properties as dace_properties,
    subsets as dace_sbs,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes
from dace.transformation.passes import analysis as dace_analysis
from ordered_set import OrderedSet

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.transformations import (
    splitting_tools as gtx_dace_split,
)


@dataclasses.dataclass(frozen=True)
class _Fragment:
    """One piece into which an AccessNode is split.

    Attributes:
        producers: The incoming edges that generate the data of the fragment.
        consumers: The outgoing edges that read data of the fragment.
        subset: The region of the original data that the fragment describes,
            i.e. the union of what `producers` write.
    """

    producers: OrderedSet[dace_graph.MultiConnectorEdge]
    consumers: OrderedSet[dace_graph.MultiConnectorEdge]
    subset: dace_sbs.Subset


def gt_split_access_nodes(
    sdfg: dace.SDFG,
    validate: bool = True,
    validate_all: bool = False,
    single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
) -> Optional[int]:
    """Applies the `SplitAccessNode` transformation to the SDFG.

    This function should be the preferred way to run the `SplitAccessNode`
    transformation. Since it will ensure that the single data is only computed
    once. Furthermore, it guarantees that the transformations are applied in
    a deterministic order.

    The transformation returns the number of AccessNodes that have been split.

    Args:
        sdfg: The SDFG to process.
        validate: Perform validation after the pass has run.
        validate_all: Perform extensive validation.
        single_use_data: Which data descriptors are used only once.
            If not passed the function will run `FindSingleUseData`, if passed the
            function will update its content and add the newly generated data.
    """

    # To ensures that the `{src,dst}_subset` are properly set, run initialization.
    #  See [issue 1703](https://github.com/spcl/dace/issues/1703)
    for state in sdfg.states():
        for edge in state.edges():
            edge.data.try_initialize(sdfg, state, edge)

    if single_use_data is None:
        find_single_use_data = dace_analysis.FindSingleUseData()
        single_use_data = find_single_use_data.apply_pass(sdfg, None)

    apply_count = 0
    for nsdfg in sdfg.all_sdfgs_recursive():
        apply_count += _apply_split_access_node_non_recursive(
            sdfg=nsdfg,
            validate=validate,
            validate_all=validate_all,
            single_use_data=single_use_data[nsdfg],
        )

    return apply_count


def _apply_split_access_node_non_recursive(
    sdfg: dace.SDFG,
    validate: bool,
    validate_all: bool,
    single_use_data: set[str],
) -> int:
    apply_count = 0
    if len(single_use_data) == 0:
        return apply_count

    # The splitter transformation. Note that we set `assume_single_use_data` to `True`
    #  because we do this test outside.
    access_node_splitter = gtx_transformations.SplitAccessNode(
        assume_single_use_data=True,
    )

    # The transformation only applies to transient single use data, thus the order
    #  in which the states are provided are irrelevant. Furthermore, all fragments
    #  that are generated does not need to be examined again, thus a one pass is
    #  enough. This is because the `SplitAccessNode` transformation does not generates
    #  new edges upon the split, it just reroutes them to the fragments.
    for state in sdfg.states():
        state_cfg_id = state.parent_graph.cfg_id
        state_id = state.block_id
        scope_dict = state.scope_dict()

        # We can only split single use data that is also transient. Because GT4Py uses
        #  an SSA style we know that there is only one AccessNode that refers to that
        #  data. Thus, all AccessNodes that are stored refer to different data
        access_nodes_to_process = sorted(
            (
                dnode
                for dnode in state.data_nodes()
                if dnode.data in single_use_data and scope_dict[dnode] is None
            ),
            key=lambda dnode: dnode.data,
        )
        assert len(access_nodes_to_process) == len(
            set(map(lambda ac: ac.data, access_nodes_to_process))
        )

        if len(access_nodes_to_process) == 0:
            # Nothing to process in this state, continue.
            continue

        # Now try to split all candidates that we have found.
        for access_node_to_process in access_nodes_to_process:
            access_node_splitter.setup_match(
                sdfg=sdfg,
                cfg_id=state_cfg_id,
                state_id=state_id,
                subgraph={gtx_transformations.SplitAccessNode.access_node: access_node_to_process},
                expr_index=0,
                override=True,
            )
            if access_node_splitter.can_be_applied(
                graph=state, expr_index=0, sdfg=sdfg, permissive=False
            ):
                splitted_access_nodes = access_node_splitter.apply(graph=state, sdfg=sdfg)
                if validate_all:
                    # Not super correct as we not check at the top of the hierarchy.
                    sdfg.validate()

                # We have to update `single_use_data`. By definition all data that we
                #  generate through splitting is also single use data.
                single_use_data.update(sac.data for sac in splitted_access_nodes.values())
                apply_count += 1

    if validate:
        sdfg.validate()

    return apply_count


@dace_properties.make_properties
class SplitAccessNode(dace_transformation.SingleStateTransformation):
    """The transformation will split an AccessNode into multiple ones.

    The node is split into fragments, such that every read is served by a
    single fragment and the AccessNode is only used at one location. A
    fragment is usually defined by a single write, but if a read spans the
    data of several writes, then those writes form one fragment together.
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
        - Create a version that is able to split over multiple states. This is
            mostly useful to enable more state fusion.

    Note:
        The actual split operation is performed using `splitting_tools.split_node()`.
        Furthermore, as a special extension, to support certain workflows the
        `apply()` function returns the return value of that function.
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
        fragments = self._find_edge_reassignment(graph)
        if fragments is None:
            return False
        if len(fragments) <= 1:
            return False
        if not self._check_split_constraints(
            state=graph,
            sdfg=sdfg,
            fragments=fragments,
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
    ) -> dict[dace_sbs.Subset, dace_nodes.AccessNode]:
        access_node: dace_nodes.AccessNode = self.access_node

        fragments = self._find_edge_reassignment(graph)
        assert fragments is not None

        # TODO(phimuell): Make it more general that it can take the full advantage
        #   of the splitter functionality.
        split_description = [fragment.subset for fragment in fragments]

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
        for split_sbs in list(fragment_access_nodes.keys()):
            ac = fragment_access_nodes[split_sbs]
            if graph.degree(ac) == 0:
                graph.remove_node(ac)
                sdfg.remove_data(ac.data, validate=False)
                fragment_access_nodes.pop(split_sbs)

        # NOTE: In some situation it happens that when a producer writes
        #   something inside `access_node` and the data is never read. This is
        #   not an error, but can be a side effect of MapFusion or similar
        #   transformations. This will lead to dead data flow, that we will
        #   not remove. Instead DDE should be run.

        # Special extension to support certain workflows.
        return fragment_access_nodes

    def _find_edge_reassignment(
        self,
        state: dace.SDFGState,
    ) -> list[_Fragment] | None:
        """Determine how the edges should be distributed to the fragments.

        A fragment, i.e. one of the pieces into which `self.access_node` is split,
        is defined by a set of incoming edges together with the outgoing edges that
        read what those incoming edges produce. In the simplest case a fragment is
        made up of a single producer, but if a consumer reads data that is generated
        by several producers, then all of them end up in the same fragment.
        The function returns the list of fragments or `None` if no such distribution
        exists.

        The function does not perform any checks if the split lead to a valid
        SDFG, for that reason the result should be checked by
        `_check_split_constraints()`.
        """
        access_node: dace_nodes.AccessNode = self.access_node

        producer_edges: list[dace_graph.MultiConnectorEdge] = []
        for iedge in state.in_edges(access_node):
            if iedge.data.dst_subset is None:
                return None  # TODO(phimuell): Lift this.
            if iedge.data.wcr is not None:
                return None
            producer_edges.append(iedge)

        # We use union find over the producers, i.e. `fragment_of_producer[i]` is
        #  the id of another producer that is in the same fragment as producer `i`,
        #  or `i` itself if `i` is the representative of its fragment. Every producer
        #  starts alone and a consumer that straddles several producers merges them.
        fragment_of_producer = list(range(len(producer_edges)))

        def find_fragment(idx: int) -> int:
            while fragment_of_producer[idx] != idx:
                fragment_of_producer[idx] = fragment_of_producer[fragment_of_producer[idx]]
                idx = fragment_of_producer[idx]
            return idx

        consumers_of_fragment: dict[int, OrderedSet[dace_graph.MultiConnectorEdge]] = {
            i: OrderedSet() for i in range(len(producer_edges))
        }
        for oedge in state.out_edges(access_node):
            if oedge.data.wcr is not None:
                return None
            # NOTE: This must be the raw property and not `get_src_subset()`. A
            #  Memlet that has no side associated to it has no `src_subset`, see
            #  [issue 1703](https://github.com/spcl/dace/issues/1703), and the check
            #  below turns that into a decline. `get_src_subset()` initializes the
            #  Memlet, which removes that guard and makes the transformation accept
            #  nodes whose rerouting then fails inside
            #  `reconfigure_dataflow_after_rerouting()`.
            consumer_subset = oedge.data.src_subset
            if consumer_subset is None:
                return None  # TODO(phimuell): Lift this.

            # All producers that might contribute to what this consumer reads have
            #  to be part of the same fragment. Several producers may already share a
            #  fragment, so the ids are deduplicated before they are merged.
            contributing = {
                find_fragment(i)
                for i, producer_edge in enumerate(producer_edges)
                if gtx_dace_split.maybe_intersecting(producer_edge.data.dst_subset, consumer_subset)
            }
            if len(contributing) == 0:
                return None

            merged_id = min(contributing)
            for fragment_id in sorted(contributing):
                if fragment_id != merged_id:
                    fragment_of_producer[fragment_id] = merged_id
                    consumers_of_fragment[merged_id] |= consumers_of_fragment.pop(fragment_id)
            consumers_of_fragment[merged_id].add(oedge)

        fragment_of = [find_fragment(i) for i in range(len(producer_edges))]
        fragments: list[_Fragment] = []
        for fragment_id, consumer_edges in consumers_of_fragment.items():
            producers = OrderedSet(
                producer_edge
                for i, producer_edge in enumerate(producer_edges)
                if fragment_of[i] == fragment_id
            )
            # NOTE: We must use the adjacency only merger here, producers that
            #  overlap would describe the same memory twice, so we must not merge
            #  them into one region.
            subsets = gtx_dace_split.subset_merger(
                [producer_edge.data.dst_subset for producer_edge in producers]
            )
            # The producers of a fragment must describe a single contiguous region,
            #  otherwise it can not be turned into one data descriptor.
            if len(subsets) != 1:
                return None
            fragments.append(
                _Fragment(producers=producers, consumers=consumer_edges, subset=subsets[0])
            )

        # `split_node()` requires that every consumer edge is fully covered by
        #  exactly one fragment, so a consumer must not straddle a fragment boundary.
        for fragment in fragments:
            for consumer_edge in fragment.consumers:
                if not fragment.subset.covers(consumer_edge.data.src_subset):
                    return None
        for i, fragment in enumerate(fragments):
            if any(
                gtx_dace_split.maybe_intersecting(fragment.subset, other.subset)
                for other in fragments[i + 1 :]
            ):
                return None

        unused_producers = [
            producer
            for fragment in fragments
            if len(fragment.consumers) == 0
            for producer in fragment.producers
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

        return fragments

    def _check_split_constraints(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
        fragments: list[_Fragment],
    ) -> bool:
        """Checks if the decomposition results in a valid SDFG.

        This function is used to validate the decomposition computed by
        `self._find_edge_reassignment()`.
        """

        for fragment in fragments:
            consumer_edges = fragment.consumers
            if len(consumer_edges) == 0:
                continue

            for producer_edge in fragment.producers:
                data_source = producer_edge.src
                if isinstance(data_source, dace_nodes.AccessNode):
                    # TODO(phimuell): Should we also ensure that the domains are tight?
                    if gtx_transformations.utils.is_view(data_source, sdfg):
                        return False
                elif not isinstance(data_source, dace_nodes.MapExit):
                    return False

            # For a fragment with a single producer we require that what is computed
            #  is also read. We can not impose that on a fragment that is fed by
            #  several producers, no consumer is associated to an individual producer
            #  of it, so requiring that the fragment is read in full would reject the
            #  whole node, including the fragments that are fine. What stays unread is
            #  dead data, which we already tolerate for a fragment without any
            #  consumer, see the warning in `_find_edge_reassignment()`.
            if len(fragment.producers) > 1:
                continue

            (producer_edge,) = fragment.producers
            data_source = producer_edge.src

            if isinstance(data_source, dace_nodes.AccessNode):
                # If the source is a global data, then we do not impose any other
                #  constraints.
                if not data_source.desc(sdfg).transient:
                    continue

                # If the source is a transient then we distinguish between two cases.
                #  In the first case there is only one consumer, in that case we
                #  require that everything is read. In the second case, more than
                #  one consumer, we do not impose any constraints.
                # TODO(phimuell): Lift this limitation.
                if len(consumer_edges) == 1:
                    if not next(iter(consumer_edges)).data.src_subset.covers(
                        producer_edge.data.dst_subset
                    ):
                        return False

            else:
                # The source is a Map, in this case we just generate a new transient
                #  output and then perform some reconnection. However, we require that
                #  all consumer read exactly what is is written by the map. This
                #  is to ensure some tightness of the domains.
                if not all(
                    consumer_edge.data.src_subset.covers(producer_edge.data.dst_subset)
                    for consumer_edge in consumer_edges
                ):
                    return False
        return True
