# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum
from typing import Any, Optional, Sequence

import dace
from dace import (
    properties as dace_properties,
    subsets as dace_sbs,
    symbolic as dace_sym,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


def gt_remove_copy_chain(
    sdfg: dace.SDFG,
    validate: bool = True,
    validate_all: bool = False,
    single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
) -> Optional[int]:
    """Applies the `CopyChainRemover` transformation to the SDFG.

    The transformation returns the number of removed data containers or `None`
    if nothing was done.

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

    result: int = sdfg.apply_transformations_repeated(
        CopyChainRemover(single_use_data=single_use_data),
        validate=validate,
        validate_all=validate_all,
    )
    return result if result != 0 else None


class CopyChainRemoverMode(Enum):
    """Switch for `CopyChainRemover` mode.

    The `NONE` mode is invalid, it is there for signaling that the transformation
    cannot be applied.
    The `PULL` mode corresponds to the case where `a1` is transient, single-use
    data and its full shape is copied to `a2`, thus `a1` can be replaced with `a2`.
    The `PUSH` mode corresponds to the case where `a2` is transient, single-use
    data and its full shape is copied from `a1`, thus `a2` can be replaced with `a1`.
    """

    NONE = 0
    PULL = 1
    PUSH = 2


@dace_properties.make_properties
class CopyChainRemover(dace_transformation.SingleStateTransformation):
    """Removes chain of redundant copies, mostly related to `concat_where`.

    `concat_where`, especially when nested, will build "chains" of AccessNodes,
    this transformation will remove them. It should be called repeatedly until a
    fix point is reached and should be seen as an addition to the array removal passes
    that ship with DaCe.
    The transformation will look for the pattern `(A1) -> (A2)`, i.e. a data container
    is copied into another one, at the global scope. The transformation will then
    remove `A1` and rewire the edges such that they now refer to `A2`. Another, and
    probably better way, is to consider the transformation as fusion transformation
    for AccessNodes.

    The transformation builds on ADR-18 and imposes the following additional
    requirements before it can be applied:
    - Through the merging of `A1` and `A2` no cycles are created.
    - `A1` can not be used anywhere else.
    - `A1` is a transient and must have the same dimensionality as `A2`.
    - `A1` is fully read by `A2`.

    This transformation corresponds to the `PULL` mode of `CopyChainRemoverMode`.
    In a second iteraton, the transformation was extended to support the mirror
    case (`PUSH`), where A2 is removed if the following requirements are satisfied:
    - Through the merging of `A1` and `A2` no cycles are created.
    - `A2` can not be used anywhere else.
    - `A2` is a transient and must have the same dimensionality as `A1`.
    - `A2` is fully written by `A1`.

    Notes:
        The transformation assumes that the domain inference adjusted the ranges of
        the maps such that, in case they write into a transient, the full shape of
        the transient array is written, has the same size, i.e. there is not padding,
        or data that is not written to.

    Args:
        single_use_data: List of data containers that are used only at one place.
            Will be stored internally and not updated.

    Todo:
        - Extend such that not the full array must be read.
        - Try to allow more than one connection between `A1` and `A2`.
    """

    node_a1 = dace_transformation.PatternNode(dace_nodes.AccessNode)
    node_a2 = dace_transformation.PatternNode(dace_nodes.AccessNode)

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
        return [
            dace.sdfg.utils.node_path_graph(
                cls.node_a1,
                cls.node_a2,
            )
        ]

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        a1: dace_nodes.AccessNode = self.node_a1
        a2: dace_nodes.AccessNode = self.node_a2

        # We only allow that we operate on the top level scope.
        if graph.scope_dict()[a1] is not None:
            return False

        a1_desc = a1.desc(sdfg)
        a2_desc = a2.desc(sdfg)

        # This avoids that we have to modify the subsets in a fancy way.
        # TODO(phimuell): Lift this limitation.
        if len(a1_desc.shape) != len(a2_desc.shape):
            return False

        # For simplicity we assume that neither of `a1` nor `a2` are views.
        # TODO(phimuell): Implement some of the cases.
        if gtx_transformations.utils.is_view(a1_desc, None):
            return False
        if gtx_transformations.utils.is_view(a2_desc, None):
            return False

        # TODO(phimuell): Relax this to only prevent host-device copies.
        if a1_desc.storage != a2_desc.storage:
            return False

        # There shall only be one edge connecting `a1` and `a2`.
        #  We even strengthen this requirement by not checking for the node `a2`,
        #  but for the data.
        connecting_edges = [
            oedge
            for oedge in graph.out_edges(a1)
            if isinstance(oedge.dst, dace_nodes.AccessNode) and (oedge.dst.data == a2.data)
        ]
        if len(connecting_edges) != 1:
            return False

        # The full array `a1` is copied into `a2`. Note that it is allowed, that
        #  `a2` is bigger than `a1`, it is just important that everything that was
        #  written into `a1` is also accessed.
        connecting_edge = connecting_edges[0]
        assert connecting_edge.dst is a2
        connecting_memlet = connecting_edge.data

        # If the destination or the source subset of the connection is not fully
        #  specified, we do not apply.
        src_subset = connecting_memlet.get_src_subset(connecting_edge, graph)
        if src_subset is None:
            return False
        dst_subset = connecting_memlet.get_dst_subset(connecting_edge, graph)
        if dst_subset is None:
            return False

        if self._get_copy_chain_mode(sdfg, graph) == CopyChainRemoverMode.NONE:
            return False

        # We have to ensure that no cycle is created through the removal of one node.
        #  For this we have to ensure that there is no connection, beside the direct
        #  one between `a1` and `a2`.
        # NOTE: We only check the outgoing edges of `a1`, it is not needed to also
        #   check the incoming edges, because this will not create a cycle.
        if gtx_transformations.utils.is_reachable(
            start=[oedge.dst for oedge in graph.out_edges(a1) if oedge.dst is not a2],
            target=a2,
            state=graph,
        ):
            return False

        # NOTE: In case the node we keep is a non-transient we do not have to
        #   check if it is read or written to somewhere else in this state.
        #   The reason is that ADR18 guarantees that every access is point-wise,
        #   therefore the temporary we remove is never used as double buffer.
        return True

    def is_single_use_data(
        self,
        sdfg: dace.SDFG,
        data: str | dace_nodes.AccessNode,
    ) -> bool:
        """Checks if `data` is a single use data."""
        assert sdfg in self._single_use_data
        if isinstance(data, dace_nodes.AccessNode):
            data = data.data
        return data in self._single_use_data[sdfg]

    def apply(
        self,
        graph: dace.SDFGState | dace.SDFG,
        sdfg: dace.SDFG,
    ) -> None:
        a1: dace_nodes.AccessNode = self.node_a1
        a2: dace_nodes.AccessNode = self.node_a2
        a1_to_a2_edge: dace_graph.MultiConnectorEdge = next(
            oedge for oedge in graph.out_edges(a1) if oedge.dst is a2
        )
        a1_to_a2_memlet: dace.Memlet = a1_to_a2_edge.data
        a1_to_a2_src_subset: dace_sbs.Range = a1_to_a2_memlet.get_src_subset(a1_to_a2_edge, graph)
        a1_to_a2_dst_subset: dace_sbs.Range = a1_to_a2_memlet.get_dst_subset(a1_to_a2_edge, graph)

        # Note that it is possible that `a1` is connected to the same node multiple
        #  times, although through different edges. We have to modify the data
        #  flow there, since the offsets and the data have changed. However, we must
        #  do this only once. Note that only matching the node is not enough, a
        #  counter example would be a Map with different connector names.
        reconfigured_neighbour: set[tuple[dace_nodes.Node, Optional[str]]] = set()

        #  In this section, `old_node` refers to the node, one of `a1` or `a2`,
        #  which will be replaced by the other one. The node which is kept is
        #  referred to as `new_node`.
        new_node: dace_nodes.AccessNode
        old_node: dace_nodes.AccessNode
        new_node_offsets: Sequence[dace_sym.SymbolicType]
        copy_chain_mode = self._get_copy_chain_mode(sdfg, graph)
        if copy_chain_mode == CopyChainRemoverMode.PULL:
            old_node = a1
            new_node = a2
            # Now we compose the new subset.
            #  For the `PULL` mode, we build on the fact that we have ensured that
            #  the whole array `old_node` (`a1`) is copied into `new_node` (`a2`).
            #  Thus the destination of the original source, i.e. whatever writes
            #  into `a1`, is just offset by the beginning of the range `a1` writes
            #  into `a2`.
            #       (s1) ------[c:d]-> (A1) -[0:N]------[a:b]-> (A2)
            #       (s1) ---------[(a + c):(a + c + (d - c))]-> (A2)
            #  Thus the offset is simply given by `a`, the start index where `a1`
            #  is written into `a2`.
            #  NOTE: If we ever allow the that `a1` is not fully read, then we would
            #   have to modify this computation slightly.
            new_node_offsets = a1_to_a2_dst_subset.min_element()
        else:
            assert copy_chain_mode == CopyChainRemoverMode.PUSH
            old_node = a2
            new_node = a1
            # We compose the new subset in a similar way, but using the index where
            #  `a2` starts reading from `a1`.
            new_node_offsets = a1_to_a2_src_subset.min_element()

        # Handle the producer side of things.
        for producer_edge in list(graph.in_edges(old_node)):
            producer: dace_nodes.Node = producer_edge.src
            producer_conn = producer_edge.src_conn
            if producer is new_node:
                assert producer_edge is a1_to_a2_edge
                assert copy_chain_mode == CopyChainRemoverMode.PUSH
                continue
            new_producer_edge = gtx_transformations.utils.reroute_edge(
                is_producer_edge=True,
                current_edge=producer_edge,
                ss_offset=new_node_offsets,
                state=graph,
                sdfg=sdfg,
                old_node=old_node,
                new_node=new_node,
            )
            if (producer, producer_conn) not in reconfigured_neighbour:
                gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                    is_producer_edge=True,
                    new_edge=new_producer_edge,
                    sdfg=sdfg,
                    state=graph,
                    ss_offset=new_node_offsets,
                    old_node=old_node,
                    new_node=new_node,
                )
                reconfigured_neighbour.add((producer, producer_conn))

        # Handle the consumer side of things, as they now have to read from `a2`.
        #  It is important that the offset is still the same.
        for consumer_edge in list(graph.out_edges(old_node)):
            consumer: dace_nodes.Node = consumer_edge.dst
            consumer_conn = consumer_edge.dst_conn
            if consumer is new_node:
                assert consumer_edge is a1_to_a2_edge
                assert copy_chain_mode == CopyChainRemoverMode.PULL
                continue
            new_consumer_edge = gtx_transformations.utils.reroute_edge(
                is_producer_edge=False,
                current_edge=consumer_edge,
                ss_offset=new_node_offsets,
                state=graph,
                sdfg=sdfg,
                old_node=old_node,
                new_node=new_node,
            )
            if (consumer, consumer_conn) not in reconfigured_neighbour:
                gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                    is_producer_edge=False,
                    new_edge=new_consumer_edge,
                    sdfg=sdfg,
                    state=graph,
                    ss_offset=new_node_offsets,
                    old_node=old_node,
                    new_node=new_node,
                )
                reconfigured_neighbour.add((consumer, consumer_conn))

        # After the rerouting we have to delete the transient data node and descriptor,
        #  this will also remove all the old edges.
        graph.remove_node(old_node)
        sdfg.remove_data(old_node.data, validate=False)

        # We will now propagate the strides starting from the access node we keep.
        #  Essentially, this will replace the strides from `old_node` with the ones
        #  of `new_node`. We do it outside to make sure that we do not forget a
        #  case and that we propagate the change into every NestedSDFG only once.
        gtx_transformations.gt_propagate_strides_from_access_node(
            sdfg=sdfg,
            state=graph,
            outer_node=new_node,
        )

    def _get_copy_chain_mode(self, sdfg: dace.SDFG, state: dace.SDFG) -> CopyChainRemoverMode:
        """
        Helper function to detect whether the `CopyChainRemover` tranbsformation
        should be applied in `PULL` or `PUSH` mode, for details see `CopyChainRemoverMode`.
        """
        a1: dace_nodes.AccessNode = self.node_a1
        a2: dace_nodes.AccessNode = self.node_a2

        a1_desc = a1.desc(sdfg)
        a1_range = dace_sbs.Range.from_array(a1_desc)

        a2_desc = a2.desc(sdfg)
        a2_range = dace_sbs.Range.from_array(a2_desc)

        # There shall only be one edge connecting `a1` and `a2`.
        #  We even strengthen this requirement by not checking for the node `a2`,
        #  but for the data.
        connecting_edges = [
            oedge
            for oedge in state.out_edges(a1)
            if isinstance(oedge.dst, dace_nodes.AccessNode) and (oedge.dst.data == a2.data)
        ]
        assert len(connecting_edges) == 1

        # The full array `a1` is copied into `a2`. Note that it is allowed, that
        #  `a2` is bigger than `a1`, it is just important that everything that was
        #  written into `a1` is also accessed.
        connecting_edge = connecting_edges[0]
        assert connecting_edge.dst is a2
        connecting_memlet = connecting_edge.data

        # If the destination or the source subset of the connection is not fully
        #  specified, we do not apply.
        src_subset = connecting_memlet.get_src_subset(connecting_edge, state)
        assert src_subset is not None
        dst_subset = connecting_memlet.get_dst_subset(connecting_edge, state)
        assert dst_subset is not None

        # Checking if the whole array is read.
        # NOTE: The main benefit of requiring that the whole array is read is that we
        #   do not have to adjust maps.
        # NOTE: In previous versions there was an ad hoc rule, to bypass the "full
        #   read rule". However, it caused problems, so it was removed.
        # TODO: We have to improve this test, because sometimes the expressions are
        #   so complex that without information about relations, such as
        #   `vertical_start <= vertical_end` it is not possible to prove this check.
        if a1_desc.transient and self.is_single_use_data(sdfg, a1) and src_subset.covers(a1_range):
            return CopyChainRemoverMode.PULL

        # Checking if the whole array is written.
        if a2_desc.transient and self.is_single_use_data(sdfg, a2) and dst_subset.covers(a2_range):
            return CopyChainRemoverMode.PUSH

        return CopyChainRemoverMode.NONE
