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
    data as dace_data,
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
        copy_chain_mode = self._get_copy_chain_mode(sdfg, graph)

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


@dace_properties.make_properties
class DoubleWriteRemover(dace_transformation.SingleStateTransformation):
    """Removes Redundant writes that are related to slices.

    Consider the following code:
    ```python
    for i in range(N):
        temp[i] = ...
    A[:, slice1] = temp
    A[:, slice2] = temp
    ```
    The function will transform it into the following:
    ```python
    for i in range(N):
        temp_scalar = ...
        A[i, slice1] = temp_scalar
        A[i, slice2] = temp_scalar
    ```

    The transformation will only apply if:
    - `temp` is single use data.
    - `temp` is only consumed by other AccessNodes and not Maps.
    - `temp` must be fully copied into its destination, i.e. not partially.

    Args:
        single_use_data: List of data containers that are used only at one place.
            Will be stored internally and not updated.

    Todo:
        - Extend such that not the full array must be read.
    """

    map_exit = dace_transformation.PatternNode(dace_nodes.MapExit)
    temp_node = dace_transformation.PatternNode(dace_nodes.AccessNode)

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and be passed at construction time. Needed until
    #  [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: Optional[dict[dace.SDFG, set[str]]]

    def __init__(
        self,
        *args: Any,
        single_use_data: Optional[dict[dace.SDFG, set[str]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._single_use_data = single_use_data

    @classmethod
    def expressions(cls) -> Any:
        return [
            dace.sdfg.utils.node_path_graph(
                cls.map_exit,
                cls.temp_node,
            )
        ]

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        map_exit: dace_nodes.MapExit = self.map_exit
        temp_node: dace_nodes.AccessNode = self.temp_node
        temp_desc = temp_node.desc(sdfg)

        # Only the Map can produce data.
        if graph.in_degree(temp_node) != 1:
            return False

        producer_edge = next(iter(graph.in_edges(temp_node)))
        produced_sbs: dace_sbs.Range = producer_edge.data.get_dst_subset(producer_edge, graph)
        assert produced_sbs is not None

        # Only step 1 is allowed.
        if any(step != 1 for _, _, step in map_exit.map.range):
            return False
        if any(step != 1 for _, _, step in produced_sbs):
            return False

        # To simplify implementation we only allow that there is one producer inside the Map.
        assert producer_edge.src_conn.startswith("OUT_")
        inner_producer_edges = [
            inner_map_edge
            for inner_map_edge in graph.in_edges_by_connector(
                map_exit, "IN_" + producer_edge.src_conn[4:]
            )
        ]
        if len(inner_producer_edges) != 1:
            return False
        inner_producer_edge = inner_producer_edges[0]

        # To simplify implementation, ensures that `temp_node` does not have any "dummy"
        #  dimensions, i.e. dimensions of size 1. Or differently, that every Map parameter
        #  is used in indexing.
        # TODO(phimuell): Lift this limitation.
        # TODO(phimuell): Is this check strong enough.
        if len(map_exit.map.params) != produced_sbs.dims():
            return False

        # To further simplify, we require that the Memlet is "simple" and that only a
        #  scalar is moved around.
        inner_production_sbs = inner_producer_edge.data.get_dst_subset(inner_producer_edge, graph)
        if inner_production_sbs.num_elements() != 1:
            return False
        if inner_producer_edge.data.allow_oob:
            return False
        if inner_producer_edge.data.is_empty():
            return False
        if inner_producer_edge.data.wcr is not None:
            return False

        # We only allow arrays that are transients.
        if not isinstance(temp_desc, dace_data.Array):
            return False
        if isinstance(temp_desc, dace_data.View):
            return False
        if not temp_desc.transient:
            return False

        # Check the consumer
        for consumer_edge in graph.out_edges(temp_node):
            consumer_sbs: dace_sbs.Range = consumer_edge.data.get_src_subset(consumer_edge, graph)
            if consumer_sbs is None:
                return False

            # What is produced must be consumed.
            if consumer_sbs != produced_sbs:
                return False
            if any(step != 1 for _, _, step in consumer_sbs):
                return False

            # Since `temp_node` is eliminated, the consumer must be another AccessNode.
            # TODO(phimuell): Allow the consumer to be a Map, in which case `temp_node`
            #   should not be removed.`
            consumer_node = consumer_edge.dst
            if not isinstance(consumer_node, dace_nodes.AccessNode):
                return False

            consumer_desc = consumer_node.desc(sdfg)
            if not isinstance(consumer_desc, dace_data.Array):
                return False
            if isinstance(consumer_desc, dace_data.View):
                return False

        # `temp_node` will be removed this means it must be single use.
        if self._single_use_data is None:
            find_single_use_data = dace_analysis.FindSingleUseData()
            single_use_data = find_single_use_data.apply_pass(sdfg, None)
        else:
            single_use_data = self._single_use_data
        if temp_node.data not in single_use_data[sdfg]:
            return False

        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        map_exit: dace_nodes.MapExit = self.map_exit
        temp_node: dace_nodes.AccessNode = self.temp_node
        temp_desc = temp_node.desc(sdfg)

        # Understand how the Map writes into the `temp_node`. For this we have to
        #  look at the subset of the inner edge that writes into the node.
        assert graph.in_degree(temp_node) == 1
        outer_producer_edge = next(iter(graph.in_edges(temp_node)))
        outer_producer_subset = outer_producer_edge.data.get_dst_subset(outer_producer_edge, graph)
        assert outer_producer_subset is not None
        assert outer_producer_edge.src_conn.startswith("OUT_")
        inner_producer_edge = next(
            iter(
                inner_producer_edge
                for inner_producer_edge in graph.in_edges_by_connector(
                    map_exit, "IN_" + outer_producer_edge.src_conn[4:]
                )
            )
        )
        inner_producer_subset = inner_producer_edge.data.get_dst_subset(inner_producer_edge, graph)

        # Because we distribute to multiple consumers, we need some temporary storage
        #  inside the Map scope, aka. AccessNode. We will first check if there is
        #  already an AccessNode that can be used or if we have to create a new one.
        current_inner_producer_node = inner_producer_edge.src
        if not isinstance(current_inner_producer_node, dace_nodes.AccessNode):
            reuse_inner_producer_node = False
        else:
            current_inner_producer_desc = current_inner_producer_node.desc(sdfg)
            if isinstance(current_inner_producer_desc, dace_data.View):
                reuse_inner_producer_node = False
            elif isinstance(current_inner_producer_desc, dace_data.Scalar):
                reuse_inner_producer_node = True
            elif (
                isinstance(current_inner_producer_desc, dace_data.Array)
                and len(current_inner_producer_desc.shape) == 1
                and (current_inner_producer_desc.shape[0] == 1) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
            ):
                reuse_inner_producer_node = True
            else:
                reuse_inner_producer_node = False

        if reuse_inner_producer_node:
            # We can reuse the one that is already there.
            inner_distribution_node: dace_nodes.AccessNode = current_inner_producer_node
            assert inner_producer_edge.src_conn is None
        else:
            # We have to create a new intermediate storage inside the Map scope.
            #  For this we are using a scalar.
            assert inner_producer_edge.data.dst_subset.num_elements() == 1
            distribution_data, _ = sdfg.add_scalar(
                "__gtx_double_write_remover_inner_inner_distribution_node",
                dtype=temp_desc.dtype,
                storage=dace.dtypes.StorageType.Register,
                transient=True,
                find_new_name=True,
            )
            inner_distribution_node = graph.add_access(distribution_data)
            graph.add_edge(
                inner_producer_edge.src,
                inner_producer_edge.src_conn,
                inner_distribution_node,
                None,
                dace.Memlet(
                    data=distribution_data,
                    subset="0",
                    other_subset=inner_producer_edge.data.src_subset,
                ),
            )

        # Now we have to delete the producer Memlet tree, which consists of
        #  `{inner, outer}_producer_edge`.
        # NOTE: `outer_producer_edge` and `inner_producer_edge` are no invalid.
        graph.remove_edge(outer_producer_edge)
        map_exit.remove_out_connector(outer_producer_edge.src_conn)
        graph.remove_edge(inner_producer_edge)
        map_exit.remove_in_connector(inner_producer_edge.dst_conn)
        assert graph.in_degree(temp_node) == 0 and graph.out_degree(temp_node) > 0

        # Now reroute the dataflow, instead going through `temp_node` directly serve
        #  them from the Map. To be precise use the `inner_distribution_node`.
        for consumer_edge in list(graph.out_edges(temp_node)):
            consumer_node = consumer_edge.dst
            consumer_destination = consumer_edge.data.get_dst_subset(consumer_edge, graph)
            assert isinstance(consumer_node, dace_nodes.AccessNode)

            # Since we want to understand how the destination is written we have to
            #  pass `dst_subset` as `sbs1` and `src_subset`, which reads from
            #  `temp_node` as `sbs2`.
            consumer_to_temp_mapping, consumer_drop, temp_drop = (
                gtx_transformations.utils.associate_dimmensions(
                    sbs1=consumer_edge.data.dst_subset,
                    sbs2=consumer_edge.data.src_subset,
                )
            )
            assert len(temp_drop) == 0

            # Now compose the subset (on the inside of the Map) that is used to write
            #  into `consumer_node`.
            inner_map_output_subset: list[tuple[Any, Any, int]] = []
            consumer_dst_subset = consumer_edge.data.dst_subset
            for consumer_dim in range(consumer_dst_subset.dims()):
                if consumer_dim in consumer_to_temp_mapping:
                    # This dimension is copied from `temp_node` into `consumer_node`.
                    #  Thus we have to use same subset that was used to write the
                    #  original `temp_node` thing.
                    temp_node_dim = consumer_to_temp_mapping[consumer_dim]

                    # Compute the correction.
                    consumer_correction = consumer_destination[consumer_dim][0]
                    temp_correction = outer_producer_subset[temp_node_dim][0]
                    correction = consumer_correction - temp_correction

                    # Compose the final subset.
                    inner_producer_idx = inner_producer_subset[temp_node_dim]
                    inner_map_output_subset.append(
                        (inner_producer_idx[0] + correction, inner_producer_idx[1] + correction, 1)
                    )
                else:
                    # The dimension is not written to by a Map parameter, which means
                    #  that it is some dummy dimension, such as an offset. Thus we use
                    #  the original subset.
                    #  There is no correction needed here.
                    assert consumer_dim in consumer_drop
                    inner_map_output_subset.append(consumer_dst_subset[consumer_dim])
            new_inner_dst_subset = dace_sbs.Range(inner_map_output_subset)

            # Now we create the new connection from the `inner_distribution_node` to
            #  the final consumer. For the connection from the MapExit to the consumer
            #  we essentially recycle the Memlet, but create a new one, to ensure that
            #  it has the canonical format.
            # TODO: We can not use `graph.add_memlet_path()` because sometimes Memlet
            #   propagation fails. Here we are reusing the subsets that were already
            #   there. Which assumes that they are correct.
            connector_name = map_exit.next_connector()
            graph.add_edge(
                inner_distribution_node,
                None,
                map_exit,
                "IN_" + connector_name,
                dace.Memlet(
                    data=consumer_edge.dst.data,
                    subset=new_inner_dst_subset,
                    other_subset=dace_sbs.Range.from_string("0"),
                ),
            )
            graph.add_edge(
                map_exit,
                "OUT_" + connector_name,
                consumer_edge.dst,
                consumer_edge.dst_conn,
                dace.Memlet(
                    data=consumer_edge.dst.data,
                    subset=consumer_edge.data.dst_subset,
                    other_subset=None,
                ),
            )
            map_exit.add_scope_connectors(connector_name)
            graph.remove_edge(consumer_edge)

        # Remove `temp_node` and the data it refers to as it is no longer needed.
        assert graph.out_degree(temp_node) == 0
        graph.remove_node(temp_node)
        sdfg.remove_data(temp_node.data, validate=False)
