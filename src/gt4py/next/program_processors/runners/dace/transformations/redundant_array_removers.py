# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
    validate: bool = False,
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


@dace_properties.make_properties
class SingleStateGlobalSelfCopyElimination(dace_transformation.SingleStateTransformation):
    """Remove global self copy.

    This transformation matches the following case `(G) -> (T) -> (G)`, i.e. `G`
    is read from and written too at the same time, however, in between is `T`
    used as a buffer. In the example above `G` is a global memory and `T` is a
    temporary. The transformation applies if the only incoming edge of the second
    `G` AccessNode comes from the `T` node.
    In case there are direct connections between the two `G` nodes, they will
    be removed as well.

    The transformation will then remove the second `G` node, all reads from the
    second `G` node will redirected such that they use the first `G` node.
    Furthermore, if `T` is not used downstream then also the `T` node will be
    removed as well.

    This transformation assumes that the SDFG follows rule 3 of ADR-18, which
    guarantees that there is only a point wise dependency of the output on the
    input.

    Todo:
        If `T` is only ready in this state, then `T` should not be created, instead
        it should be read directly from `G`.
    """

    node_read_g = dace_transformation.PatternNode(dace_nodes.AccessNode)
    node_tmp = dace_transformation.transformation.PatternNode(dace_nodes.AccessNode)
    node_write_g = dace_transformation.PatternNode(dace_nodes.AccessNode)

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.node_read_g, cls.node_tmp, cls.node_write_g)]

    def can_be_applied(
        self,
        graph: dace.SDFGState | dace.SDFG,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        read_g = self.node_read_g
        write_g = self.node_write_g
        tmp_node = self.node_tmp
        g_desc = read_g.desc(sdfg)
        tmp_desc = tmp_node.desc(sdfg)

        # NOTE: We do not check if `G` is read downstream.
        if read_g.data != write_g.data:
            return False
        if g_desc.transient:
            return False
        if not tmp_desc.transient:
            return False
        if graph.scope_dict()[read_g] is not None:
            return False

        # For ease of implementation we require that only `G` defines `T`. This is
        #  needed to ensure that `G` only depends on `G`.
        if graph.in_degree(tmp_node) != 1:
            return False

        # Now we look at all incoming connections of the second `G` nodes, they must
        #  either come from `T` or from the first `G` node directly.
        for iedge in graph.in_edges(write_g):
            if iedge.src is read_g:
                continue
            if iedge.src is tmp_node:
                continue
            return False

        return True

    def _is_read_downstream(
        self,
        start_state: dace.SDFGState,
        sdfg: dace.SDFG,
        data_to_look: str,
    ) -> bool:
        """Scans for reads to `data_to_look`.

        The function will go through states that are reachable from `start_state`
        (including) and test if there is a read to the data container `data_to_look`.
        It will return `True` the first time it finds such a node.
        It is important that the matched nodes, i.e. `self.node_{read_g, write_g, tmp}`
        are ignored.

        Args:
            start_state: The state where the scanning starts.
            sdfg: The SDFG on which we operate.
            data_to_look: The data that we want to look for.

        Todo:
            Port this function to use DaCe pass pipeline.
        """
        read_g: dace_nodes.AccessNode = self.node_read_g
        write_g: dace_nodes.AccessNode = self.node_write_g
        tmp_node: dace_nodes.AccessNode = self.node_tmp

        # TODO(phimuell): Run the `StateReachability` pass in a pipeline and use
        #  the `_pipeline_results` member to access the data.
        return gtx_transformations.utils.is_accessed_downstream(
            start_state=start_state,
            sdfg=sdfg,
            reachable_states=None,
            data_to_look=data_to_look,
            nodes_to_ignore={read_g, write_g, tmp_node},
        )

    def apply(
        self,
        graph: dace.SDFGState | dace.SDFG,
        sdfg: dace.SDFG,
    ) -> None:
        read_g: dace_nodes.AccessNode = self.node_read_g
        write_g: dace_nodes.AccessNode = self.node_write_g
        tmp_node: dace_nodes.AccessNode = self.node_tmp

        # First check if the `T` node is still needed. The node is needed if the data
        #  is referenced downstream or if there are edges other than to the second
        #  `G` node.
        if not all(oedge.dst is write_g for oedge in graph.out_edges(tmp_node)):
            tmp_node_is_still_needed = True
        elif self._is_read_downstream(start_state=graph, sdfg=sdfg, data_to_look=tmp_node.data):
            tmp_node_is_still_needed = True
        else:
            tmp_node_is_still_needed = False

        # Redirect the reads from the second `G` node such that they now go through
        #  the first `G` node. Because the names are the same there is no special
        #  handling needed.
        # NOTE: There are no writes to the second `G` nodes beside directly the one
        #  coming from the first `G` node or the `T` node. Thus nothing to do.
        for wg_oedge in list(graph.out_edges(write_g)):
            graph.add_edge(
                read_g,
                wg_oedge.src_conn,
                wg_oedge.dst,
                wg_oedge.dst_conn,
                wg_oedge.data,
            )
            graph.remove_edge(wg_oedge)

        # Now we remove the second `G` node as it is no longer needed.
        graph.remove_node(write_g)

        # If the `T` is no longer needed then remove it. If the first node has become
        #  isolated, also remove it.
        if not tmp_node_is_still_needed:
            graph.remove_node(tmp_node)

            if graph.degree(read_g) == 0:
                graph.remove_node(read_g)

            # If it fails then `T` is still used in a parallel controlflow path, for
            #  example in `if` branches.
            try:
                sdfg.remove_data(tmp_node.data, validate=True)
            except ValueError as e:
                if not str(e).startswith(f"Cannot remove data descriptor {tmp_node.data}:"):
                    raise


@dace_properties.make_properties
class SingleStateGlobalDirectSelfCopyElimination(dace_transformation.SingleStateTransformation):
    """Removes an unbuffered self copy in a state.

    This transformation is extremely similar to `SingleStateGlobalSelfCopyElimination`,
    however, this transformation does not need a buffer between the two global
    AccessNodes. Therefore it matches the pattern `(G) -> (G)`, where `G` refers to
    global data. Note that the transformation only considers copies on global scope.
    The transformation has two modes in how this is achieved, which one is chosen,
    depends on if cycles are created or not. The first mode, which is selected if
    cycles would be created, is to merge the two AccessNodes together. For this
    the edges of the first AccessNode are reconnected to the second AccessNode.

    In the second mode, which is selected if cycles would be created, for example in
    case of `if` branches, then the edges are removed, which will create two
    independent `G` AccessNodes.

    Todo:
        Merge this transformation with `SingleStateGlobalSelfCopyElimination`.
    """

    node_read_g = dace_transformation.PatternNode(dace_nodes.AccessNode)
    node_write_g = dace_transformation.PatternNode(dace_nodes.AccessNode)

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.node_read_g, cls.node_write_g)]

    def can_be_applied(
        self,
        graph: dace.SDFGState | dace.SDFG,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        read_g = self.node_read_g
        write_g = self.node_write_g
        g_desc = read_g.desc(sdfg)

        # NOTE: We do not check if `G` is read downstream.
        if read_g.data != write_g.data:
            return False
        if g_desc.transient:
            return False
        # Restrict to global scope.
        if graph.scope_dict()[read_g] is not None:
            return False

        return True

    def apply(
        self,
        graph: dace.SDFGState | dace.SDFG,
        sdfg: dace.SDFG,
    ) -> None:
        read_g: dace_nodes.AccessNode = self.node_read_g
        write_g: dace_nodes.AccessNode = self.node_write_g

        # Select in which mode the transformation should run, depending on
        #  if this would lead to cycles.
        merge_g_nodes = True
        for oedge in graph.out_edges(read_g):
            if oedge.dst is write_g:
                continue
            if gtx_transformations.utils.is_reachable(
                start=oedge.dst,
                target=write_g,
                state=graph,
            ):
                merge_g_nodes = False
                break

        if merge_g_nodes:
            self._merge_g_nodes(state=graph, sdfg=sdfg)
        else:
            self._split_g_nodes(state=graph)

    def _split_g_nodes(
        self,
        state: dace.SDFGState,
    ) -> None:
        """Second mode of the transformation, i.e. the splitting.

        All edges between the two nodes, will be removed. There is no longer a
        direct connection between the two nodes.
        """
        read_g: dace_nodes.AccessNode = self.node_read_g
        write_g: dace_nodes.AccessNode = self.node_write_g

        for oedge in list(state.out_edges(read_g)):
            if oedge.dst is write_g:
                state.remove_edge(oedge)

        assert state.degree(read_g) != 0
        assert state.degree(write_g) != 0

    def _merge_g_nodes(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        """The first mode of the transformation, i.e. the merging of the nodes.

        All incoming edges of the first node will be reconnected to the second node.
        The outgoing edges will be reconnected to the second node, if they connect
        the two nodes, they will be removed.
        """
        read_g: dace_nodes.AccessNode = self.node_read_g
        write_g: dace_nodes.AccessNode = self.node_write_g

        # Reconnect any incoming edges of the read AccessNode, aka. the first one.
        for iedge in list(state.in_edges(read_g)):
            # Since the names are the same, just creating a new edge is enough.
            state.add_edge(
                iedge.src,
                iedge.src_conn,
                write_g,
                iedge.dst_conn,
                iedge.data,
            )
            state.remove_edge(iedge)

        # Now reconnect all edges that leave the first AccessNodes, with the
        #  exception of the edges that connecting the two, they are just removed.
        for oedge in list(state.out_edges(read_g)):
            if oedge.dst is not write_g:
                state.add_edge(
                    write_g,
                    oedge.src_conn,
                    oedge.dst,
                    oedge.dst_conn,
                    oedge.data,
                )
            state.remove_edge(oedge)

        # Now remove the first node to `G` and if the second has become isolated
        #  then also remove it.
        state.remove_node(read_g)

        if state.degree(write_g) == 0:
            state.remove_node(write_g)


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
    - `A1` is a transient and must have the same dimensionality than `A2`.
    - `A1` is fully read by `A2`.

    In certain cases the last rule can not be verified, an alternative formulation,
    which is a consequence of the lowering and domain interference, will be checked:
    - `A1` is read from the beginning, i.e. all subsets starts at literal `0`.
    - `A1` has only one going edge.
    - `A2` is global memory.

    Notes:
        - The transformation assumes that the domain inference adjusted the ranges of
            the maps such that, in case they write into a transient, the full shape of the transient array is written.
            has the same size, i.e. there is not padding, or data that is not written
            to.

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

        # We remove `a1` so it must be a transient and used only once.
        if not a1_desc.transient:
            return False
        if not self.is_single_use_data(sdfg, a1):
            return False

        # This avoids that we have to modify the subsets in a fancy way.
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

        # Checking if the whole array is read.
        #  As described in the description of the class there are two different
        #  formulation of this rule, either the simple one or one that proves this
        #  indirectly.
        # NOTE: The main benefit of requiring that the whole array is read is
        #  that we do not have to adjust maps.
        a1_range = dace_sbs.Range.from_array(a1_desc)
        if src_subset.covers(a1_range):
            pass
        elif all(ss.is_constant() for ss in src_subset.size()):
            # If the subset is fully known then we require that it is covered.
            return False
        elif (
            (not a2_desc.transient)
            and (graph.out_degree(a1) == 1)
            and all(ss_start == 0 for ss_start in src_subset.min_element())
        ):
            pass
        else:
            return False

        # We have to ensure that no cycle is created through the removal of `a1`.
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

        # NOTE: In case `a2` is a non transient we do not have to check if it is read
        #   or written to somewhere else in this state. The reason is that ADR18
        #   guarantees us that everything is point wise, therefore `a1` is never
        #   used as double buffer.
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
        a1_to_a2_dst_subset: dace_sbs.Range = a1_to_a2_memlet.get_dst_subset(a1_to_a2_edge, graph)

        # Note that it is possible that `a1` is connected to the same node multiple
        #  times, although through different edges. We have to modify the data
        #  flow there, since the offsets and the data have changed. However, we must
        #  do this only once. Note that only matching the node is not enough, a
        #  counter example would be a Map with different connector names.
        reconfigured_neighbour: set[tuple[dace_nodes.Node, Optional[str]]] = set()

        # Now we compose the new subset.
        #  We build on the fact that we have ensured that the whole array `a1` is
        #  copied into `a2`. Thus the destination of the original source, i.e.
        #  whatever write into `a1`, is just offset by the beginning of the range
        #  `a1` writes into `a2`.
        #       (s1) ------[c:d]-> (A1) -[0:N]------[a:b]-> (A2)
        #       (s1) ---------[(a + c):(a + c + (d - c))]-> (A2)
        #  Thus the offset is simply given by `a`, the start where `a1` is written into
        #  `a2`.
        #  NOTE: If we ever allow the that `a1` is not fully read, then we would have
        #   to modify this computation slightly.
        a2_offsets: Sequence[dace_sym.SymExpr] = a1_to_a2_dst_subset.min_element()

        # Handle the producer side of things.
        for producer_edge in list(graph.in_edges(a1)):
            producer: dace_nodes.Node = producer_edge.src
            producer_conn = producer_edge.src_conn
            new_producer_edge = gtx_transformations.utils.reroute_edge(
                is_producer_edge=True,
                current_edge=producer_edge,
                ss_offset=a2_offsets,
                state=graph,
                sdfg=sdfg,
                old_node=a1,
                new_node=a2,
            )
            if (producer, producer_conn) not in reconfigured_neighbour:
                gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                    is_producer_edge=True,
                    new_edge=new_producer_edge,
                    sdfg=sdfg,
                    state=graph,
                    ss_offset=a2_offsets,
                    old_node=a1,
                    new_node=a2,
                )
                reconfigured_neighbour.add((producer, producer_conn))

        # Handle the consumer side of things, as they now have to read from `a2`.
        #  It is important that the offset is still the same.
        for consumer_edge in list(graph.out_edges(a1)):
            consumer: dace_nodes.Node = consumer_edge.dst
            consumer_conn = consumer_edge.dst_conn
            if consumer is a2:
                assert consumer_edge is a1_to_a2_edge
                continue
            new_consumer_edge = gtx_transformations.utils.reroute_edge(
                is_producer_edge=False,
                current_edge=consumer_edge,
                ss_offset=a2_offsets,
                state=graph,
                sdfg=sdfg,
                old_node=a1,
                new_node=a2,
            )
            if (consumer, consumer_conn) not in reconfigured_neighbour:
                gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                    is_producer_edge=False,
                    new_edge=new_consumer_edge,
                    sdfg=sdfg,
                    state=graph,
                    ss_offset=a2_offsets,
                    old_node=a1,
                    new_node=a2,
                )
                reconfigured_neighbour.add((consumer, consumer_conn))

        # After the rerouting we have to delete the `a1` data node and descriptor,
        #  this will also remove all the old edges.
        graph.remove_node(a1)
        sdfg.remove_data(a1.data, validate=False)

        # We will now propagate the strides starting from the access nodes `a2`.
        #  Essentially, this will replace the strides from `a1` with the ones of
        #  `a2`. We do it outside to make sure that we do not forget a case and
        #  that we propagate the change into every NestedSDFG only once.
        gtx_transformations.gt_propagate_strides_from_access_node(
            sdfg=sdfg,
            state=graph,
            outer_node=a2,
        )
