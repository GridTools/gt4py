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

        # If the container that survives the merge is a global that is also read
        #  upstream of the copy (`PULL` mode) or written downstream of it (`PUSH`
        #  mode) in the same state, then the temporary we remove is a needed
        #  double buffer between the old and the new value of that global: merging
        #  it away lets the write overtake a read of the old value, which would
        #  then observe the new one. This is the situation of a field that a
        #  `field_operator` call takes as an input and also produces as an output;
        #  `test_copy_chain_destination_read_in_map_no_apply` and
        #  `test_push_chain_source_written_in_map_no_apply` reproduce the `PULL`
        #  and `PUSH` variants of it.
        # NOTE: This can happen even though the writes are elementwise in the
        #   sense of ADR-18 rule 3, i.e. the write to `A[i]` never reads `A[j]`
        #   for `i != j` (`a += 1` is allowed while `A[0:10] = A[1:11]` is not):
        #   the hazardous access is a different read or write of the same global
        #   that only the removed temporary keeps ordered against the copy. Since
        #   that ordering is not always made explicit in the SDFG, we have to detect
        #   the case here. Alternatively, empty Memlets could be used to explicitly
        #   order the write after read access.
        if self._creates_write_after_read_hazard(sdfg, graph):
            return False

        return True

    def _creates_write_after_read_hazard(
        self,
        sdfg: dace.SDFG,
        graph: dace.SDFGState,
    ) -> bool:
        """Checks if removing the copy chain would let a write overtake a read.

        In `PULL` mode (`(T) -> (G)`, survivor `G`) the surviving container
        takes over the writes of the removed temporary, moving them upstream to
        where that temporary was produced.  If the surviving container is a
        global that another AccessNode of this state reads upstream of the copy,
        that read is no longer guaranteed to happen before the write.

        In `PUSH` mode (`(G) -> (T)`, survivor `G`) the surviving container
        takes over the reads of the removed temporary, moving them downstream to
        where that temporary was consumed.  If the surviving container is a
        global that another AccessNode of this state writes downstream of the
        copy, the moved read would observe the new value instead of the one that
        was frozen in the temporary.

        Args:
            sdfg: The SDFG on which the transformation is applied.
            graph: The state on which the transformation is applied.

        Returns:
            `True` if the merge would create a write-after-read hazard.
        """
        copy_chain_mode = self._get_copy_chain_mode(sdfg, graph)
        if copy_chain_mode == CopyChainRemoverMode.PULL:
            surviving_node = self.node_a2
        elif copy_chain_mode == CopyChainRemoverMode.PUSH:
            surviving_node = self.node_a1
        else:
            return True

        # Transients are written only once (ADR-18), therefore they can never act
        #  as a double buffer: there is no "old" value that a reader could observe,
        #  so moving the single write cannot introduce a write-after-read hazard.
        if surviving_node.desc(sdfg).transient:
            return False

        # TODO(phimuell): This ignores empty Memlets and the order they impose, fix that.
        # TODO(phimuell): This restriction could be relaxed as in some cases (Maps) only
        #    point wise accesses are allowed anyway.
        if copy_chain_mode == CopyChainRemoverMode.PULL:
            # A global AccessNode serializes its incoming writes before its outgoing
            #  reads, so reads that are downstream of the survivor, i.e. pass through,
            #  remain ordered after the moved write.  Only reads that are upstream of
            #  the survivor, i.e. ordered before the original copy write, can observe
            #  the old value and may become racy when the write is moved upstream.
            #  We therefore scan the ancestor region of the survivor for reads of the
            #  same data.  Every node in the upstream set has at least one outgoing
            #  edge (the one we walked backward through), so a plain data match is
            #  enough to detect a read.
            upstream_nodes = gtx_transformations.utils.find_upstream_nodes(surviving_node, graph)
            same_data_nodes = [
                node
                for node in upstream_nodes
                if isinstance(node, dace_nodes.AccessNode) and node.data == surviving_node.data
            ]
            if not same_data_nodes:
                return False

            # Relaxation: a same-data read `n` upstream of the survivor is safe
            #  when, after merging, the moved write can not overtake it.  Two
            #  cases are recognised:
            #
            #  1. Single path (the original relaxation): `a1` has a single
            #     producer `P` and `a2` has no other producer, so `n` reaches
            #     the survivor only through `a1 -> a2`.  When that path is a
            #     single *connector-aware* one, `n` is ordered before `P` both
            #     before and after the merge, so the moved write can not overtake
            #     it.  The connector-aware count is needed because a plain search
            #     fans out across a scope's connectors (a tasklet reading two
            #     inputs gives its entry two `OUT_*`) and would report multiple
            #     paths for a single data-flow branch.  The single-producer
            #     guards are needed because an extra producer `X != a1` could
            #     give `n` a single path through `X` that is not ordered before
            #     `P`.
            #
            #  2. Multiple paths with disjoint subsets: `a1` has several
            #     producers, each writing a disjoint subset of `a1` (which, after
            #     merging, becomes a disjoint subset of the survivor).  A
            #     same-data read `n` reaches the survivor through exactly one of
            #     those producers per outgoing branch.  The moved write of a
            #     *different* producer can only overtake `n` on the subset that
            #     producer writes; if the read subset of every branch is disjoint
            #     from every other branch's moved write, no read can observe a
            #     new value and the merge is safe.  This is the common
            #     field-operator pattern where an output field is updated in
            #     place through several `concat_where`/`where` branches that
            #     each touch a disjoint part of the field.
            producers_of_a1 = [iedge.src for iedge in graph.in_edges(self.node_a1)]
            producers_of_a2 = [iedge.src for iedge in graph.in_edges(surviving_node)]
            if len(producers_of_a1) == 1 and len(producers_of_a2) == 1:
                assert producers_of_a2[0] is self.node_a1
                if all(
                    gtx_transformations.utils.count_paths(node, surviving_node, graph, limit=2) == 1
                    for node in same_data_nodes
                ):
                    return False

                return True

            if len(producers_of_a2) != 1 or producers_of_a2[0] is not self.node_a1:
                # The survivor is written by something other than the copy we
                #  are about to remove; the ordering assumptions below do not
                #  hold.
                return True

            # Map each producer's write subset of `a1` through the copy to the
            #  surviving global's coordinate space.  `copy_dst.compose(W_i)`
            #  applies `W_i` (a sub-range of the copy's source extent) to the
            #  destination, yielding the slice of the survivor that the moved
            #  write would touch.
            copy_edge = next(e for e in graph.out_edges(self.node_a1) if e.dst is surviving_node)
            copy_dst = dace_sbs.Range(copy_edge.data.get_dst_subset(copy_edge, graph))
            moved_write_subsets: list[dace_sbs.Range] = []
            for iedge in graph.in_edges(self.node_a1):
                w_i = dace_sbs.Range(iedge.data.subset)
                try:
                    moved_write_subsets.append(copy_dst.compose(w_i))
                except (ValueError, TypeError):
                    # If the subset mapping cannot be computed, fall back to the
                    #  conservative answer.
                    return True

            for node in same_data_nodes:
                num_paths = gtx_transformations.utils.count_paths(
                    node, surviving_node, graph, limit=2
                )
                if num_paths == 1:
                    # Single connector-aware path through `a1`: the read is
                    #  ordered before the moved write.
                    continue
                # Zero or multiple paths: a "same branch" read is always safe
                #  (the read is ordered before the producer's moved write, either
                #  because they share a single data-flow branch or because the
                #  read is hoisted to a MapEntry whose scope contains the
                #  producer).  For every other pair `(e_i, P_j)` the moved write
                #  `W_j'` must be disjoint from the subset `R_i` that `e_i` reads
                #  of the surviving global.  A zero-path read is only a hazard if
                #  it touches a subset a different producer's moved write reaches:
                #  in the common field-operator pattern the read is hoisted to a
                #  MapEntry (ordered before every inner write) and the producers
                #  write disjoint sub-arrays, so the merge is safe even though no
                #  connector-aware path links the read to the survivor through
                #  the copy chain.  To avoid a MapEntry fanning out across all
                #  connectors, the "same branch" check starts from the first
                #  node downstream of `e_i.dst` through the connector `e_i`
                #  arrives at, and from the first node upstream of `P_j`
                #  through the connector that feeds it.
                producer_edges = list(graph.in_edges(self.node_a1))
                for e_i in graph.out_edges(node):
                    r_i = dace_sbs.Range(e_i.data.subset)
                    # The read `e_i` arrives at `e_i.dst` through
                    #  `e_i.dst_conn`; follow the paired `OUT_*` connector.
                    paired_out = (
                        "OUT_" + e_i.dst_conn[3:]
                        if e_i.dst_conn is not None and e_i.dst_conn.startswith("IN_")
                        else None
                    )
                    read_first = [
                        oe.dst
                        for oe in (
                            graph.out_edges(e_i.dst)
                            if paired_out is None
                            else graph.out_edges_by_connector(e_i.dst, paired_out)
                        )
                    ]
                    for j, p_edge in enumerate(producer_edges):
                        p_j = p_edge.src
                        # The producer `P_j` feeds `a1` through
                        #  `p_edge.src_conn`; follow the paired `IN_*`
                        #  connector to find the producer tasklet.
                        paired_in = (
                            "IN_" + p_edge.src_conn[4:]
                            if p_edge.src_conn is not None and p_edge.src_conn.startswith("OUT_")
                            else None
                        )
                        prod_first = [
                            ie.src
                            for ie in (
                                graph.in_edges(p_j)
                                if paired_in is None
                                else graph.in_edges_by_connector(p_j, paired_in)
                            )
                        ]
                        # The read and the producer are on the same branch when
                        #  they belong to the same scope (a read hoisted to a
                        #  MapEntry is ordered before any inner write of that
                        #  map) or when a downstream node of the read reaches an
                        #  upstream node of the producer.
                        same_scope = (
                            hasattr(e_i.dst, "map")
                            and hasattr(p_j, "map")
                            and e_i.dst.map is p_j.map
                        )
                        same_branch = (
                            e_i.dst is p_j
                            or same_scope
                            or any(
                                (
                                    rf is pf
                                    or (
                                        gtx_transformations.utils.count_paths(  # type: ignore[operator]
                                            rf, pf, graph, limit=1
                                        )
                                        > 0
                                    )
                                )
                                for rf in read_first
                                for pf in prod_first
                            )
                        )
                        if same_branch:
                            # `P_j` is on the same branch as `e_i`: the read
                            #  is ordered before this producer's moved write.
                            continue
                        try:
                            if r_i.intersects(moved_write_subsets[j]):
                                return True
                        except (ValueError, TypeError):
                            return True
            return False
        else:
            assert copy_chain_mode == CopyChainRemoverMode.PUSH
            # A global AccessNode serializes its incoming writes before its outgoing
            #  reads, so writes that are upstream of the survivor remain ordered
            #  before the moved read.  Only writes that are downstream of the
            #  survivor, i.e. between the original copy read and where the
            #  temporary was consumed, can overtake the moved read and make it
            #  observe the new value.  We therefore scan the descendant region of
            #  the survivor for writes to the same data.  Every node in the
            #  downstream set has at least one incoming edge (the one we walked
            #  forward through), so a plain data match is enough to detect a write.
            downstream_nodes = gtx_transformations.utils.find_downstream_nodes(
                surviving_node, graph
            )
            same_data_nodes = [
                node
                for node in downstream_nodes
                if isinstance(node, dace_nodes.AccessNode) and node.data == surviving_node.data
            ]
            if not same_data_nodes:
                return False

            # Mirror of the PULL relaxation: a same-data write `n` downstream
            #  of the survivor is safe when, after merging, the moved read can
            #  not overtake it.  Two cases are recognised, mirroring PULL:
            #
            #  1. Single path (the original relaxation): `a2` has a single
            #     consumer `P` and `a1` has no other consumer, so `n` reaches
            #     `a1` only through `a1 -> a2`.  When that path is a single
            #     *connector-aware* one, `n` stays ordered after `P` and the
            #     moved read keeps observing the frozen value.
            #
            #  2. Multiple paths with disjoint subsets: `a2` has several
            #     consumers, each reading a disjoint subset of `a2` (which,
            #     after merging, becomes a disjoint subset of the survivor).
            #     A same-data write `n` downstream of the survivor is reached
            #     by exactly one of those consumers per incoming branch.  The
            #     moved read of a *different* consumer can only observe a new
            #     value on the subset that consumer reads; if the write subset
            #     of every branch is disjoint from every other branch's moved
            #     read, the merge is safe.  This is the common field-operator
            #     pattern where an input field is also written in place through
            #     several `concat_where`/`where` branches that each touch a
            #     disjoint part of the field.
            consumers_of_a2 = [oedge.dst for oedge in graph.out_edges(self.node_a2)]
            consumers_of_a1 = [oedge.dst for oedge in graph.out_edges(surviving_node)]
            if len(consumers_of_a2) == 1 and len(consumers_of_a1) == 1:
                assert consumers_of_a1[0] is self.node_a2
                if all(
                    gtx_transformations.utils.count_paths(surviving_node, node, graph, limit=2) == 1
                    for node in same_data_nodes
                ):
                    return False
                return True

            # The survivor is a global that is also read by the same-data write
            #  nodes downstream (those reads are what the moved reads would
            #  overtake).  The only consumers of the survivor that are NOT
            #  same-data nodes must be (i) `a2` (the copy we are about to
            #  remove) and (ii) scope nodes (e.g. MapEntries) that only reach
            #  same-data writes — the reads through those scope nodes are what
            #  the write-back consumes, and they are ordered after the moved
            #  read through the copy chain.  Any other consumer could give a
            #  write a single path through it that is not ordered after the
            #  moved read.
            same_data_set = set(same_data_nodes)
            unsafe_consumers = []
            for c in consumers_of_a1:
                if c is self.node_a2 or c in same_data_set:
                    continue
                c_downstream = gtx_transformations.utils.find_downstream_nodes(c, graph)
                c_same_data = {
                    n
                    for n in c_downstream
                    if isinstance(n, dace_nodes.AccessNode) and n.data == surviving_node.data
                }
                if c_same_data.issubset(same_data_set):
                    continue
                unsafe_consumers.append(c)
            if len(unsafe_consumers) != 0 or not any(c is self.node_a2 for c in consumers_of_a1):
                return True

            # Map each consumer's read subset of `a2` through the copy to the
            #  surviving global's coordinate space.  `copy_src.compose(R_j)`
            #  applies `R_j` (a sub-range of the copy's destination extent,
            #  which starts at 0 because the whole transient `a2` is copied)
            #  to the source, yielding the slice of the survivor that the
            #  moved read would access.
            copy_edge = next(e for e in graph.in_edges(self.node_a2) if e.src is surviving_node)
            copy_src = dace_sbs.Range(copy_edge.data.get_src_subset(copy_edge, graph))
            # Track each consumer edge so we can start connector-aware path
            #  searches from the specific branch the transient feeds into.
            consumer_edges = list(graph.out_edges(self.node_a2))
            moved_read_subsets: list[dace_sbs.Range] = []
            for c_edge in consumer_edges:
                r_j = dace_sbs.Range(c_edge.data.subset)
                try:
                    moved_read_subsets.append(copy_src.compose(r_j))
                except (ValueError, TypeError):
                    # If the subset mapping cannot be computed, fall back to
                    #  the conservative answer.
                    return True

            for node in same_data_nodes:
                # For each incoming branch `e_i` (write `W_i` of the surviving
                #  global by node `n`) and every consumer `C_j` of `a2` that is
                #  *not* on `e_i`'s branch, the moved read `R_j'` must be
                #  disjoint from `W_i`.  Otherwise the read moved into `C_j`'s
                #  branch could overtake the write in `e_i`'s branch.  When
                #  `C_j` IS on `e_i`'s branch the moved read is ordered before
                #  the write (read of `b` before write of `a` in the same
                #  tasklet), so that pair is always safe.
                for e_i in graph.in_edges(node):
                    w_i = dace_sbs.Range(e_i.data.subset)
                    for j, c_edge in enumerate(consumer_edges):
                        c_j = c_edge.dst
                        # To check whether `C_j` is on the same branch as the
                        #  write `e_i`, start a connector-aware search from
                        #  the first node downstream of `C_j` through the
                        #  connector the transient feeds into (so a MapEntry
                        #  with multiple branches does not fan out across
                        #  all of them).  The write `e_i` comes from
                        #  `e_i.src` through `e_i.src_conn`; to avoid matching
                        #  through a *different* connector of the same scope
                        #  node, we check reachability to the actual write
                        #  tasklet — the source of the edge that feeds
                        #  `e_i.src` through `e_i.src_conn` — not to
                        #  `e_i.src` itself.
                        paired_conn = (
                            "OUT_" + c_edge.dst_conn[3:]
                            if c_edge.dst_conn is not None and c_edge.dst_conn.startswith("IN_")
                            else None
                        )
                        c_first = [
                            oe.dst
                            for oe in (
                                graph.out_edges(c_j)
                                if paired_conn is None
                                else graph.out_edges_by_connector(c_j, paired_conn)
                            )
                        ]
                        # The write tasklet: the source of the edge into
                        #  `e_i.src` through the connector paired with
                        #  `e_i.src_conn` (e.g. `e_i.src_conn=OUT_aw` pairs
                        #  with `IN_aw`).
                        paired_in = (
                            "IN_" + e_i.src_conn[4:]
                            if e_i.src_conn is not None and e_i.src_conn.startswith("OUT_")
                            else e_i.src_conn
                        )
                        write_tasklets = [
                            ie.src for ie in graph.in_edges(e_i.src) if ie.dst_conn == paired_in
                        ]
                        same_branch = e_i.src is c_j or any(
                            (
                                f is wt
                                or gtx_transformations.utils.count_paths(f, wt, graph, limit=1) > 0  # type: ignore[operator]
                            )
                            for f in c_first
                            for wt in write_tasklets
                        )
                        if same_branch:
                            # `C_j` is on the same branch as `e_i`: the write
                            #  is ordered after this consumer's moved read.
                            continue
                        try:
                            if w_i.intersects(moved_read_subsets[j]):
                                return True
                        except (ValueError, TypeError):
                            return True
            return False

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
