# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Elimination of write back buffers whose content is also consumed."""

from typing import Any, Optional

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    subsets as dace_subsets,
    transformation as dace_transformation,
)
from dace.sdfg import nodes as dace_nodes
from dace.transformation import pass_pipeline as dace_ppl
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


def _is_written(state: dace.SDFGState, node: dace_nodes.AccessNode) -> bool:
    """Tells if data flows into `node`; an empty Memlet only imposes an order."""
    return any(not edge.data.is_empty() for edge in state.in_edges(node))


def _has_view(sdfg: dace.SDFG) -> bool:
    """Tells if `sdfg`, or an SDFG nested inside it, contains a View.

    A View has its own descriptor, whose strides are derived from the data it refers
    to, and nothing updates them when the accesses that the View is built from are
    moved to another array.
    """
    return any(
        isinstance(node.desc(nsdfg), dace_data.View)
        for nsdfg in sdfg.all_sdfgs_recursive()
        for state in nsdfg.states()
        for node in state.data_nodes()
    )


def _runs_strictly_before(
    first: dace.SDFGState,
    second: dace.SDFGState,
    reachable: dict[dace.SDFGState, set[dace.SDFGState]],
) -> bool:
    """Tells if `second` always starts after `first` has finished.

    ADR-18 rule 5 forbids cycles in the state graph but mandates `LoopRegion` for
    loops, so `reachable`, which is a may reach relation, lists the states of a loop
    body as reaching each other. Two states of the same loop body are therefore never
    strictly ordered, whatever their order inside the body is.
    """
    return second in reachable[first] and first not in reachable[second]


def _accesses_region(
    state: dace.SDFGState,
    node: dace_nodes.AccessNode,
    region: dace_subsets.Subset,
    incoming: bool,
) -> bool:
    """Tells if the in or out edges of `node` may touch `region` of its data."""
    edges = state.in_edges(node) if incoming else state.out_edges(node)
    for edge in edges:
        if edge.data.is_empty():
            continue
        subset = (
            edge.data.get_dst_subset(edge, state)
            if incoming
            else edge.data.get_src_subset(edge, state)
        )
        if subset is None or dace_subsets.intersects(subset, region) is not False:
            return True
    return False


@dace_transformation.explicit_cf_compatible
@dace_properties.make_properties
class GT4PyWriteBackBufferElimination(dace_transformation.Pass):
    """Removes a write back buffer whose content is also consumed.

    Matches a transient `T` that is written in one or more states, whatever produces
    it, copied in full into a non transient `G` by a later state and additionally
    read by other consumers. Neither
    `DistributedBufferRelocator` (requires `out_degree(T) == 1`) nor
    `GT4PyMapBufferElimination` (requires the copy to be in the state where `T` is
    written, and `T` to be unused downstream) applies then, so the full array copy
    survives as a device to device transfer.

    Every access to `T` is rewritten into an access to `G`, shifted by the offset of
    the copy, and the copy is removed. The producer of `T` and all of its consumers
    thus operate on `G` directly.

    `T` must be copied in full into `G`. Otherwise the producer would write `G`
    outside the range that the copy would have touched, which is not allowed because
    the range of `G` outside the copy is not part of the requested write.

    What the rewrite changes is when `G` is written: from the write back to the point
    where `T` is defined. The range of `G` that the copy covers therefore has to hold
    its old value between the definition of `T` and the write back, which is what
    `_has_conflicting_global_access()` establishes.

    The consumers of `T` need no condition of their own. From the definition of `T`
    onwards that range of `G` holds what the producer computed, exactly what `T`
    held, and `_has_conflicting_global_access()` declines when anything else writes
    that range without being ordered before every definition of `T`. A consumer of
    `T` is therefore served by `G` wherever it is, before or after the write back.

    Args:
        assume_pointwise: Assume that a read of `G` in a state that defines `T` is
            elementwise, so that ADR-18 rule 3 permits it to alias the write.

    Notes:
        - Like `GT4PyMapBufferElimination`, the pass does not test that a read of `G`
            by the producer of `T` is really elementwise, it only tests that it is
            ordered before the definition of `T`, see `_is_read_by_tmp_producer()`.
            Specifying `assume_pointwise` asserts the rest, which ADR-18 rule 3
            guarantees for a valid GT4Py program.
    """

    assume_pointwise = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="See docs.",
    )

    def __init__(
        self,
        assume_pointwise: Optional[bool] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if assume_pointwise is not None:
            self.assume_pointwise = assume_pointwise

    def modifies(self) -> dace_ppl.Modifies:
        # `Descriptors` because `T` is removed, `NestedSDFGs` because the strides of
        #  the descriptors that were mapped from `T` are adjusted.
        return (
            dace_ppl.Modifies.Memlets
            | dace_ppl.Modifies.AccessNodes
            | dace_ppl.Modifies.Descriptors
            | dace_ppl.Modifies.NestedSDFGs
        )

    def should_reapply(self, modified: dace_ppl.Modifies) -> bool:
        # `Descriptors` because data that becomes transient turns into a candidate,
        #  `NestedSDFGs` because a View inside one blocks the pass, see `_has_view()`.
        return modified & (
            dace_ppl.Modifies.Memlets
            | dace_ppl.Modifies.AccessNodes
            | dace_ppl.Modifies.Descriptors
            | dace_ppl.Modifies.NestedSDFGs
        )

    def depends_on(self) -> list[type[dace_transformation.Pass]]:
        return [dace_analysis.StateReachability, dace_analysis.FindAccessNodes]

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: dict[str, Any]) -> Optional[set[str]]:
        reachable: dict[dace.SDFGState, set[dace.SDFGState]] = pipeline_results[
            "StateReachability"
        ][sdfg.cfg_id]
        access_nodes: dict[str, dict[dace.SDFGState, tuple[set[dace_nodes.AccessNode], ...]]] = (
            pipeline_results["FindAccessNodes"][sdfg.cfg_id]
        )

        removed: set[str] = set()
        for candidate in self._find_candidates(sdfg, reachable, access_nodes):
            self._eliminate(sdfg, *candidate)
            removed.add(candidate[0])
        return removed or None

    def _find_candidates(
        self,
        sdfg: dace.SDFG,
        reachable: dict[dace.SDFGState, set[dace.SDFGState]],
        access_nodes: dict[str, dict[dace.SDFGState, tuple[set[dace_nodes.AccessNode], ...]]],
    ) -> list[tuple[str, dace.sdfg.graph.MultiConnectorEdge, dace.SDFGState]]:
        """Scans `sdfg` for write back buffers that can be eliminated.

        A candidate is a triple `(tmp_name, wb_edge, wb_state)`, where `tmp_name` is
        the transient `T`, `wb_edge` is the edge that copies `T` into the global `G`
        and `wb_state` is the state that contains `wb_edge`. `_eliminate()` may be
        called on every returned candidate.

        Args:
            sdfg: The SDFG to scan.
            reachable: Result of the `StateReachability` pass for `sdfg`.
            access_nodes: Result of the `FindAccessNodes` pass for `sdfg`.
        """
        if _has_view(sdfg):
            return []

        candidates = []
        for tmp_name, tmp_desc in sdfg.arrays.items():
            # Only a transient can be removed; a global outlives the SDFG.
            if not tmp_desc.transient:
                continue
            # ADR-18 rule 4 allows an interstate edge to read a Scalar, and we only
            #  follow dataflow edges, so we would miss such a read.
            if isinstance(tmp_desc, dace_data.Scalar):
                continue

            tmp_access = access_nodes.get(tmp_name, {})
            # Find all locations where `T` is written to global data, i.e. the write
            #  backs. The other out edges of `T` lead to consumers, which
            #  `_eliminate()` reroutes to `G` just like the write back itself.
            write_backs = [
                (edge, state)
                for state, (tmp_reads, _) in tmp_access.items()
                for node in tmp_reads
                for edge in state.out_edges(node)
                if isinstance(edge.dst, dace_nodes.AccessNode) and not edge.dst.desc(sdfg).transient
            ]
            # With several write backs there is no single `G` to rewrite `T` into.
            # TODO(havogt): Copying `T` back into more than one global could be
            #   handled by keeping all write backs but the one that is rewritten.
            if len(write_backs) != 1:
                continue
            wb_edge, wb_state = write_backs[0]

            src_subset = wb_edge.data.get_src_subset(wb_edge, wb_state)
            dst_subset = wb_edge.data.get_dst_subset(wb_edge, wb_state)
            # Without both sides the shift that the rewrite has to apply is unknown.
            if src_subset is None or dst_subset is None:
                continue
            # `T` must be copied in full, see the class documentation.
            if src_subset != dace_subsets.Range.from_array(tmp_desc):
                continue
            # We map the index space of `T` onto the one of `G` by adding an offset,
            #  which is a simplification: it needs the two to have the same shape,
            #  and comparing the sizes also rejects a `G` with a different number of
            #  dimensions.
            if dst_subset.size() != src_subset.size():
                continue
            # `Range.size()` ignores the step, so the check above also passes for a
            #  strided destination, but we only shift the accesses and can not
            #  scatter them.
            if any(step != 1 for _, _, step in dst_subset.ndrange()):
                continue
            # A write back with conflict resolution (`wcr`) combines `T` with the old
            #  value of `G`, while the rewritten producer overwrites `G`.
            if wb_edge.data.wcr is not None:
                continue

            glob_node = wb_edge.dst
            # ADR-18 rule 7 recommends a single incoming edge for a write access
            #  node; we require it, so that removing `wb_edge` leaves a node that
            #  only reads `G` and can replace the AccessNodes of `T`.
            # TODO(havogt): This stands in for a test whether rerouting the other
            #   incoming edges through the consumers of `glob_node` would create a
            #   cycle. With that test the requirement can be dropped.
            if wb_state.in_degree(glob_node) != 1:
                continue

            def_states = {
                state
                for state, (_, tmp_writes) in tmp_access.items()
                if any(_is_written(state, node) for node in tmp_writes)
            }
            # Without a producer there is nothing that could write `G` in its place.
            if not def_states:
                continue
            # The reordering below is justified by the order of the states alone. If
            #  the write back sits in a state that also defines `T`, their order is a
            #  property of the dataflow inside that state, which we do not analyse.
            if wb_state in def_states:
                continue
            # `_eliminate()` moves the write into `G` to the states that define `T`,
            #  which is only a valid reordering if they run before the write back.
            if any(not _runs_strictly_before(state, wb_state, reachable) for state in def_states):
                continue

            if self._has_conflicting_global_access(
                access_nodes.get(glob_node.data, {}),
                glob_node=glob_node,
                wb_state=wb_state,
                wb_region=dst_subset,
                wb_is_shifted=dst_subset.min_element() != src_subset.min_element(),
                def_states=def_states,
                tmp_name=tmp_name,
                reachable=reachable,
            ):
                continue

            candidates.append((tmp_name, wb_edge, wb_state))
        return candidates

    def _has_conflicting_global_access(
        self,
        glob_access: dict[dace.SDFGState, tuple[set[dace_nodes.AccessNode], ...]],
        glob_node: dace_nodes.AccessNode,
        wb_state: dace.SDFGState,
        wb_region: dace_subsets.Subset,
        wb_is_shifted: bool,
        def_states: set[dace.SDFGState],
        tmp_name: str,
        reachable: dict[dace.SDFGState, set[dace.SDFGState]],
    ) -> bool:
        """Checks if an access to `G` other than the write back prevents the rewrite.

        After the rewrite `wb_region` holds what the producer computed from the
        definition of `T` onwards instead of from the write back onwards. An access to
        `G` through an AccessNode other than `glob_node` is harmless if
        - it does not touch `wb_region`,
        - it runs strictly before every state that defines `T`, where nothing has
            changed yet, or
        - it only reads and runs strictly after the write back, where both versions of
            the SDFG agree on the content of `wb_region`.

        What is left is rejected:
        - A write to `wb_region`. For global memory ADR-18 rule 3 takes precedence
            over rule 6, so a second AccessNode writing `G` is allowed and may sit on
            the same path as `glob_node` or on a sibling one. Preponing the write is
            wrong either way: it would end up behind that write instead of in front
            of it, or put a write of `G` on a path that had none.
        - A read of `wb_region` in a state that defines `T`. Only the elementwise read
            that ADR-18 rule 3 allows is admissible, see `_is_read_by_tmp_producer()`
            and `assume_pointwise`.
        - Anything else is unordered with the write that the rewrite prepones and
            would race with it.

        Todo:
            A read inside a state that defines `T` and that is ordered before the
            producer, rather than feeding it, is safe as well and is rejected here.
        """
        for state, (glob_reads, glob_writes) in glob_access.items():
            # Every node is classified on its own, so the iteration order of the
            #  union of the two sets does not matter.
            for node in glob_reads | glob_writes:
                if node is glob_node:
                    continue
                if all(
                    _runs_strictly_before(state, def_state, reachable) for def_state in def_states
                ):
                    continue

                if node in glob_writes and _accesses_region(state, node, wb_region, incoming=True):
                    return True
                if not (
                    node in glob_reads and _accesses_region(state, node, wb_region, incoming=False)
                ):
                    continue

                if _runs_strictly_before(wb_state, state, reachable):
                    continue
                if state not in def_states:
                    return True
                if not self.assume_pointwise:
                    return True
                # ADR-18 rule 3 is about the very same memory. With a shifted write
                #  back the producer reads `G[i]` and writes `G[i + off]`, so the
                #  iterations clobber each other's input.
                if wb_is_shifted:
                    return True
                if not self._is_read_by_tmp_producer(state, node, tmp_name):
                    return True
        return False

    def _is_read_by_tmp_producer(
        self,
        state: dace.SDFGState,
        glob_read_node: dace_nodes.AccessNode,
        tmp_name: str,
    ) -> bool:
        """Checks if the read of `G` is ordered before every write to `T` in `state`.

        The read happens at the consumers of `glob_read_node`. If every AccessNode of
        `T` that is written in `state` is downstream of every one of them, then the
        write to `T`, which the rewrite turns into a write to `G`, can not start
        before the read has finished, the Map scope that hosts both excepted. That
        exception is the elementwise case of ADR-18 rule 3, which `assume_pointwise`
        covers; the pass does not distinguish it from the strictly ordered case, so it
        demands the assumption for both.

        Args:
            state: The state that contains `glob_read_node`; it also defines `T`.
            glob_read_node: The AccessNode of `G` that is read.
            tmp_name: The name of `T`.
        """
        tmp_defs = {
            node
            for node in state.data_nodes()
            if node.data == tmp_name and _is_written(state, node)
        }
        if not tmp_defs:
            return False
        return all(
            tmp_defs.issubset(state.bfs_nodes(oedge.dst))
            for oedge in state.out_edges(glob_read_node)
        )

    def _eliminate(
        self,
        sdfg: dace.SDFG,
        tmp_name: str,
        wb_edge: dace.sdfg.graph.MultiConnectorEdge,
        wb_state: dace.SDFGState,
    ) -> None:
        """Replaces every access to `T` with an access to `G` and removes `T`.

        Args:
            sdfg: The SDFG on which we operate.
            tmp_name: The name of `T`.
            wb_edge: The edge that copies `T` into `G`.
            wb_state: The state that contains `wb_edge`.
        """
        glob_name = wb_edge.dst.data
        src_subset = wb_edge.data.get_src_subset(wb_edge, wb_state)
        dst_subset = wb_edge.data.get_dst_subset(wb_edge, wb_state)
        ss_offset = [
            dst_start - src_start
            for dst_start, src_start in zip(dst_subset.min_element(), src_subset.min_element())
        ]

        wb_state.remove_edge(wb_edge)

        for state in sdfg.states():
            tmp_nodes = [node for node in state.data_nodes() if node.data == tmp_name]
            for tmp_node in tmp_nodes:
                new_node = self._replacement_node(
                    state, glob_name, writes_glob=_is_written(state, tmp_node)
                )
                reconfigured_neighbours: set[tuple[dace_nodes.Node, Optional[str]]] = set()

                for is_producer_edge, old_edges in [
                    (True, state.in_edges(tmp_node)),
                    (False, state.out_edges(tmp_node)),
                ]:
                    for old_edge in list(old_edges):
                        if old_edge.data.is_empty():
                            # An empty Memlet only imposes an order and has nothing
                            #  to shift, and `reroute_edge()` would substitute the full
                            #  range of `T` for the missing subset and make it non
                            #  empty. All edges of `tmp_node` end up on `new_node`, so
                            #  recreating it as it is preserves the order it imposes.
                            #  copy.
                            if is_producer_edge:
                                state.add_edge(
                                    old_edge.src, old_edge.src_conn, new_node, None, dace.Memlet()
                                )
                            else:
                                state.add_edge(
                                    new_node, None, old_edge.dst, old_edge.dst_conn, dace.Memlet()
                                )
                            state.remove_edge(old_edge)
                            continue

                        new_edge = gtx_transformations.utils.reroute_edge(
                            is_producer_edge=is_producer_edge,
                            current_edge=old_edge,
                            ss_offset=ss_offset,
                            state=state,
                            sdfg=sdfg,
                            old_node=tmp_node,
                            new_node=new_node,
                        )
                        neighbour = (
                            (old_edge.src, old_edge.src_conn)
                            if is_producer_edge
                            else (old_edge.dst, old_edge.dst_conn)
                        )
                        if neighbour not in reconfigured_neighbours:
                            reconfigured_neighbours.add(neighbour)
                            gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                                is_producer_edge=is_producer_edge,
                                new_edge=new_edge,
                                ss_offset=ss_offset,
                                state=state,
                                sdfg=sdfg,
                                old_node=tmp_node,
                                new_node=new_node,
                            )
                        state.remove_edge(old_edge)

                state.remove_node(tmp_node)
                if state.degree(new_node) == 0:
                    state.remove_node(new_node)

        sdfg.remove_data(tmp_name, validate=dace.Config.get_bool("debugprint"))

        # The accesses now refer to `G`, whose strides generally differ from the ones of
        #  the `T` they were derived from, so every descriptor inside a NestedSDFG that
        #  was mapped from `T` has to be updated.
        gtx_transformations.gt_propagate_strides_of(sdfg, glob_name)

    def _replacement_node(
        self,
        state: dace.SDFGState,
        glob_name: str,
        writes_glob: bool,
    ) -> dace_nodes.AccessNode:
        """Returns the AccessNode of `G` that takes the place of an AccessNode of `T`.

        ADR-18 rule 8 allows a single AccessNode per data in a state and rule 3 makes
        the exception that `G` may have a second one when it is used as input and as
        output at the same time. A node that will be written therefore gets its own
        AccessNode, reusing one that is read would make the state cyclic, while a node
        that is only read joins the AccessNode that already reads `G`, if there is one.
        """
        if not writes_glob:
            scope_dict = state.scope_dict()
            for node in state.data_nodes():
                # The AccessNodes of `T` are on the top level of the state, ADR-18
                #  rule 10, so the node that replaces them has to be there too.
                if (
                    node.data == glob_name
                    and state.in_degree(node) == 0
                    and scope_dict[node] is None
                ):
                    return node
        return state.add_access(glob_name)
