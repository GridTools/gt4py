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

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


@dace_properties.make_properties
class GT4PyWriteBackBufferElimination(dace_transformation.Pass):
    """Removes a write back buffer whose content is also consumed.

    Matches a transient `T` that is written by maps, copied in full into a non
    transient `G`, and additionally read by other consumers. Neither
    `DistributedBufferRelocator` (requires `out_degree(T) == 1`) nor
    `GT4PyMapBufferElimination` (requires the copy to be in the state where `T` is
    written, and `T` to be unused downstream) applies then, so the full array copy
    survives as a device to device transfer.

    All accesses to `T` are rewritten to `G`, shifted by the offset of the copy, and
    the copy is removed.

    `T` must be copied in full into `G`. Otherwise the producer would write `G` outside the
    range that the copy would have touched, which is not allowed because the range
    of `G` outside the copy is not part of the requested write.

    Because the rewritten producer writes `G` where previously only `T` was written,
    `G` must still hold its original value everywhere between the definition of `T`
    and the write back. Therefore `G` must neither be written nor read in that range,
    the only exception being the pointwise read described by ADR-18 rule 3, which is
    covered by `assume_pointwise`.
    """

    assume_pointwise = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="Assume that reads of `G` in the state that defines `T` are pointwise.",
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
        return dace_ppl.Modifies.Memlets | dace_ppl.Modifies.AccessNodes

    def should_reapply(self, modified: dace_ppl.Modifies) -> bool:
        return modified & (dace_ppl.Modifies.Memlets | dace_ppl.Modifies.AccessNodes)

    def depends_on(self) -> list[type[dace_transformation.Pass]]:
        return [dace_transformation.passes.StateReachability]

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: dict[str, Any]) -> Optional[set[str]]:
        reachable: dict[dace.SDFGState, set[dace.SDFGState]] = pipeline_results[
            "StateReachability"
        ][sdfg.cfg_id]

        removed: set[str] = set()
        for candidate in self._find_candidates(sdfg, reachable):
            self._eliminate(sdfg, *candidate)
            removed.add(candidate[0])
        return removed or None

    def _find_candidates(
        self,
        sdfg: dace.SDFG,
        reachable: dict[dace.SDFGState, set[dace.SDFGState]],
    ) -> list[tuple[str, dace.sdfg.graph.MultiConnectorEdge, dace.SDFGState]]:
        """Scans `sdfg` for write back buffers that can be eliminated.

        A candidate is a triple `(tmp_name, wb_edge, wb_state)`, where `tmp_name` is
        the transient `T`, `wb_edge` is the edge that copies `T` into the global `G`
        and `wb_state` is the state that contains `wb_edge`. For every candidate the
        function guarantees that:
        - `T` is a non view, non scalar transient and all its AccessNodes are on the
            top level of their state, i.e. none of them is inside a Map.
        - `wb_edge` is the only edge that connects `T` with a non transient array and
            it copies `T` in full, into a range of `G` of the same size. `G` is not a
            view and has the same number of dimensions as `T`.
        - `wb_state` neither writes `T` nor reads `G`, and the AccessNode of `G` in
            `wb_state` is only used by `wb_edge`.
        - Every state that writes `T` reaches `wb_state`, i.e. the copy always happens
            after `T` has been defined.
        - No other access to `G` conflicts with turning the writes to `T` into writes
            to `G`, see `_has_conflicting_global_access()`.

        Args:
            sdfg: The SDFG to scan.
            reachable: Result of the `StateReachability` pass for `sdfg`.
        """
        candidates = []
        for tmp_name, tmp_desc in sdfg.arrays.items():
            if not tmp_desc.transient:
                continue
            if isinstance(tmp_desc, dace_data.Scalar):
                continue
            if gtx_transformations.utils.is_view(tmp_desc, sdfg):
                continue

            locations = [
                (node, state)
                for state in sdfg.states()
                for node in state.data_nodes()
                if node.data == tmp_name
            ]
            if not locations:
                continue
            if any(state.scope_dict()[node] is not None for node, state in locations):
                continue

            write_backs = [
                (edge, state)
                for node, state in locations
                for edge in state.out_edges(node)
                if isinstance(edge.dst, dace_nodes.AccessNode) and not edge.dst.desc(sdfg).transient
            ]
            if len(write_backs) != 1:
                continue
            wb_edge, wb_state = write_backs[0]

            glob_node = wb_edge.dst
            glob_desc = glob_node.desc(sdfg)
            if gtx_transformations.utils.is_view(glob_desc, sdfg):
                continue
            if len(glob_desc.shape) != len(tmp_desc.shape):
                continue
            if wb_state.scope_dict()[glob_node] is not None:
                continue
            if wb_state.in_degree(glob_node) != 1 or wb_state.out_degree(glob_node) != 0:
                continue

            src_subset = wb_edge.data.get_src_subset(wb_edge, wb_state)
            dst_subset = wb_edge.data.get_dst_subset(wb_edge, wb_state)
            if src_subset is None or dst_subset is None:
                continue
            # `T` must be copied in full, see the class documentation.
            if src_subset != dace_subsets.Range.from_array(tmp_desc):
                continue
            if dst_subset.size() != src_subset.size():
                continue

            def_states = {state for node, state in locations if state.in_degree(node) > 0}
            if not def_states:
                continue
            if any(
                state is not wb_state and wb_state not in reachable[state] for state in def_states
            ):
                continue
            if wb_state in def_states:
                continue

            if self._has_conflicting_global_access(
                sdfg, reachable, glob_node, wb_state, def_states, tmp_name
            ):
                continue

            candidates.append((tmp_name, wb_edge, wb_state))
        return candidates

    def _has_conflicting_global_access(
        self,
        sdfg: dace.SDFG,
        reachable: dict[dace.SDFGState, set[dace.SDFGState]],
        glob_node: dace_nodes.AccessNode,
        wb_state: dace.SDFGState,
        def_states: set[dace.SDFGState],
        tmp_name: str,
    ) -> bool:
        """Checks if any AccessNode of `G` other than `glob_node` prevents the removal.

        Eliminating the write back turns the writes to `T` into writes to `G`, so `G`
        no longer holds its old value from the definition of `T` onwards, instead of
        only from the write back onwards. Any of the following disqualifies the
        candidate:
        - A write to `G`, i.e. an AccessNode with a non zero in degree. ADR-18 rule 6
            would then no longer hold, as `G` would be written in two places, and
            depending on the order the two writes would clobber each other.
        - A read of `G` in a state that defines `T`. Such a read is unordered with
            respect to the producer of `T`, so after the rewrite it would race with
            the write to `G`. The only exception is the elementwise read allowed by
            ADR-18 rule 3, which requires `assume_pointwise`, see
            `_only_feeds_tmp_producer()`.
        - A read of `G` in a state that is not reachable from `wb_state`. Such a read
            expects the old value of `G`, which the rewrite destroys.

        Returns:
            `True` if the write back must be kept.

        Todo:
            The check is conservative: once the write back has happened `G` may be
            written freely, so writes in states reachable from `wb_state` are
            harmless. Lifting the restriction requires taking the topology of the
            state machine into account.
        """
        for state in sdfg.states():
            for node in state.data_nodes():
                if node.data != glob_node.data or node is glob_node:
                    continue
                if state.in_degree(node) != 0:
                    return True
                if state in def_states:
                    if not self._only_feeds_tmp_producer(state, node, tmp_name):
                        return True
                    if not self.assume_pointwise:
                        return True
                elif state not in reachable[wb_state]:
                    return True
        return False

    def _only_feeds_tmp_producer(
        self,
        state: dace.SDFGState,
        glob_read_node: dace_nodes.AccessNode,
        tmp_name: str,
    ) -> bool:
        """Checks if the read of `G` is the elementwise read allowed by ADR-18 rule 3.

        ADR-18 rule 3 allows using the same global memory as input and output, but
        only if the output depends elementwise on the input. In the SDFG this shows
        up as `G` being an input of the very Maps that compute `T`, which is what this
        function tests: every consumer of `glob_read_node` must be the MapEntry of a
        Map that writes into `T`.

        Anything else, for example a second, unrelated Map in the same state that also
        reads `G`, is unordered with respect to the producer of `T`. After the write
        back has been removed the producer writes `G` directly, so such a reader would
        race with it and `assume_pointwise` must not waive it.

        Args:
            state: The state that contains `glob_read_node`; it also defines `T`.
            glob_read_node: The AccessNode of `G` that is read.
            tmp_name: The name of `T`.

        Returns:
            `True` if the read is elementwise in the sense of ADR-18 rule 3.
        """
        producer_entries = {
            state.entry_node(iedge.src)
            for tmp_node in state.data_nodes()
            if tmp_node.data == tmp_name
            for iedge in state.in_edges(tmp_node)
            if isinstance(iedge.src, dace_nodes.MapExit)
        }
        if not producer_entries:
            return False
        return all(oedge.dst in producer_entries for oedge in state.out_edges(glob_read_node))

    def _eliminate(
        self,
        sdfg: dace.SDFG,
        tmp_name: str,
        wb_edge: dace.sdfg.graph.MultiConnectorEdge,
        wb_state: dace.SDFGState,
    ) -> None:
        glob_node = wb_edge.dst
        glob_name = glob_node.data
        src_subset = wb_edge.data.get_src_subset(wb_edge, wb_state)
        dst_subset = wb_edge.data.get_dst_subset(wb_edge, wb_state)
        correcting_offset = dst_subset.offset_new(src_subset, negative=True)

        wb_state.remove_edge(wb_edge)

        for state in sdfg.states():
            for node in [node for node in state.data_nodes() if node.data == tmp_name]:
                if state.degree(node) == 0:
                    state.remove_node(node)
            for edge in state.edges():
                touches_tmp = (
                    isinstance(edge.src, dace_nodes.AccessNode) and edge.src.data == tmp_name
                ) or (isinstance(edge.dst, dace_nodes.AccessNode) and edge.dst.data == tmp_name)
                if edge.data.data == tmp_name:
                    edge.data.data = glob_name
                    if edge.data.subset is not None:
                        edge.data.subset.offset(correcting_offset, negative=False)
                elif touches_tmp and edge.data.other_subset is not None:
                    edge.data.other_subset.offset(correcting_offset, negative=False)
            for node in state.data_nodes():
                if node.data == tmp_name:
                    node.data = glob_name

        if wb_state.degree(glob_node) == 0:
            wb_state.remove_node(glob_node)

        sdfg.remove_data(tmp_name, validate=True)

        # The accesses now refer to `G`, whose strides generally differ from the ones
        #  of the (contiguous) `T` they were derived from, so every descriptor inside
        #  a NestedSDFG that was mapped from `T` has to be updated.
        gtx_transformations.gt_propagate_strides_of(sdfg, glob_name)
