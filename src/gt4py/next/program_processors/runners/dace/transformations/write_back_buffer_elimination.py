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

    `T` must be copied in full. Otherwise the producer would write `G` outside the
    range that the copy would have touched, which is not allowed because the range
    of `G` outside the copy is not part of the requested write.
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
        """`G` must not be read where it is still expected to hold its old value."""
        for state in sdfg.states():
            for node in state.data_nodes():
                if node.data != glob_node.data or node is glob_node:
                    continue
                if state.in_degree(node) != 0:
                    return True
                if state in def_states:
                    # `assume_pointwise` only covers `G` being an input of the very
                    #  Map that computes `T`, which is what ADR-18 rule 3 is about.
                    #  Any other reader in the same state is unordered with respect
                    #  to the producer and would race with the now direct write.
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
        glob_read: dace_nodes.AccessNode,
        tmp_name: str,
    ) -> bool:
        """Checks that `glob_read` is consumed exclusively by the Maps writing `T`."""
        producer_entries = {
            state.entry_node(iedge.src)
            for tmp_node in state.data_nodes()
            if tmp_node.data == tmp_name
            for iedge in state.in_edges(tmp_node)
            if isinstance(iedge.src, dace_nodes.MapExit)
        }
        if not producer_entries:
            return False
        return all(oedge.dst in producer_entries for oedge in state.out_edges(glob_read))

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

        try:
            sdfg.remove_data(tmp_name, validate=True)
        except ValueError as e:
            if not str(e).startswith(f"Cannot remove data descriptor {tmp_name}:"):
                raise

        # The accesses now refer to `G`, whose strides generally differ from the ones
        #  of the (contiguous) `T` they were derived from, so every descriptor inside
        #  a NestedSDFG that was mapped from `T` has to be updated.
        gtx_transformations.gt_propagate_strides_of(sdfg, glob_name)
