# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    transformation as dace_transformation,
)
from dace.sdfg import nodes as dace_nodes
from dace.transformation import pass_pipeline as dace_ppl

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


def gt_multi_state_global_self_copy_elimination(
    sdfg: dace.SDFG,
    validate: bool = False,
) -> Optional[dict[dace.SDFG, set[str]]]:
    """Runs `MultiStateGlobalSelfCopyElimination` on the SDFG recursively.

    For the return value see `MultiStateGlobalSelfCopyElimination.apply_pass()`.
    """
    pipeline = dace_ppl.Pipeline([gtx_transformations.MultiStateGlobalSelfCopyElimination()])
    res = pipeline.apply_pass(sdfg, {})

    if validate:
        sdfg.validate()

    if "MultiStateGlobalSelfCopyElimination" not in res:
        return None
    return res["MultiStateGlobalSelfCopyElimination"][sdfg]


@dace_properties.make_properties
class MultiStateGlobalSelfCopyElimination(dace_transformation.Pass):
    """Removes self copying across different states.

    This transformation is very similar to `SingleStateGlobalSelfCopyElimination`, but
    addresses a slightly different case. Assume we have the pattern `(G) -> (T)`
    in one state, i.e. the global data `G` is copied into a transient. In another
    state, we have the pattern `(T) -> (G)`, i.e. the data is written back.

    If the following conditions are satisfied, this transformation will remove all
    writes to `G`:
    - The only write access to `G` happens in the `(T) -> (G)` pattern. ADR-18
        guarantees, that if `G` is used as an input and output it must be pointwise.
        Thus there is no weird shifting.

    If the only usage of `T` is to write into `G` then the transient `T` will be
    removed.

    Note that this transformation does not consider the subsets of the writes from
    `T` to `G` because ADR-18 guarantees to us, that _if_ `G` is a genuine input
    and output, then the `G` read and write subsets have the exact same range.
    If `G` is not an output then any mutating changes to `G` would be invalid.

    Todo:
        - Implement the pattern `(G) -> (T) -> (G)` which is handled currently by
            `SingleStateGlobalSelfCopyElimination`, see `_classify_candidate()` and
            `_remove_writes_to_global()` for more.
        - Make it more efficient such that the SDFG is not scanned multiple times.
    """

    def modifies(self) -> dace_ppl.Modifies:
        return dace_ppl.Modifies.Memlets | dace_ppl.Modifies.AccessNodes

    def should_reapply(self, modified: dace_ppl.Modifies) -> bool:
        return modified & (dace_ppl.Modifies.Memlets | dace_ppl.Modifies.AccessNodes)

    def depends_on(self) -> set[type[dace_transformation.Pass]]:
        return {
            dace_transformation.passes.FindAccessStates,
        }

    def apply_pass(
        self, sdfg: dace.SDFG, pipeline_results: dict[str, Any]
    ) -> Optional[dict[dace.SDFG, set[str]]]:
        """Applies the pass.

        The function will return a `dict` that contains for every SDFG, the name
        of the processed data descriptors. If a name refers to a global memory,
        then it means that all write backs, i.e. `(T) -> (G)` patterns, have
        been removed for that `G`. If the name refers to a data descriptor that no
        longer exists, then it means that the write `(G) -> (T)` was also eliminated.
        Currently there is no possibility to identify which transient name belonged
        to a global name.
        """
        assert "FindAccessStates" in pipeline_results

        result: dict[dace.SDFG, set[str]] = dict()
        for nsdfg in sdfg.all_sdfgs_recursive():
            single_level_res: set[str] = self._process_sdfg(nsdfg, pipeline_results)
            if single_level_res:
                result[nsdfg] = single_level_res

        return result if result else None

    def _process_sdfg(
        self,
        sdfg: dace.SDFG,
        pipeline_results: dict[str, Any],
    ) -> set[str]:
        """Apply the pass to a single level of an SDFG, i.e. do not handle nested SDFG.

        The return value of this function is the same as for `apply_pass()`, but
        only for the SDFG that was passed.
        """
        t_mapping = self._find_candidates(sdfg, pipeline_results)
        if len(t_mapping) == 0:
            return set()
        self._remove_writes_to_globals(sdfg, t_mapping, pipeline_results)
        removed_transients = self._remove_transient_buffers_if_possible(
            sdfg, t_mapping, pipeline_results
        )

        return removed_transients | t_mapping.keys()

    def _find_candidates(
        self,
        sdfg: dace.SDFG,
        pipeline_results: dict[str, Any],
    ) -> dict[str, set[str]]:
        """The function searches for all candidates of that must be processed.

        The function returns a `dict` that maps the name of a global memory, `G` in
        the above pattern, to the name of the buffer transient, `T` in the above
        pattern.
        """
        access_states: dict[str, set[dace.SDFGState]] = pipeline_results["FindAccessStates"][
            sdfg.cfg_id
        ]
        global_data = [
            aname
            for aname, desc in sdfg.arrays.items()
            if not desc.transient
            and isinstance(desc, dace_data.Array)
            and not isinstance(desc, dace_data.View)
        ]

        candidates: dict[str, set[str]] = dict()
        for gname in global_data:
            candidate_tnames = self._classify_candidate(sdfg, gname, access_states)
            if candidate_tnames is not None:
                assert len(candidate_tnames) > 0
                candidates[gname] = candidate_tnames

        return candidates

    def _classify_candidate(
        self,
        sdfg: dace.SDFG,
        gname: str,
        access_states: dict[str, set[dace.SDFGState]],
    ) -> Optional[set[str]]:
        """The function tests if the global data `gname` can be handled.

        It essentially checks all conditions above, which is that the global is
        only written through transients that are fully defined by the data itself.
        writes to it are through transients that are fully defined by the data
        itself.

        The function returns `None` if `gname` can not be handled by the function.
        If `gname` can be handled the function returns a set of all data descriptors
        that act as distributed buffers.
        """
        # The set of access nodes that reads from the global, i.e. `gname`, essentially
        #  the set of candidates of `T` defined through the way it is defined.
        #  And the same set, but this time defined through who writes into the global.
        reads_from_g: set[str] = set()
        writes_to_g: set[str] = set()

        # In a first step we will identify the possible `T` only from the angle of
        #  how they interact with `G`. At a later point we will look at the `T` again,
        #  because in case of branches there might be multiple definitions of `T`.
        for state in access_states[gname]:
            for dnode in state.data_nodes():
                if dnode.data != gname:
                    continue

                # Note that we allow that `G` can be written to by multiple `T` at
                #  once. However, we require that all this data, is fully defined by
                #  a read to `G` itself.
                for iedge in state.in_edges(dnode):
                    possible_t = iedge.src

                    # If `G` is a pseudo output, see definition above, then it is only
                    #  allowed that access nodes writes to them. Note, that here we
                    #  will only collect which nodes writes to `G`, if these are
                    #  valid `T`s will be checked later, after we cllected all of them.
                    if not isinstance(possible_t, dace_nodes.AccessNode):
                        return None

                    possible_t_desc = possible_t.desc(sdfg)
                    if not possible_t_desc.transient:
                        return None  # we must write into a transient.
                    if isinstance(possible_t_desc, dace_data.View):
                        return None  # The global data must be written to from an array
                    if not isinstance(possible_t_desc, dace_data.Array):
                        return None
                    writes_to_g.add(possible_t.data)

                # Let's look who reads from `g` this will contribute to the `reads_from_g` set.
                for oedge in state.out_edges(dnode):
                    possible_t = oedge.dst
                    # `T` must be an AccessNode. Note that it is not important
                    #  what also reads from `G`. We just have to find everything that
                    #  can act as `T`.
                    if not isinstance(possible_t, dace_nodes.AccessNode):
                        continue

                    # It is important that only `G` defines `T`, so it must have
                    #  an incoming degree of one, since we have SSA.
                    if state.in_degree(possible_t) != 1:
                        continue

                    # `T` must fulfil some condition, like that it is transient.
                    possible_t_desc = possible_t.desc(sdfg)
                    if not possible_t_desc.transient:
                        continue  # we must write into a transient.
                    if isinstance(possible_t_desc, dace_data.View):
                        continue  # We must write into an array and not a view.
                    if not isinstance(possible_t_desc, dace_data.Array):
                        continue

                    # Currently we do not handle the pattern `(T) -> (G) -> (T)`,
                    #  see `_remove_writes_to_global()` for more, thus we filter
                    #  this pattern here.
                    if any(
                        tnode_oedge.dst.data == gname
                        for tnode_oedge in state.out_edges(possible_t)
                        if isinstance(tnode_oedge.dst, dace_nodes.AccessNode)
                    ):
                        return None

                    # Now add the data to the list of data that reads from `G`.
                    reads_from_g.add(possible_t.data)

        if len(writes_to_g) == 0:
            return None

        # Now every write to `G` necessarily comes from an access node that was created
        #  by a direct read from `G`. We ensure this by checking that `writes_to_g` is
        #  a subset of `reads_to_g`.
        # Note that the `T` nodes might not be unique, which happens in case
        # of separate memlets for different subsets.
        #  of different subsets, are contained in `
        if not writes_to_g.issubset(reads_from_g):
            return None

        # If we have branches, it might be that different data is written to `T` depending
        #  on which branch is selected, i.e. `T = G if cond else foo(A)`. For that
        #  reason we must now check that `G` is the only data source of `T`, but this
        #  time we must do the check on `T`. Note we only have to remove the particular access node
        #  to `T` where `G` is the only data source, while we keep the other access nodes.
        #  `T`.
        for tname in list(writes_to_g):
            for state in access_states[tname]:
                for dnode in state.data_nodes():
                    if dnode.data != tname:
                        continue
                    if state.in_degree(dnode) == 0:
                        continue  # We are only interested at definitions.

                    # Now ensures that only `gname` defines `T`.
                    for iedge in state.in_edges(dnode):
                        t_def_node = iedge.src
                        if not isinstance(t_def_node, dace_nodes.AccessNode):
                            writes_to_g.discard(tname)
                            break
                        if t_def_node.data != gname:
                            writes_to_g.discard(tname)
                            break
                if tname not in writes_to_g:
                    break

        return None if len(writes_to_g) == 0 else writes_to_g

    def _remove_writes_to_globals(
        self,
        sdfg: dace.SDFG,
        t_mapping: dict[str, set[str]],
        pipeline_results: dict[str, Any],
    ) -> None:
        """Remove all writes to the global data defined through `t_mapping`.

        The function does not handle reads from the global to the transients.

        Args:
            sdfg: The SDFG on which we should process.
            t_mapping: Maps the name of the global data to the transient data.
                This set was computed by the `_find_candidates()` function.
            pipeline_results: The results of the pipeline.
        """
        access_states: dict[str, set[dace.SDFGState]] = pipeline_results["FindAccessStates"][
            sdfg.cfg_id
        ]
        for gname, tnames in t_mapping.items():
            self._remove_writes_to_global(
                sdfg=sdfg, gname=gname, tnames=tnames, access_states=access_states
            )

    def _remove_writes_to_global(
        self,
        sdfg: dace.SDFG,
        gname: str,
        tnames: set[str],
        access_states: dict[str, set[dace.SDFGState]],
    ) -> None:
        """Remove writes to the global data `gname`.

        The function is the same as `_remove_writes_to_globals()` but only processes
        one global data descriptor.
        """
        # Here we delete the `T` node that writes into `G`, this might turn the `G`
        #  node into an isolated node.
        #  It is important that this code does not handle the `(G) -> (T) -> (G)`
        #  pattern, which is difficult to handle. The issue is that by removing `(T)`,
        #  what this function does, it also removes the definition `(T)`. However,
        #  it can only do that if it ensures that `T` is not used anywhere else.
        #  This is currently handle by the `SingleStateGlobalSelfCopyElimination` pass
        #  and the classifier rejects this pattern.
        for state in access_states[gname]:
            for dnode in list(state.data_nodes()):
                if dnode.data != gname:
                    continue
                for iedge in list(state.in_edges(dnode)):
                    tnode = iedge.src
                    if not isinstance(tnode, dace_nodes.AccessNode):
                        continue
                    if tnode.data in tnames:
                        state.remove_node(tnode)

                # It might be that the `dnode` has become isolated so remove it.
                if state.degree(dnode) == 0:
                    state.remove_node(dnode)

    def _remove_transient_buffers_if_possible(
        self,
        sdfg: dace.SDFG,
        t_mapping: dict[str, set[str]],
        pipeline_results: dict[str, Any],
    ) -> set[str]:
        """Remove the transient data if it is possible, listed in `t_mapping`.

        Essentially the function will look if there is a read to any data that is
        mentioned in `tnames`. If there isn't it will remove the write to it and
        remove it from the registry.
        The function must run after `_remove_writes_to_globals()`.

        The function returns the list of transients that were eliminated.
        """
        access_states: dict[str, set[dace.SDFGState]] = pipeline_results["FindAccessStates"][
            sdfg.cfg_id
        ]
        result: set[str] = set()
        for gname, tnames in t_mapping.items():
            result.update(
                self._remove_transient_buffer_if_possible(
                    sdfg=sdfg,
                    gname=gname,
                    tnames=tnames,
                    access_states=access_states,
                )
            )
        return result

    def _remove_transient_buffer_if_possible(
        self,
        sdfg: dace.SDFG,
        gname: str,
        tnames: set[str],
        access_states: dict[str, set[dace.SDFGState]],
    ) -> set[str]:
        obsolete_ts: set[str] = set()
        for tname in tnames:
            # We can remove the (defining) write to `T` only if it is not read
            #  anywhere else.
            if self._has_read_access_for(sdfg, tname, access_states):
                continue
            # Now we look for all writes to `tname` and remove them, since there
            #  are no reads.
            for state in access_states[tname]:
                neighbourhood: set[dace_nodes.Node] = set()
                for dnode in list(state.data_nodes()):
                    if dnode.data == tname:
                        # We have to store potential sources nodes, which is `G`.
                        #  This is because the local `G` node could become isolated.
                        #  We do not need to consider the outgoing edges, because
                        #  they are reads which we have handled above.
                        for iedge in state.in_edges(dnode):
                            assert (
                                isinstance(iedge.src, dace_nodes.AccessNode)
                                and iedge.src.data == gname
                            )
                            neighbourhood.add(iedge.src)
                        state.remove_node(dnode)
                        obsolete_ts.add(dnode.data)

                # We now have to check if an node has become isolated.
                for nh_node in neighbourhood:
                    if state.degree(nh_node) == 0:
                        state.remove_node(nh_node)

        for tname in obsolete_ts:
            sdfg.remove_data(tname, validate=False)

        return obsolete_ts

    def _has_read_access_for(
        self,
        sdfg: dace.SDFG,
        dname: str,
        access_states: dict[str, set[dace.SDFGState]],
    ) -> bool:
        """Checks if there is a read access on `dname`."""
        for state in access_states[dname]:
            for dnode in state.data_nodes():
                if state.out_degree(dnode) == 0:
                    continue  # We are only interested in read accesses
                if dnode.data == dname:
                    return True
        return False


@dace_properties.make_properties
class SingleStateGlobalSelfCopyElimination(dace_transformation.SingleStateTransformation):
    """Remove global self copy.

    This transformation matches the following case `(G) -> (T) -> (G)`, i.e. `G`
    is read from and written too at the same time, however, in between is `T`
    used as a buffer. In the example above `G` is a global memory and `T` is a
    temporary. This situation is generated by the lowering if the data node is
    not needed (because the computation on it is only conditional).

    In case `G` refers to global memory rule 3 of ADR-18 guarantees that we can
    only have a point wise dependency of the output on the input.
    This transformation will remove the write into `G`, i.e. we thus only have
    `(G) -> (T)`. The read of `G` and the definition of `T`, will only be removed
    if `T` is not used downstream. If it is used `T` will be maintained.
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
        if graph.in_degree(read_g) != 0:
            return False
        if graph.out_degree(read_g) != 1:
            return False
        if graph.degree(tmp_node) != 2:
            return False
        if graph.in_degree(write_g) != 1:
            return False
        if graph.out_degree(write_g) != 0:
            return False
        if graph.scope_dict()[read_g] is not None:
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

        # We first check if `T`, the intermediate is not used downstream. In this
        #  case we can remove the read to `G` and `T` itself from the SDFG.
        #  We have to do this check before, because the matching is not fully stable.
        is_tmp_used_downstream = self._is_read_downstream(
            start_state=graph, sdfg=sdfg, data_to_look=tmp_node.data
        )

        # The write to `G` can always be removed.
        graph.remove_node(write_g)

        # Also remove the read to `G` and `T` from the SDFG if possible.
        if not is_tmp_used_downstream:
            graph.remove_node(read_g)
            graph.remove_node(tmp_node)
            # It could still be used in a parallel branch.
            try:
                sdfg.remove_data(tmp_node.data, validate=True)
            except ValueError as e:
                if not str(e).startswith(f"Cannot remove data descriptor {tmp_node.data}:"):
                    raise
