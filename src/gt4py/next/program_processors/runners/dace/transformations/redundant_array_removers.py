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
    data as dace_data,
    properties as dace_properties,
    subsets as dace_sbs,
    symbolic as dace_sym,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes, validation as dace_validation
from dace.transformation import pass_pipeline as dace_ppl
from dace.transformation.passes import analysis as dace_analysis

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


@dace_properties.make_properties
class CopyBypasser(dace_transformation.SingleStateTransformation):
    """Bypasses a transient that is bypassed.

    NOTE:
        - It is not valid to call this transformation after the GPU transformation
            or the strides have been decided.
    """
    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)

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
        if access_node.data not in self._single_use_data[sdfg]:
            return False
        if not desc.transient:
            return False
        if gtx_transformations.utils.is_view(desc, sdfg):
            return False

        # There must be multiple producers, otherwise this transformation
        #  does not make sense.
        number_of_producers = graph.in_degree(access_node)
        if number_of_producers <= 1:
            return False

        # We also require, that every producer is distinct.
        # NOTE: That this is for simplifying the implementation.
        if number_of_producers != len({producer for producer in graph.in_edges(access_node)}):
            return False

        # To make sense there must also be different consumers.
        number_of_consumers = graph.out_degree(access_node)
        if number_of_consumers <= 1:
            return False

        # Furthermore, they must also be different distinct consumers.
        # NOTE: This is for simplifying the implementation.
        if number_of_consumers != len({consumer for consumer in graph.out_edges(access_node)}):
            return False

        # Now check if a decomposition exist.
        assignment = self._match_consumers_to_producers(graph)
        if assignment is None:
            return False
        if not self._check_inner_constraints(
                state=graph,
                sdfg=sdfg,
                assignment=assignment,
        ):
            return False

        return True


    def apply(
        self,
        graph: dace.SDFGState | dace.SDFG,
        sdfg: dace.SDFG,
    ) -> None:
        access_node: dace_nodes.AccessNode = self.access_node

        # Compute the assignment.
        assignment = self._match_consumers_to_producers(graph)
        assert assignment is not None

        for producer_edge, consumer_edges in assignment.items():
            self._reroute_read(
                    producer_edge=producer_edge,
                    consumer_edges=consumer_edges,
                    sdfg=graph,
                    state=state,
            )

        # Remove the old intermediate that is no longer needed.
        assert graph.degree(access_node) == 0
        graph.remove_node(access_node)

    def _match_consumers_to_producers(
            self,
            state: dace.SDFGState,
    ) -> dict[dace_graph.MultiConnectorEdge, set[dace_graph.MultiConnectorEdge]] | None:
        """For each incoming (writing) edge, find the edges that read the data.

        The function will go through each outgoing edge and determine which
        incoming edge write that data. If it is not possible to clearly assign
        each consumer to a producer then `None` is returned.

        No additional feasibility checks are performed.
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
            possible_producer = self._find_producer(oedge, matching.keys())
            if possible_producer is None:
                return None
            assignment[possible_producer].add(oedge)

        # At least every producer should have at least one consumer. If this is not the
        #  case then we compute something that is not needed.
        # TODO(phimuell): Figuring out what we should actually do.
        assert any(len(assigned_consumers) == 0 for assigned_consumers in assignment.values())

        return assignment

    def _find_producer(
            self,
            consumer_edge: dace_graph.MultiConnectorEdge,
            producer_edges: Sequence[dace_graph.MultiConnectorEdge],
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
                producer_edge.data.dst_subset.covers(consumer_subset)
                for producer_edge in producer_edges
        ]

        # We only allow the case that one producer covers the consumer. If we found
        #  multiple candidates then we have an invalid SDFG, because multiple
        #  producer writes to the same memory location.
        if len(possible_producers) == 0:
            return None
        elif len(possible_producers) == 1:
            return possible_producers[0]
        raise ValueError(
                f"Found an invalid SDFG, there are multiple producer for '{self.access_node.data}"
        )

    def _check_inner_constraints(
            self,
            state: dace.SDFGState,
            sdfg: dace.SDFG,
            assignment: dict[dace_graph.MultiConnectorEdge, set[dace_graph.MultiConnectorEdge]],
    ) -> bool:
        """Checks if the decomposition can be handled.

        Essentially perform the feasibility tests that were not performed by
        `_match_consumers_to_producers()`.
        """

        for producer_edge, consumer_edges in assignment.items():
            data_source = producer_edge.src

            if len(consumer_edges) == 0:
                continue

            # If the producer edge or any consumer edge has an active WCR we
            #  do not apply.
            if producer_edge.data.wcr is not None:
                return False
            if any(
                consumer_edge.data.wcr is not None for consumer_edge in consumer_edges
            ):
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
                if len(consumer_edges) == 1:
                    if not consumer_edges[0].data.src_subset.covers(producer_edge.data.dst_subset):
                        return False

            elif isinstance(data_source, dace_nodes.MapExit):
                # The source is a Map, in this case we just generate a new transient
                #  output and then perform some reconnection. However, we require that
                #  all consumer read exactly what is is written by the map. This
                #  is to ensure some tightness of the domains.
                if not all(
                        consumer_edge.data.src_subset.covers(producer_edge.data.dst_subset)
                        for consumer_edge in consumer_edges
                ):
                    return False

            else:
                return False

    def _reroute_consumer(
            self,
            producer_edge: dace_graph.MultiConnectorEdge,
            consumer_edges: set[dace_graph.MultiConnectorEdge],
            state: dace.SDFGState,
            sdfg: dace.SDFG,
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
        """
        data_producer: dace_nodes.Node = producer_edge.src

        if isinstance(data_producer, dace_nodes.MapExit):
            self._reroute_consumer_map_producer(
                    producer_edge=producer_edge,
                    consumer_edges=consumer_edges,
                    sdfg=graph,
                    state=state,
            )

        elif isinstance(data_producer, dace_nodes.AccessNode):
            self._reroute_consumer_access_node_producer(
                    producer_edge=producer_edge,
                    consumer_edges=consumer_edges,
                    sdfg=graph,
                    state=state,
            )
        else:
            raise NotImplementedError(f"Can not handle a '{data_producer.__class__.__name__}' producer.")


    def _reroute_consumer_access_node_producer(
            self,
            producer_edge: dace_graph.MultiConnectorEdge,
            consumer_edges: set[dace_graph.MultiConnectorEdge],
            state: dace.SDFGState,
            sdfg: dace.SDFG,
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
        """
        access_node: dace_nodes.AccessNode = producer_edge.dst
        data_producer: dace_nodes.AccessNode = producer_edge.src

        old_producer_read_start = producer_edge.data.src_subset.min_element()
        old_producer_write_start = producer_edge.data.dst_subset.min_element()

        reconfigured_consumer: set[tuple[dace_nodes.Node, str]] = set()
        new_consumer_edges: list[dace_graph.MultiConnectorEdge] = []
        for consumer_edge in consumer_edges:
            consumer_node: dace_nodes.Node = consumer_edge.dst
            consumer_conn = consumer_edge.dst_conn

            # Index from where the consumer should start reading from the producer
            #  directly. That data that is read has not changed, but there might be
            #  some offsets since it has gone through `self.access_node`. The size
            #  of the reads, i.e. how much is read has not changed and we take it
            #  from what is read from `self.access_node`.
            old_consumer_read_start = consumer_subset.min_element()
            consumer_read_size = consumer_subset.size()
            new_consumer_direct_read_start = [
                dace.symbolic.pystr_to_symbolic(f"({old_producer_read}) + (({old_consumer_read}) - ({old_producer_write}))", simplify=True)
                for old_producer_read, old_consumer_read, old_producer_write
                in zip(old_producer_read_start, old_consumer_read_start, old_producer_write_start, strict=True)
            ]
            new_consumer_read_subset = dace_sbs.Range(
                    [
                        (new_consumer_direct_read, dace.symbolic.pystr_to_symbolic(f"({new_consumer_direct_read}) + ({read_size})", simplify=True), 1)
                        for new_consumer_direct_read, read_size in zip(new_consumer_direct_read_start, consumer_read_size)
                    ]
            )

            # Create a new edge that reads from the producer directly and remove the
            #  old edge.
            new_consumer_edge = state.add_edge(
                    producer_edge.src,
                    producer_edge.src_conn,
                    consumer_edge.dst,
                    consumer_edge.dst_conn,
                    dace.Memlet(
                        data=data_producer.data,
                        subset=consumer_direct_read,
                        other_subset=consumer_edge.data.dst_subset,
                        dynamic=consumer_edge.data.dynamic or producer_edge.data.dynamic,
                    )
            )
            state.remove_edge(consumer_edge)
            new_consumer_edges.append( new_consumer_edge )

            # If needed reconfigure the consumers since it now involves the
            #  original producer.
            #  The stride propagation is done after all edges have been updated.
            if (consumer_node, consumer_conn) in reconfigured_consumer:
                consumer_subset_correction = [
                    dace.symbolic.pystr_to_symbolic(f"-(({old_consumer_read}) - ({old_producer_write}))", simplify=True)
                    for old_producer_read, old_consumer_read, old_producer_write
                    in zip(old_producer_read_start, old_consumer_read_start, old_producer_write_start, strict=True)
                ]
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
        for new_consumer_edge in new_consumer_edges:
            gtx_transformations.gt_map_strides_to_dst_nested_sdfg(
                outer_node=data_producer,
                edge=new_consumer_edge,
                sdfg=sdfg,
                state=state,
            )


    def _reroute_consumer_map_producer(
            self,
            producer_edge: dace_graph.MultiConnectorEdge,
            consumer_edges: set[dace_graph.MultiConnectorEdge],
            state: dace.SDFGState,
            sdfg: dace.SDFG,
    ) -> None:
        """Perform the rerouting in case where the data producer is a Map.

        This will create a new intermediate node where the results of the Map is stored.
        It will 
        """
        access_node: dace_nodes.AccessNode = producer_edge.dst
        data_producer: dace_nodes.Node = producer_edge.src
        producer_subset = producer_edge.data.dst_subset

        # Create the intermediate storage.
        tmp_shape = producer_subset.size()
        tmp_desc = access_node.desc(sdfg).clone()
        tmp_name, tmp_desc = sdfg.add_temp_transient(
                shape=tmp_shape,
                dtype=tmp_dtype,
        )
        tmp_node: dace_nodes.AccessNode = state.add_access(tmp_name)

        # Subset for the modification 
        #  Before the Map was writing _somewhere_ into `access_nodes`, but now it
        #  writes into `tmp_node` from the beginning, Thus we have to substract
        #  the old offsets from the producer region. The minus is needed because the
        #  correction is added to the subsets.
        ss_correction = [
                f"- ({old_ss_start})" for old_ss_start in producer_subset.min_element()
        ]

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
        reconfigured_consumer: set[tuple[dace_nodes.Node, str]] = set()
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
            state=graph,
            outer_node=tmp_node,
        )
