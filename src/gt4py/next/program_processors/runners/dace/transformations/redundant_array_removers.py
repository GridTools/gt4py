# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Literal, Optional, Sequence, TypeAlias, Union, overload

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    subsets as dace_sbs,
    symbolic as dace_sym,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes
from dace.transformation import pass_pipeline as dace_ppl
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.transformations import spliting_tools as gtx_st


def gt_multi_state_global_self_copy_elimination(
    sdfg: dace.SDFG,
    validate: bool = False,
) -> Optional[dict[dace.SDFG, set[str]]]:
    """Runs `MultiStateGlobalSelfCopyElimination` on the SDFG recursively.

    For the return value see `MultiStateGlobalSelfCopyElimination.apply_pass()`.
    """
    transforms = [
        gtx_transformations.MultiStateGlobalSelfCopyElimination(),
        gtx_transformations.MultiStateGlobalSelfCopyElimination2(),
    ]

    pipeline = dace_ppl.Pipeline(transforms)
    pip_res = pipeline.apply_pass(sdfg, {})

    if validate:
        sdfg.validate()

    res: dict[dace.SDFG, set[str]] = {}
    for trans in transforms:
        tname = trans.__class__.__name__
        if tname in pip_res:
            for nsdfg, processed_data in pip_res[tname].items():
                res.setdefault(nsdfg, set()).update(processed_data)
    return res if res else None


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
        - Merge with the `MultiStateGlobalSelfCopyElimination2`.
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

    This transformation matches the following case `(G) -> (T) -> (G)`, i.e. parts
    of `G` are copied into `T` and then back into `G`.
    According to ADR-18 we have the guarantee that the data is "pointwise", i.e.
    that the data is not copied to another location, as the NumPy instruction
    `A[2:10] = A[0:8]` would do, as this is invalid to do with a Memlet.

    In the simplest case the function will eliminate the redundant copy. However,
    the transformation is also able to handle cases where data is written
    into `T` and both `G` nodes. In that case the function checks if the
    three nodes act as one and will merge them together.

    However, the transformation will only apply if `T` is not used anywhere else.

    Todo:
        There are two interesting and related cases that should also be handled.
        - If `T` is accessed downstream, then one could replace `T` with `G`
            as they are the same (in some cases).
        - If the pattern was found in two parallel branches, it will not apply.
            This should be fixed.
    """

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and be passed at construction time. Needed until
    #  [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: Optional[dict[dace.SDFG, set[str]]]

    node_g1 = dace_transformation.PatternNode(dace_nodes.AccessNode)
    node_tmp = dace_transformation.transformation.PatternNode(dace_nodes.AccessNode)
    node_g2 = dace_transformation.PatternNode(dace_nodes.AccessNode)

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.node_g1, cls.node_tmp, cls.node_g2)]

    def __init__(
        self,
        *args: Any,
        single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._single_use_data = None
        if single_use_data is not None:
            self._single_use_data = single_use_data

    def can_be_applied(
        self,
        graph: dace.SDFGState | dace.SDFG,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        node_g1 = self.node_g1
        node_g2 = self.node_g2
        node_tmp = self.node_tmp
        g_desc = node_g1.desc(sdfg)
        tmp_desc = node_tmp.desc(sdfg)

        if node_g1.data != node_g2.data:
            return False
        assert not g_desc.transient
        if not tmp_desc.transient:
            return False
        if graph.scope_dict()[node_g1] is not None:
            return False
        if gtx_transformations.utils.is_view(tmp_desc, sdfg):
            return False
        if gtx_transformations.utils.is_view(g_desc, sdfg):
            return False

        # If there is an empty Memlet connected to any of the tree nodes we give
        #  up. We might be able to handle them, but it is too complicated.
        # TODO(phimuell): Figuring out if we have to check the reads more as this one.
        #   I am pretty sure that this is enough, because the reads to `g1` would
        #   be concurrent, thus there is no guarantee if the path `g1 -> tmp -> g2`
        #   is executed before or after another potential read to `g1`. Thus, I think
        #   that there is no need for further checks.
        if any(
            any(e.data.is_empty() for e in graph.out_edges(node) + graph.in_edges(node))
            for node in [node_g1, node_g2, node_tmp]
        ):
            return False

        # Now we check if all writes into G1 and G2 are disjunct, i.e. they kind of
        #  act like one big memory, thus this makes it possible to merge it.
        # TODO(phimuell): Improve such that the parts that get overwritten
        #   get filtered out.
        # TODO(phimuell): The part below does not consider the case that something
        #   is written into `G1` copied to `T` and then copied to `G2`.
        all_writes_into_g1 = set(gtx_st.describe_incoming_edges(graph, node_g1))
        all_writes_into_g2: set[gtx_st.EdgeConnectionSpec] = set(
            gtx_st.describe_incoming_edges(graph, node_g2)
        )
        if any(
            any(
                write_into_g2.subset.intersects(write_into_g1.subset)
                for write_into_g1 in all_writes_into_g1
            )
            for write_into_g2 in all_writes_into_g2
        ):
            return False

        # Now we have to check that everything is written to `tmp` is also transferred
        #  back to `G2`. However, we ignore all the writes that came from `G1`.
        #  To maximize the success rate we also merge the reads from `tmp`.
        # TODO(phimuell): Handle the case were we end up with more then one remaining
        #   subset. This either indicates that the merger was not optimal or that
        #   the copied domain is not a hypercube.
        refers_to_data = (  # noqa: E731 [lambda-assignment]
            lambda node, data: isinstance(node, dace_nodes.AccessNode) and node.data == data
        )
        non_g_writes_into_tmp = [
            desc
            for desc in gtx_st.describe_incoming_edges(graph, node_tmp)
            if not refers_to_data(gtx_st.get_other_node(desc), node_g1.data)
        ]
        data_copied_into_g2 = self._merge_write_back_subsets(
            graph, node_tmp, node_g2, for_test=True
        )
        if data_copied_into_g2 is None:
            return False

        if len(non_g_writes_into_tmp) == 0:
            # This is a special case, it means that `g1` fully defines `tmp`.
            # TODO(phimuell): Can we already stop here?
            pass
        elif not all(
            any(tmp_to_g2.covers(into_tmp.subset) for into_tmp in non_g_writes_into_tmp)
            for tmp_to_g2 in data_copied_into_g2
        ):
            return False

        # To avoid race conditions we require that no other node referring to
        #  `G` is inside this graph.
        if any(
            dnode.data == node_g1.data
            for dnode in graph.data_nodes()
            if not (dnode is node_g2 or dnode is node_g1)
        ):
            return False

        # Check if we can remove the transient.
        if not self._is_single_use_data(node_tmp, sdfg):
            return False

        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        node_g2 = self.node_g2  # Have to be kept alive!
        node_tmp = self.node_tmp

        # Now merge the nodes together such that only `g2` remains.
        self._merge_g1_tmp_g2_nodes_together(graph, sdfg)

        # In certain cases it will happen that `g2` has become isolated. In
        #  that case we have to delete it, otherwise we have to propagate
        #  the strides.
        if graph.degree(node_g2) != 0:
            gtx_transformations.gt_propagate_strides_from_access_node(
                sdfg=sdfg,
                state=graph,
                outer_node=node_g2,
            )
        else:
            graph.remove_node(node_g2)

        # Now delete the descriptor of `tmp`.
        sdfg.remove_data(node_tmp.data, validate=False)

    def _merge_g1_tmp_g2_nodes_together(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        """Merges the three nodes together, such that only `g2` remains.

        The function will redirect all read and writes from `node_tmp` and
        `node_g1` to `node_g2`. The `node_g1` and `node_tmp` AccessNodes
        will be removed but the data descriptor will not be removed and no
        stride propagation will take place.
        """
        node_g1 = self.node_g1
        node_g2 = self.node_g2
        node_tmp = self.node_tmp

        # First we relocate all writes that go into `g1`. We can also relocate the
        #  other reads from `g1` to `g2`. See the `TODO` in the `can_be_applied()`
        #  function, at the empty Memlet check.
        #  This will isolate the `g1` node and we can remove it.
        for iedge in list(state.in_edges(node_g1)):
            state.add_edge(
                iedge.src,
                iedge.src_conn,
                node_g2,
                iedge.dst_conn,
                iedge.data,
            )
            state.remove_edge(iedge)
        for oedge in list(state.out_edges(node_g1)):
            if oedge.dst is node_g2 or oedge.dst is node_tmp:
                pass
            elif oedge.dst is not node_tmp:
                state.add_edge(
                    node_g2,
                    oedge.src_conn,
                    oedge.dst,
                    oedge.dst_conn,
                    oedge.data,
                )
            state.remove_edge(oedge)
        assert state.degree(node_g1) == 0
        state.remove_node(node_g1)

        # These are all (reads and writes) that we have to relocate from `tmp` to `g2`.
        edges_to_relocate_desc = [
            desc
            for desc in gtx_st.describe_all_edges(state, node_tmp)
            if gtx_st.get_other_node(desc) is not node_g2
        ]

        # We merge the subsets to know how to readjust the subsets such that
        #  we can directly write into `g2`. There is also the possibility of
        #  non deterministic behaviour, see `_merge_write_back_subsets()`.
        subset_map = self._merge_write_back_subsets(
            state=state, node_tmp=node_tmp, node_g2=node_g2, for_test=False
        )
        if subset_map is None:
            raise RuntimeError("The subsets were merged differently in `can_be_applied()`.")

        # Now we have to relocate the reads/writes involving `tmp` to `g2`. This is
        #  a bit more complicated as the two might have different sizes and we have
        #  to reconfigure the dataflow at the other side.
        already_reconfigured_dataflow: set[tuple[dace_nodes.Node, str]] = set()
        for edge_to_relocate_desc in edges_to_relocate_desc:
            edge_subset = edge_to_relocate_desc.subset

            # Find the associated patch the edge writes or reads from and find the
            #  subset that corresponds to the subset at the `g2` node.
            associated_tmp_subset: Optional[dace_sbs.Subset] = next(
                (
                    merged_tmp_subset
                    for merged_tmp_subset in subset_map.keys()
                    if merged_tmp_subset.covers(edge_subset)
                ),
                None,
            )
            if associated_tmp_subset is None:
                raise RuntimeError("The subsets were merged differently in `can_be_applied()`.")
            associated_g2_subset = subset_map[associated_tmp_subset]

            tmp_subset_start = associated_tmp_subset.min_element()
            g2_subset_start = associated_g2_subset.min_element()
            ss_offset = [
                g2_patch_start - tmp_start
                for g2_patch_start, tmp_start in zip(g2_subset_start, tmp_subset_start)
            ]

            if gtx_st.describes_incoming_edge(edge_to_relocate_desc):
                is_producer_edge = True
                relocate_key = (edge_to_relocate_desc.edge.src, edge_to_relocate_desc.edge.src_conn)
            else:
                is_producer_edge = False
                relocate_key = (edge_to_relocate_desc.edge.dst, edge_to_relocate_desc.edge.dst_conn)

            new_edge = gtx_transformations.utils.reroute_edge(
                is_producer_edge=is_producer_edge,
                current_edge=edge_to_relocate_desc.edge,
                ss_offset=ss_offset,
                state=state,
                sdfg=sdfg,
                old_node=node_tmp,
                new_node=node_g2,
            )

            if relocate_key not in already_reconfigured_dataflow:
                already_reconfigured_dataflow.add(relocate_key)
                # Stride propagation is done outside.
                gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                    is_producer_edge=is_producer_edge,
                    new_edge=new_edge,
                    ss_offset=ss_offset,
                    state=state,
                    sdfg=sdfg,
                    old_node=node_tmp,
                    new_node=node_g2,
                )

            state.remove_edge(edge_to_relocate_desc.edge)

        # Now remove the node that referred to the temporary node, but do not remove
        #  the data descriptor yet.
        assert state.in_degree(node_tmp) == 0
        assert all(oedge.dst is node_g2 for oedge in state.out_edges(node_tmp))
        state.remove_node(node_tmp)

    @overload
    def _merge_write_back_subsets(
        self,
        state: dace.SDFGState,
        node_tmp: dace_nodes.AccessNode,
        node_g2: dace_nodes.AccessNode,
        for_test: Literal[True],
    ) -> Union[None, list[dace_sbs.Subset]]: ...

    @overload
    def _merge_write_back_subsets(
        self,
        state: dace.SDFGState,
        node_tmp: dace_nodes.AccessNode,
        node_g2: dace_nodes.AccessNode,
        for_test: Literal[False],
    ) -> Union[None, dict[dace_sbs.Subset, dace_sbs.Subset]]: ...

    def _merge_write_back_subsets(
        self,
        state: dace.SDFGState,
        node_tmp: dace_nodes.AccessNode,
        node_g2: dace_nodes.AccessNode,
        for_test: bool,
    ) -> Union[None, list[dace_sbs.Subset], dict[dace_sbs.Subset, dace_sbs.Subset]]:
        """Merges the subsets of the edges that copy data from `node_tmp` to `node_g2`.

        The main issue is that it is not enough to "just merge" the subsets
        describing the read at `tmp`. In addition one also has to make sure
        that the subsets at the destination, `g2`, can be merged in the same
        way.
        The example why this is important would be the following two Memlets
        `tmp[0:5] -> [0:5]` and `tmp[5:10] -> [6:11]`. Although they can be
        merged at the `tmp` side, they can not be merged at the destination
        side.

        The function has two modes, with are selected by setting `for_test`.
        If it is `True` then, the function will only return the merged subsets
        that read from `node_tmp`. In case it is `False` the function will
        return a `dict` that maps a merged subset at `node_tmp` to the
        corresponding merged subset at `node_g2`.
        In case it is not possible to merge, the function returns `None`.
        This should not happen if `for_test` is `False`, but it might happen.

        Note:
            - This function is currently not able to handle detect the case
                `tmp[0:5] -> [5:10]` and `tmp[6:10] -> [0:5]` however, this
                should not be important for GT4Py.
            - If the function returns `None` while `for_test` was `False`
                indicates non deterministic behaviour, probably because
                the order of the edges was different.
        """

        # NOTE: this is an ugly hack to make the function more deterministic.
        #   We order them according to their ID of the Memlet and if the
        #   Memlets have not been modified between `can_be_apply()` and
        #   `apply()` then this will ensure that the edges are obtained in
        #   the right order.
        write_back_edges = list(state.edges_between(node_tmp, node_g2))
        write_back_edges.sort(key=lambda e: id(e.data))
        assert len(write_back_edges) > 0

        if len(write_back_edges) == 1:
            return [write_back_edges[0].data.src_subset]
        assert all(
            e.data.src_subset is not None and e.data.dst_subset is not None
            for e in write_back_edges
        )

        # Now merge the subsets at the `tmp` side together.
        merged_subsets_at_tmp = gtx_st.subset_merger([e.data.src_subset for e in write_back_edges])

        subset_map: dict[dace_sbs.Subset, dace_sbs.Subset] = {}
        for merged_subset_at_tmp in merged_subsets_at_tmp:
            # Find all edges that are involved in this subset and make sure
            #  that they can be merged to one consecutive domain.
            merged_subset_at_g2 = gtx_st.subset_merger(
                [
                    e.data.dst_subset
                    for e in write_back_edges
                    if merged_subset_at_tmp.covers(e.data.src_subset)
                ]
            )
            if len(merged_subset_at_g2) != 1:
                return None
            subset_map[merged_subset_at_tmp] = merged_subset_at_g2[0]

        return merged_subsets_at_tmp if for_test else subset_map

    def _is_single_use_data(
        self,
        access_node: dace_nodes.AccessNode,
        sdfg: dace.SDFG,
    ) -> bool:
        if self._single_use_data is None:
            find_single_use_data = dace_analysis.FindSingleUseData()
            single_use_data = find_single_use_data.apply_pass(sdfg, None)
        else:
            single_use_data = self._single_use_data
        return access_node.data in single_use_data[sdfg]


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


AccessLocation: TypeAlias = tuple[dace.SDFGState, dace_nodes.AccessNode]
"""An AccessNode and the state it is located in.
"""


@dace_properties.make_properties
class MultiStateGlobalSelfCopyElimination2(dace_transformation.Pass):
    """Removes self copying across different states.

    This function is very similar to `MultiStateGlobalSelfCopyElimination2`, however,
    it is a bit more restricted, as `MultiStateGlobalSelfCopyElimination` has a much
    better way to handle some edge cases.
    The main difference is that `MultiStateGlobalSelfCopyElimination2` uses an other
    way to locate the redundant data. Instead of focusing on the on the globals this
    transformation focuses on the transients.

    Todo:
        Merge with the `MultiStateGlobalSelfCopyElimination2`.
    """

    def modifies(self) -> dace_ppl.Modifies:
        return dace_ppl.Modifies.Memlets | dace_ppl.Modifies.AccessNodes

    def should_reapply(self, modified: dace_ppl.Modifies) -> bool:
        return modified & (dace_ppl.Modifies.Memlets | dace_ppl.Modifies.AccessNodes)

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
        """Processes the SDFG in a not recursive way, it returns the set of transients
        that have been eliminated.
        """
        redundant_transients = self._find_redundant_transients(sdfg)
        if len(redundant_transients) == 0:
            return set()

        for global_data, write_locations, read_locations in redundant_transients.values():
            self._remove_transient_at_definition_point(global_data, write_locations)
            self._remove_transient_at_using_location(
                sdfg,
                global_data,
                read_locations,
            )

        return set(redundant_transients.keys())

    def _remove_transient_at_using_location(
        self,
        sdfg: dace.SDFG,
        global_data: str,
        read_locations: list[AccessLocation],
    ) -> None:
        """Removes the write back from the transient into the global.

        If the global node has become isolated it will be removed. In addition the
        function will also remove the transient data from the SDFG.
        """
        for state, transient_ac in read_locations:
            assert state.in_degree(transient_ac) == 0
            assert state.out_degree(transient_ac) == 1

            transient_neighbours = [oedge.dst for oedge in state.out_edges(transient_ac)]
            state.remove_node(transient_ac)

            for transient_neighbour in transient_neighbours:
                if state.degree(transient_neighbour) == 0:
                    state.remove_node(transient_neighbour)

            if transient_ac.data in sdfg.arrays:
                sdfg.remove_data(transient_ac.data)

    def _remove_transient_at_definition_point(
        self,
        global_data: str,
        write_locations: list[AccessLocation],
    ) -> None:
        """Clean up the transient at its definition point.

        The function removes the write from the global into the transient. The
        function will also remove any node that has become isolated.
        However, the function will not remove the transient data from the registry.
        """
        for state, transient_ac in write_locations:
            assert state.in_degree(transient_ac) == 1
            assert state.out_degree(transient_ac) == 0

            transient_neighbours = [iedge.src for iedge in state.in_edges(transient_ac)]
            state.remove_node(transient_ac)

            for transient_neighbour in transient_neighbours:
                if state.degree(transient_neighbour) == 0:
                    state.remove_node(transient_neighbour)

    def _find_redundant_transients(
        self,
        sdfg: dace.SDFG,
    ) -> dict[str, tuple[str, list[AccessLocation], list[AccessLocation]]]:
        """Find all redundant transients that can be eliminated.

        The function returns a `dict` mapping the name of the transient data to a tuple
        of length 3. The first element is the name of the global data that defines
        the transient. The second element are the locations where the transient is
        defined, i.e. written to and in the third element are the locations where the
        transient is used.
        """

        # Scan all transients and find their location.
        possible_redundant_transients: dict[
            str, tuple[list[AccessLocation], list[AccessLocation]]
        ] = {}
        for data_name, desc in sdfg.arrays.items():
            if not desc.transient:
                continue
            write_read_locations = self._find_exclusive_read_and_write_locations_of(sdfg, data_name)
            if write_read_locations is None:
                continue
            if len(write_read_locations[1]) != 0:
                possible_redundant_transients[data_name] = write_read_locations

        # Nothing was found.
        if len(possible_redundant_transients) == 0:
            return {}

        # Determine the associated global data.
        redundant_transients_and_associated_data: dict[
            str, tuple[str, list[AccessLocation], list[AccessLocation]]
        ] = {}
        involved_global_data: set[str] = set()
        for transient_data, (
            write_locations,
            read_locations,
        ) in possible_redundant_transients.items():
            associated_global_data = self._filter_candidate(
                sdfg,
                transient_data,
                write_locations,
                read_locations,
            )
            if associated_global_data is not None:
                # This is for simplifying implementation.
                assert associated_global_data not in involved_global_data
                involved_global_data.add(associated_global_data)
                redundant_transients_and_associated_data[transient_data] = (
                    associated_global_data,
                    write_locations,
                    read_locations,
                )

        return redundant_transients_and_associated_data

    def _filter_candidate(
        self,
        sdfg: dace.SDFG,
        transient_data: str,
        write_locations: list[AccessLocation],
        read_locations: list[AccessLocation],
    ) -> Union[None, str]:
        """Test if the transient can be eliminated.

        The function tests if transient data can be eliminated and be replaced by a
        global data. If this is the case the function returns the name of the data
        and if this is not possible `None`.
        """

        # Currently we require that there is exactly one location where the temporary
        #  is defined and written back. This is a simplification, that should be
        #  removed.
        if len(write_locations) != 1 and len(read_locations) != 1:
            return None

        # We actually have to check if the global data is not modified between the
        #  location where the transient is defined and where it is written back.
        #  `MultiStateGlobalSelfCopyElimination` does this in a quite sophisticated
        #  way. We do it cheap we require that the state in which the definition of
        #  the transient happens is the immediate predecessor of where it is used.
        #  We now also hope that the transient is not used somewhere else.
        for defining_state, _ in write_locations:
            # If we are in a nested CFR, we will go upward until we found some
            #  outgoing edge.
            successor_states = gtx_transformations.utils.find_successor_state(defining_state)
            if not all(
                write_back_state in successor_states for write_back_state, _ in read_locations
            ):
                return None

        # We now must look at what defines the transient. For that we look at what
        #  defines it. For that there must be global data that writes into it.
        # TODO(phimuell): To better handle `concat_where` also allows that more
        #   producers are allowed, also differently.
        # TODO(phimuell): In `concat_where` we are using `dynamic` Memlets, they should
        #   also be checked.
        global_data: Union[None, str] = None
        for state, transient_access_node in write_locations:
            for iedge in state.in_edges(transient_access_node):
                src_node = iedge.src
                if not isinstance(src_node, dace_nodes.AccessNode):
                    return None

                if src_node.desc(sdfg).transient:
                    # Non global data writes into the transient.
                    # TODO(phimuell): Lift this.
                    return None

                # TODO(phimuell): Extend this such that there could be multiple
                #   connection from the same global.
                if global_data is None:
                    global_data = src_node.data
                else:
                    return None

        # No global data defines the transient.
        if global_data is None:
            return None

        # We now check that the transient is used to write back to the global.
        #  Currently we only allow that there is a single connection.
        #  The main issue with allowing multiple connections is that we can no
        #  longer be sure that everything is written. I encountered some dead
        #  writes once.
        assert global_data is not None
        for state, transient_access_node in read_locations:
            assert state.in_degree(transient_access_node) == 0
            if state.out_degree(transient_access_node) != 1:
                return None
            if not all(
                isinstance(oedge.dst, dace_nodes.AccessNode) and oedge.dst.data == global_data
                for oedge in state.out_edges(transient_access_node)
            ):
                return None

        return global_data

    def _find_exclusive_read_and_write_locations_of(
        self,
        sdfg: dace.SDFG,
        data_name: str,
    ) -> Union[None, tuple[list[AccessLocation], list[AccessLocation]]]:
        """The function finds all locations were `data_name` is written and read.

        The function will scan the SDFG and returns all places where `data_name` is
        written and where it is read from. If there is however, a location where the
        data is read and written to in the same place then the function returns
        `None`.

        In essence this function returns the set of all possible matches the
        transformation is looking for, but further processing has to be performed.
        """
        read_locations: list[AccessLocation] = []
        write_locations: list[AccessLocation] = []

        for state in sdfg.states():
            for dnode in state.data_nodes():
                if dnode.data != data_name:
                    continue
                out_deg = state.out_degree(dnode)
                in_deg = state.in_degree(dnode)

                # This is not the pattern we are looking for.
                if out_deg > 0 and in_deg > 0:
                    return None
                elif out_deg > 0:
                    read_locations.append((state, dnode))
                else:
                    assert in_deg > 0
                    write_locations.append((state, dnode))

        return (write_locations, read_locations)
