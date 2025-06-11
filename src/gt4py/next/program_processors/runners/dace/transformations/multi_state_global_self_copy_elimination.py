# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional, TypeAlias, Union

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    transformation as dace_transformation,
)
from dace.sdfg import nodes as dace_nodes
from dace.transformation import pass_pipeline as dace_ppl

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


AccessLocation: TypeAlias = tuple[dace.SDFGState, dace_nodes.AccessNode]
"""An AccessNode and the state it is located in.
"""


def gt_multi_state_global_self_copy_elimination(
    sdfg: dace.SDFG,
    validate: bool = False,
) -> Optional[dict[dace.SDFG, set[str]]]:
    """Runs `MultiStateGlobalSelfCopyElimination` on the SDFG recursively.

    For the return value see `MultiStateGlobalSelfCopyElimination.apply_pass()`.

    Note:
        The function will also run `MultiStateGlobalSelfCopyElimination2`, but the
        results are merged together.
    """
    transforms = [
        gtx_transformations.MultiStateGlobalSelfCopyElimination(),
        gtx_transformations.MultiStateGlobalSelfCopyElimination2(),
    ]

    pipeline = dace_ppl.Pipeline(transforms)
    pip_res = pipeline.apply_pass(sdfg, {})

    if validate:
        sdfg.validate()

    # Merge the results of the two transformations together.
    res: dict[dace.SDFG, set[str]] = {}
    for trans in transforms:
        tname = trans.__class__.__name__
        if tname in pip_res:
            for nsdfg, processed_data in pip_res[tname].items():
                res.setdefault(nsdfg, set()).update(processed_data)
    return res if res else None


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
class MultiStateGlobalSelfCopyElimination2(dace_transformation.Pass):
    """Removes self copying across different states.

    This function is very similar to `MultiStateGlobalSelfCopyElimination2`, however,
    it is a bit more restricted, as `MultiStateGlobalSelfCopyElimination` has a much
    better way to handle some edge cases.
    The main difference is that `MultiStateGlobalSelfCopyElimination2` uses another
    way to locate the redundant data. Instead of focusing on the globals this
    transformation focuses on the transients.

    Todo:
        Merge with the `MultiStateGlobalSelfCopyElimination`.
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
        #  way. Here we do it in a cheap way: we require that the state in which the
        #  definition of the transient happens is the immediate predecessor of where
        #  it is used. We now also hope that the transient is not used somewhere else.
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
        # TODO(phimuell): To better handle `concat_where` also allow multiple producers.
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
            consumer = next(iter(oedge.dst for oedge in state.out_edges(transient_access_node)))
            if not (isinstance(consumer, dace_nodes.AccessNode) and consumer.data == global_data):
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
