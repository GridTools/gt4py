# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The GT4Py specific simplification pass."""

import collections
import copy
import uuid
from typing import Any, Iterable, Optional, TypeAlias

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    subsets as dace_subsets,
    transformation as dace_transformation,
)
from dace.cli import progress as dace_cliprogress
from dace.sdfg import nodes as dace_nodes
from dace.transformation import (
    dataflow as dace_dataflow,
    interstate as dace_interstate,
    pass_pipeline as dace_ppl,
    passes as dace_passes,
)

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


def gt_simplify(
    sdfg: dace.SDFG,
    validate: bool = True,
    validate_all: bool = False,
    skip: Optional[Iterable[str]] = None,
) -> Optional[dict[str, Any]]:
    """Performs simplifications on the SDFG in place.

    Instead of calling `sdfg.simplify()` directly, you should use this function,
    as it is specially tuned for GridTool based SDFGs.

    This function runs the DaCe simplification pass, but the following passes are
    replaced:
    - `InlineSDFGs`: Instead `gt_inline_nested_sdfg()` will be called.
    - `FuseStates`: The normal DaCe transformation is still run, but after the DaCe
        simplify pass has ended the function will run `GT4PyStateFusion`.

    Further, the function will run the following passes in addition to DaCe simplify:
    - `SingleStateGlobalSelfCopyElimination`: Special copy pattern that in the context
        of GT4Py based SDFG behaves as a no op, i.e. `(G) -> (T) -> (G)`.
    - `SingleStateGlobalDirectSelfCopyElimination`: Special copy pattern of the form
        `(G) -> (G)` which can always be eliminated.
    - `MultiStateGlobalSelfCopyElimination`: Very similar to
        `SingleStateGlobalSelfCopyElimination`, with the exception that the write to
        `T`, i.e. `(G) -> (T)` and the write back to `G`, i.e. `(T) -> (G)` might be
        in different states.
    - `CopyChainRemover`: Which removes some chains that are introduced by the
        `concat_where` built-in function.
    - `GT4PyDeadDataflowElimination`: Run `gt_eliminate_dead_dataflow()` on the SDFG,
        which removes more dead dataflow than the native DaCe version.
    - `MapToCopy`: Called to remove some slices.

    Furthermore, by default, or if `None` is passed for `skip` the passes listed in
    `GT_SIMPLIFY_DEFAULT_SKIP_SET` will be skipped.

    Args:
        sdfg: The SDFG to optimize.
        validate: Perform validation after the pass has run.
        validate_all: Perform extensive validation.
        skip: List of simplify passes that should not be applied, defaults
            to `GT_SIMPLIFY_DEFAULT_SKIP_SET`.

    Note:
        Currently DaCe does not provide a way to inject or exchange sub passes in
        simplify. The custom inline pass is run at the beginning and the array
        elimination at the end. The whole process is run inside a loop that ensures
        that `gt_simplify()` results in a fix point.
    """
    # Ensure that `skip` is a `set`
    skip = gtx_transformations.constants.GT_SIMPLIFY_DEFAULT_SKIP_SET if skip is None else set(skip)

    result: Optional[dict[str, Any]] = None

    at_least_one_xtrans_run = True
    starvation_protection = 30
    while at_least_one_xtrans_run:
        at_least_one_xtrans_run = False

        if starvation_protection == 0:
            raise ValueError("Simplify did not converge.")
        starvation_protection -= 1

        # NOTE: See comment in `gt_inline_nested_sdfg()` for more.
        sdfg.reset_cfg_list()

        if "InlineSDFGs" not in skip:
            inline_res = gt_inline_nested_sdfg(
                sdfg=sdfg,
                multistate=True,
                permissive=False,
                validate=False,
                validate_all=validate_all,
            )
            if inline_res is not None:
                at_least_one_xtrans_run = True
                result = result or {}
                result.update(inline_res)

        simplify_res = dace_passes.SimplifyPass(
            validate=False,
            validate_all=validate_all,
            verbose=False,
            skip=(skip | {"InlineSDFGs"}),
        ).apply_pass(sdfg, {})

        if simplify_res is not None:
            at_least_one_xtrans_run = True
            result = result or {}
            result.update(simplify_res)

        # Note that it is not nice that we run the state fusion twice, but to be fully
        #  effective there are some preparatory transformations that are run in DaCe
        #  simplify. So the GT4Py transformation is more like a clean up to handle
        #  the parts DaCe is not able to do.
        if "FuseStates" not in skip:
            fuse_state_res = sdfg.apply_transformations_repeated(
                [gtx_transformations.GT4PyStateFusion],
                validate=False,
                validate_all=validate_all,
            )
            if fuse_state_res:
                at_least_one_xtrans_run = True
                result = result or {}
                if "FuseStates" not in result:
                    result["FuseStates"] = 0
                result["FuseStates"] += fuse_state_res

        if "MapToCopy" not in skip:
            find_single_use_data = dace_transformation.passes.analysis.FindSingleUseData()
            single_use_data = find_single_use_data.apply_pass(sdfg, None)
            removed_maps = sdfg.apply_transformations_once_everywhere(
                gtx_transformations.MapToCopy(
                    single_use_data=single_use_data,
                ),
                validate=False,
                validate_all=validate_all,
            )
            if removed_maps:
                at_least_one_xtrans_run = True
                result = result or {}
                if "MapToCopy" not in result:
                    result["MapToCopy"] = 0
                result["MapToCopy"] += removed_maps

        if "GT4PyDeadDataflowElimination" not in skip:
            eliminate_dead_dataflow_res = gtx_transformations.gt_eliminate_dead_dataflow(
                sdfg=sdfg,
                run_simplify=False,
                validate=False,
                validate_all=validate_all,
            )
            if eliminate_dead_dataflow_res != 0:
                at_least_one_xtrans_run = True
                result = result or {}
                if "GT4PyDeadDataflowElimination" not in result:
                    result["GT4PyDeadDataflowElimination"] = 0
                result["GT4PyDeadDataflowElimination"] += eliminate_dead_dataflow_res

        if "CopyChainRemover" not in skip:
            copy_chain_remover_result = gtx_transformations.gt_remove_copy_chain(
                sdfg=sdfg,
                validate=False,
                validate_all=validate_all,
            )
            if copy_chain_remover_result is not None:
                at_least_one_xtrans_run = True
                result = result or {}
                if "CopyChainRemover" not in result:
                    result["CopyChainRemover"] = 0
                result["CopyChainRemover"] += copy_chain_remover_result

        if "SingleStateGlobalDirectSelfCopyElimination" not in skip:
            direct_self_copy_removal_result = sdfg.apply_transformations_repeated(
                gtx_transformations.SingleStateGlobalDirectSelfCopyElimination(),
                validate=False,
                validate_all=validate_all,
            )
            if direct_self_copy_removal_result > 0:
                at_least_one_xtrans_run = True
                result = result or {}
                if "SingleStateGlobalDirectSelfCopyElimination" not in result:
                    result["SingleStateGlobalDirectSelfCopyElimination"] = 0
                result["SingleStateGlobalDirectSelfCopyElimination"] += (
                    direct_self_copy_removal_result
                )

        if "SingleStateGlobalSelfCopyElimination" not in skip:
            self_copy_removal_result = sdfg.apply_transformations_repeated(
                gtx_transformations.SingleStateGlobalSelfCopyElimination(),
                validate=False,
                validate_all=validate_all,
            )
            if self_copy_removal_result > 0:
                at_least_one_xtrans_run = True
                result = result or {}
                if "SingleStateGlobalSelfCopyElimination" not in result:
                    result["SingleStateGlobalSelfCopyElimination"] = 0
                result["SingleStateGlobalSelfCopyElimination"] += self_copy_removal_result

        if "MultiStateGlobalSelfCopyElimination" not in skip:
            distributed_self_copy_result = (
                gtx_transformations.gt_multi_state_global_self_copy_elimination(
                    sdfg, validate=validate_all
                )
            )
            if distributed_self_copy_result is not None:
                at_least_one_xtrans_run = True
                result = result or {}
                if "MultiStateGlobalSelfCopyElimination" not in result:
                    result["MultiStateGlobalSelfCopyElimination"] = set()
                result["MultiStateGlobalSelfCopyElimination"].update(distributed_self_copy_result)

    if validate:
        sdfg.validate()

    return result


def gt_inline_nested_sdfg(
    sdfg: dace.SDFG,
    multistate: bool = True,
    permissive: bool = False,
    validate: bool = True,
    validate_all: bool = False,
    progress: Optional[bool] = None,
) -> Optional[dict[str, int]]:
    """Perform inlining of nested SDFG into their parent SDFG.

    The function uses DaCe's `InlineSDFG` transformation, the same used in simplify.
    However, before the inline transformation is run the function will run some
    cleaning passes that allows inlining nested SDFGs.
    As a side effect, the function will split stages into more states.

    Args:
        sdfg: The SDFG that should be processed, will be modified in place and returned.
        multistate: Allow inlining of multistate nested SDFG, defaults to `True`.
        permissive: Be less strict on the accepted SDFGs.
        validate: Perform validation after the transformation has finished.
        validate_all: Performs extensive validation.
    """

    nb_preproccess_total = 0
    nb_inlines_total = 0
    nsdfgs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace_nodes.NestedSDFG)]
    for nsdfg_node in dace_cliprogress.optional_progressbar(
        reversed(nsdfgs), title="Inlining SDFGs", n=len(nsdfgs), progress=progress
    ):
        nsdfg: dace.SDFG = nsdfg_node.sdfg
        parent_state = nsdfg.parent
        parent_sdfg = parent_state.sdfg
        parent_state_id = parent_state.block_id

        # Clean the symbols and connectors of the nested SDFG.
        for xform in [dace_dataflow.PruneSymbols, dace_dataflow.PruneConnectors]:
            candidate = {xform.nsdfg: nsdfg_node}
            cleaner = xform()
            cleaner.setup_match(
                sdfg=parent_sdfg,
                cfg_id=parent_state.parent_graph.cfg_id,
                state_id=parent_state_id,
                subgraph=candidate,
                expr_index=0,
                override=True,
            )
            if cleaner.can_be_applied(parent_state, 0, parent_sdfg, permissive=False):
                cleaner.apply(parent_state, parent_sdfg)
                nb_preproccess_total += 1

        # NOTE: In [PR#2178](https://github.com/GridTools/gt4py/pull/2178) this function was
        #   modified to be more efficient. It also changed the order in which the inlining
        #   transformations of DaCe were applied. Instead of trying `InlineMultistateSDFG`
        #   it changed that such that `InlineSDFG` was used. However, this triggered
        #   [issue#2108](https://github.com/spcl/dace/issues/2108) which lead to the removals
        #   of some writes. As a temporary solution we no longer use `InlineSDFG` but only
        #   the multistate version.
        # TODO(phimuell): As soon as the DaCe issue is resolved start using `InlineSDFG` again.
        multi_state_candidate = {dace_interstate.InlineMultistateSDFG.nested_sdfg: nsdfg_node}
        multi_state_inliner = dace_interstate.InlineMultistateSDFG()
        multi_state_inliner.setup_match(
            sdfg=parent_sdfg,
            cfg_id=parent_state.parent_graph.cfg_id,
            state_id=parent_state_id,
            subgraph=multi_state_candidate,
            expr_index=0,
            override=True,
        )
        if multi_state_inliner.can_be_applied(parent_state, 0, parent_sdfg, permissive=permissive):
            multi_state_inliner.apply(parent_state, parent_sdfg)
            nb_inlines_total += 1

    result: dict[str, int] = {}
    if nb_inlines_total != 0:
        result["InlineSDFGs"] = nb_inlines_total
    if nb_preproccess_total != 0:
        result["PruneSymbols|PruneConnectors"] = nb_preproccess_total

    if validate or validate_all:
        sdfg.validate()

    return result if result else None


def gt_substitute_compiletime_symbols(
    sdfg: dace.SDFG,
    repl: dict[str, Any],
    simplify: bool = False,
    skip: Optional[set[str]] = None,
    validate: bool = True,
    validate_all: bool = False,
    **kwargs: Any,
) -> None:
    """Substitutes symbols that are known at compile time with their value.

    Some symbols are known to have a constant value. This function will remove these
    symbols from the SDFG and replace them with the value.
    An example where this makes sense are strides that are known to be one.

    Args:
        sdfg: The SDFG to process.
        repl: Maps the name of the symbol to the value it should be replaced with.
        simplify: If `False` do not call `gt_simplify()` after the substitution.
        skip: List of simplify stages that should not run. Is passed to the `skip` argument
            of `gt_simplify()`.
        validate: Perform validation at the end of the function.
        validate_all: Perform validation also on intermediate steps.

    Note:
        Due to [DaCe issue 1817](https://github.com/spcl/dace/issues/1817) this function
        runs `gt_simplify()` on the SDFG. It will do so with also enable some passes
        that are by default disabled.

    Todo:
        This function needs improvement.
    """

    # NOTE: If a symbol, that should be replaced with a constant, is a data, i.e. has
    #   an entry in `sdfg.arrays` _and_ an AccessNode, then constant substitution
    #   fails. Furthermore, there is [DaCe issue 1817](https://github.com/spcl/dace/issues/1817),
    #   causing problems if there are two states in a nested SDFG. To avoid them
    #   we initially call simplify and hope for the best.
    # For testing purposes we need to be able to disable this initial simplify.
    #  This is an implementation detail that we should get rid of.
    if kwargs.get("simplify_at_entry", True):
        # NOTE: To ensure uniform behaviour and as a performance optimization,
        #   `gt_auto_optimizer()` performs the initial simplification before this
        #   function is called. If something is changed here then the change might
        #   be necessary to be applied there as well.
        gtx_transformations.gt_simplify(
            sdfg=sdfg,
            skip=skip,
            validate=False,
            validate_all=validate_all,
        )

    # We will use the `replace` function of the top SDFG, however, lower levels
    #  are handled using ConstantPropagation.
    # TODO(phimuell, iomaganaris): Revisit once the replace function in DaCe has been updated.
    sdfg.replace_dict(repl)

    # TODO(phimuell): Get rid of the `ConstantPropagation`
    const_prop = dace_passes.ConstantPropagation()
    const_prop.recursive = True
    const_prop.progress = False

    const_prop.apply_pass(
        sdfg=sdfg,
        initial_symbols=repl,
        _=None,
    )
    if simplify:
        gt_simplify(
            sdfg=sdfg,
            skip=skip,
            validate=False,
            validate_all=validate_all,
        )
    dace.sdfg.propagation.propagate_memlets_sdfg(sdfg)

    if validate:
        sdfg.validate()


def gt_reduce_distributed_buffering(
    sdfg: dace.SDFG,
    validate: bool = True,
    validate_all: bool = False,
) -> Optional[dict[dace.SDFG, dict[dace.SDFGState, set[str]]]]:
    """Removes distributed write back buffers."""
    pipeline = dace_ppl.Pipeline([DistributedBufferRelocator()])
    all_result = {}

    for rsdfg in sdfg.all_sdfgs_recursive():
        ret = pipeline.apply_pass(sdfg, {})
        if ret is not None:
            all_result[rsdfg] = ret

        if validate_all:
            rsdfg.validate()

    if len(all_result) == 0:
        return None

    if validate:
        sdfg.validate()

    return all_result


AccessLocation: TypeAlias = tuple[dace_nodes.AccessNode, dace.SDFGState]
"""Describes an access node and the state in which it is located.
"""


@dace_properties.make_properties
class DistributedBufferRelocator(dace_transformation.Pass):
    """Moves the final write back of the results to where it is needed.

    In certain cases, especially in case where we have `if` the result is computed
    in each branch and then in the join state written back. Thus there is some
    additional storage needed.
    The transformation will look for the following situation:
    - A transient data container, called `temp_storage`, is written into another
        container, called `dest_storage`, which is not transient.
    - The access node of `temp_storage` has an in degree of zero and an out degree of one.
    - The access node of `dest_storage` has an in degree of of one and an
        out degree of zero (this might be lifted).
    - `temp_storage` is not used afterwards.
    - `dest_storage` is only used to implement the buffering.

    The function will relocate the writing of `dest_storage` to where `temp_storage` is
    written, which might be multiple locations.
    It will also remove the writing back.
    It is advised that after this transformation simplify is run again.

    The relocation will not take place if it might create data race. A necessary
    but not sufficient condition for a data race is if `dest_storage` is present
    in the state where `temp_storage` is defined. In addition at least one of the
    following conditions has to be met:
    - There are accesses to `dest_storage` that are not predecessor to the node where
        the data is stored inside `temp_storage`. This check will ignore empty Memlets.
    - There is a `dest_storage` access node, that has an output degree larger
        than one.

    Note:
        - Essentially this transformation removes the double buffering of
            `dest_storage`. Because we ensure that that `dest_storage` is non
            transient this is okay, as our rule guarantees this.

    Todo:
        - Completely remove `temp_storage`, i.e. write directly into `dest_storage`.
        - Allow that `dest_storage` can also be transient.
        - Allow that `dest_storage` does not need to be a sink node, this is most
            likely most relevant if it is transient.
        - Check if `dest_storage` is used between where we want to place it and
            where it is currently used.
    """

    def modifies(self) -> dace_ppl.Modifies:
        return dace_ppl.Modifies.Memlets | dace_ppl.Modifies.AccessNodes

    def should_reapply(self, modified: dace_ppl.Modifies) -> bool:
        return modified & (dace_ppl.Modifies.Memlets | dace_ppl.Modifies.AccessNodes)

    def depends_on(self) -> set[type[dace_transformation.Pass]]:
        return {
            dace_transformation.passes.StateReachability,
            dace_transformation.passes.FindAccessStates,
        }

    def apply_pass(
        self, sdfg: dace.SDFG, pipeline_results: dict[str, Any]
    ) -> Optional[dict[dace.SDFGState, set[str]]]:
        # NOTE: We can not use `AccessSets` because this pass operates on
        #  `ControlFlowBlock`s, which might consists of multiple states. Thus we are
        #  using `FindAccessStates` which has this `SDFGState` granularity. The downside
        #  is, however, that we have to determine if the access in that state is a
        #  write or not, which means we have to find it first.
        access_states: dict[str, set[dace.SDFGState]] = pipeline_results["FindAccessStates"][
            sdfg.cfg_id
        ]

        # For speeding up the `is_accessed_downstream()` calls.
        reachable: dict[dace.SDFGState, set[dace.SDFGState]] = pipeline_results[
            "StateReachability"
        ][sdfg.cfg_id]

        result: dict[dace.SDFGState, set[str]] = collections.defaultdict(set)

        to_relocate = self._find_candidates(sdfg, reachable, access_states)
        if len(to_relocate) == 0:
            return None
        self._relocate_write_backs(sdfg, to_relocate)

        for (wb_an, wb_state), _ in to_relocate:
            result[wb_state].add(wb_an.data)

        return result

    def _relocate_write_backs(
        self,
        sdfg: dace.SDFG,
        to_relocate: list[tuple[AccessLocation, list[AccessLocation]]],
    ) -> None:
        """Perform the actual relocation."""
        for (wb_an, wb_state), def_locations in to_relocate:
            # Get the memlet that we have to replicate.
            wb_edge = next(iter(wb_state.out_edges(wb_an)))
            wb_memlet: dace.Memlet = wb_edge.data
            final_dest_name: str = wb_edge.dst.data

            for def_an, def_state in def_locations:
                # TODO(phimuell): Do not create a copy from `temp_storage` to the newly
                #   created `dest_storage`. Instead bypass `temp_storage` fully.
                def_state.add_edge(
                    def_an,
                    wb_edge.src_conn,
                    def_state.add_access(final_dest_name, copy.copy(wb_edge.dst.debuginfo)),
                    wb_edge.dst_conn,
                    copy.deepcopy(wb_memlet),
                )

            # Now remove the old node and if the old target become isolated
            #  remove that as well.
            old_dst = wb_edge.dst
            wb_state.remove_node(wb_an)
            if wb_state.degree(old_dst) == 0:
                wb_state.remove_node(old_dst)

    def _find_candidates(
        self,
        sdfg: dace.SDFG,
        reachable: dict[dace.SDFGState, set[dace.SDFGState]],
        access_states: dict[str, set[dace.SDFGState]],
    ) -> list[tuple[AccessLocation, list[AccessLocation]]]:
        """Determines all temporaries that have to be relocated.

        Returns:
            A list of tuples. The first element element of the tuple is an
            `AccessLocation` that describes where the temporary is read.
            The second element is a list of `AccessLocation`s that describes
            where the temporary is defined.
        """
        # All nodes that are used as distributed buffers.
        candidate_temp_storage: list[AccessLocation] = []

        # Which `temp_storage` access node is written back to which global memory.
        temp_storage_to_global: dict[dace_nodes.AccessNode, str] = {}

        for state in sdfg.states():
            # These are the possible targets we want to write into.
            candidate_dst_nodes: set[dace_nodes.AccessNode] = {
                node
                for node in state.sink_nodes()
                if (
                    isinstance(node, dace_nodes.AccessNode)
                    and state.in_degree(node) == 1
                    and (not node.desc(sdfg).transient)
                )
            }
            if len(candidate_dst_nodes) == 0:
                continue

            for temp_storage in state.data_nodes():
                if not temp_storage.desc(sdfg).transient:
                    continue
                if state.out_degree(temp_storage) != 1:
                    continue
                dst_candidate: dace_nodes.AccessNode = next(
                    iter(edge.dst for edge in state.out_edges(temp_storage))
                )
                if dst_candidate not in candidate_dst_nodes:
                    continue
                candidate_temp_storage.append((temp_storage, state))
                temp_storage_to_global[temp_storage] = dst_candidate.data

        if len(candidate_temp_storage) == 0:
            return []

        # Now we have to find the places where the temporary sources are defined.
        #  I.e. This is also the location where the temporary source was initialized.
        result_candidates: list[tuple[AccessLocation, list[AccessLocation]]] = []

        def find_upstream_states(dst_state: dace.SDFGState) -> set[dace.SDFGState]:
            return {
                src_state
                for src_state in sdfg.states()
                if dst_state in reachable[src_state] and dst_state is not src_state
            }

        for temp_storage in candidate_temp_storage:
            temp_storage_node, temp_storage_state = temp_storage
            def_locations: list[AccessLocation] = []
            for upstream_state in find_upstream_states(temp_storage_state):
                if self._is_written_to_in_state(
                    data=temp_storage_node.data,
                    state=upstream_state,
                    access_states=access_states,
                ):
                    # NOTE: We do not impose any restriction on `temp_storage`. Thus
                    #   It could be that we do read from it (we can never write to it)
                    #   in this state or any other state later.
                    # TODO(phimuell): Should we require that `temp_storage` is a sink
                    #   node? It might prevent or allow other optimizations.
                    new_locations = [
                        (data_node, upstream_state)
                        for data_node in upstream_state.data_nodes()
                        if data_node.data == temp_storage_node.data
                    ]
                    def_locations.extend(new_locations)
            if len(def_locations) != 0:
                result_candidates.append((temp_storage, def_locations))

        # This transformation removes `temp_storage` by writing its content directly
        #  to `dest_storage`, at the point where it is defined.
        #  For this transformation to be valid the following conditions have to be met:
        #   - Between the definition of `temp_storage` and the write back to `dest_storage`,
        #       `dest_storage` can not be accessed.
        #   - Between the definitions of `temp_storage` and the point where it is written
        #       back, `temp_storage` can only be accessed in the range that is written back.
        #   - After the write back point, `temp_storage` shall not be accessed. This
        #       restriction could be lifted.
        #
        #  To keep the implementation simple, we use the conditions:
        #   - `temp_storage` is only accessed were it is defined and at the write back
        #       point.
        #   - Between the definitions of `temp_storage` and the write back point,
        #       `dest_storage` is not used.

        result: list[tuple[AccessLocation, list[AccessLocation]]] = []

        for wb_location, def_locations in result_candidates:
            # Get the state and the location where the temporary is written back
            #  into the global data container.
            wb_node, wb_state = wb_location

            for def_node, def_state in def_locations:
                # Test if `temp_storage` is only accessed where it is defined and
                #  where it is written back.
                if gtx_transformations.utils.is_accessed_downstream(
                    start_state=def_state,
                    sdfg=sdfg,
                    reachable_states=reachable,
                    data_to_look=wb_node.data,
                    nodes_to_ignore={def_node, wb_node},
                ):
                    break

                # Check if the global data is not used between the definition of
                #  `dest_storage` and where its written back. However, we ignore
                #  the state were `temp_storage` is defined. The checks if these
                #  checks are performed by the `_check_read_write_dependency()`
                #  function.
                global_data_name = temp_storage_to_global[wb_node]
                global_nodes_in_def_state = {
                    dnode for dnode in def_state.data_nodes() if dnode.data == global_data_name
                }

                # The `is_accessed_downstream()` function has some odd behaviour
                #  regarding `states_to_ignore`. Because of the special SDFGs we have
                #  this should not be an issue.
                if gtx_transformations.utils.is_accessed_downstream(
                    start_state=def_state,
                    sdfg=sdfg,
                    reachable_states=reachable,
                    data_to_look=global_data_name,
                    nodes_to_ignore=global_nodes_in_def_state,
                    states_to_ignore={wb_state},
                ):
                    break
                if self._check_read_write_dependency(sdfg, wb_location, def_locations):
                    break
            else:
                result.append((wb_location, def_locations))

        return result

    def _is_written_to_in_state(
        self,
        data: str,
        state: dace.SDFGState,
        access_states: dict[str, set[dace.SDFGState]],
    ) -> bool:
        """This function determines if there is a write to data `data` in state `state`.

        Args:
            data: Name of the data descriptor that should be tested.
            state: The state that should be examined.
            access_states: The set of state that writes to a specific data.
        """
        assert data in access_states, f"Did not found '{data}' in 'access_states'."

        # According to `access_states` `data` is not accessed inside `state`.
        #  Therefore there is no write.
        if state not in access_states[data]:
            return False

        # There is an AccessNode for `data` inside `state`. Now we have to find the
        #  node and determine if it is a write or not.
        for dnode in state.data_nodes():
            if dnode.data != data:
                continue
            if state.in_degree(dnode) > 0:
                return True

        return False

    def _check_read_write_dependency(
        self,
        sdfg: dace.SDFG,
        write_back_location: AccessLocation,
        target_locations: list[AccessLocation],
    ) -> bool:
        """Tests if read-write conflicts would be created.

        This function ensures that the substitution of `write_back_location` into
        `target_locations` will not create a read-write conflict.
        The rules that are used for this are outlined in the class description.

        Args:
            sdfg: The SDFG on which we operate.
            write_back_location: Where currently the write back occurs.
            target_locations: List of the locations where we would like to perform
                the write back instead.

        Returns:
            If a read-write dependency is detected then the function will return
            `True` and if none was detected `False` will be returned.
        """
        for target_location in target_locations:
            if self._check_read_write_dependency_impl(sdfg, write_back_location, target_location):
                return True
        return False

    def _check_read_write_dependency_impl(
        self,
        sdfg: dace.SDFG,
        write_back_location: AccessLocation,
        target_location: AccessLocation,
    ) -> bool:
        """Tests if read-write conflict would be created for a single location.

        Args:
            sdfg: The SDFG on which we operate.
            write_back_location: Where currently the write back occurs.
            target_locations: Location where the new write back should be performed.

        Todo:
            Refine these checks later.

        Returns:
            If a read-write dependency is detected then the function will return
            `True` and if none was detected `False` will be returned.
        """
        assert write_back_location[0].data == target_location[0].data

        # Get the state and the location where the temporary is written back
        #  into the global data container. Because `write_back_node` refers to
        #  the temporary we must query the graph to find the global node.
        write_back_node, write_back_state = write_back_location
        write_back_edge = next(iter(write_back_state.out_edges(write_back_node)))
        global_data_name = write_back_edge.dst.data
        assert not sdfg.arrays[global_data_name].transient
        assert write_back_state.out_degree(write_back_node) == 1
        assert write_back_state.in_degree(write_back_node) == 0

        # Get the location and the state where the temporary is originally defined.
        def_location_of_intermediate, state_to_inspect = target_location

        # These are all access nodes that refers to the global data, that we want
        #  to move into the state `state_to_inspect`. We need them to do the
        #  second test.
        accesses_to_global_data: set[dace_nodes.AccessNode] = set()

        # In the first check we look for an access node, to the global data, that
        #  has an output degree larger than one. However, for this we ignore all
        #  empty Memlets. This is done because such Memlets are used to induce a
        #  schedule or order in the dataflow graph.
        #  As a byproduct, for the second test, we also collect all of these nodes.
        # TODO(phimuell): Refine this such that it takes the location of the data
        #   into account.
        for dnode in state_to_inspect.data_nodes():
            if dnode.data != global_data_name:
                continue
            dnode_degree = sum(
                (1 for oedge in state_to_inspect.out_edges(dnode) if not oedge.data.is_empty())
            )
            if dnode_degree > 1:
                return True
            # TODO(phimuell): Maybe AccessNodes with zero input degree should be ignored.
            accesses_to_global_data.add(dnode)

        # There is no reference to the global data, so no need to do more tests.
        if len(accesses_to_global_data) == 0:
            return False

        # For the second test we will explore the dataflow graph, in reverse order,
        #  starting from the definition of the temporary node. If we find an access
        #  to the global data we remove it from the `accesses_to_global_data` list.
        #  If the list has not become empty, then we know that there is some sind
        #  branch (or concurrent dataflow) in this state that accesses the global
        #  data and we will have read-write conflicts.
        #  It is however, important to realize that passing this check does not
        #  imply that there are no read-write. We assume here that all accesses to
        #  the global data that was made before the write back were constructed in
        #  a correct way.
        to_process: list[dace_nodes.Node] = [def_location_of_intermediate]
        seen: set[dace_nodes.Node] = set()
        while len(to_process) != 0:
            node = to_process.pop()
            seen.add(node)

            if isinstance(node, dace_nodes.AccessNode):
                if node.data == global_data_name:
                    accesses_to_global_data.discard(node)
                    if len(accesses_to_global_data) == 0:
                        return False

            # Note that we only explore the ingoing edges, thus we will not necessarily
            #  explore the whole graph. However, this is fine, because we will see the
            #  relevant parts. To see that assume that we would also have to check the
            #  outgoing edges, this would mean that there was some branching point,
            #  which is a serialization point, so the dataflow would have been invalid
            #  before.
            to_process.extend(
                iedge.src for iedge in state_to_inspect.in_edges(node) if iedge.src not in seen
            )

        assert len(accesses_to_global_data) > 0
        return True


@dace_properties.make_properties
class GT4PyMoveTaskletIntoMap(dace_transformation.SingleStateTransformation):
    """Moves a Tasklet, with no input into a map.

    Tasklets without inputs, are mostly used to generate constants.
    However, if they are outside a Map, then this constant value is an
    argument to the kernel, and can not be used by the compiler.

    This transformation moves such Tasklets into a Map scope.
    """

    tasklet = dace_transformation.PatternNode(dace_nodes.Tasklet)
    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.tasklet, cls.access_node, cls.map_entry)]

    def can_be_applied(
        self,
        graph: dace.SDFGState | dace.SDFG,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        tasklet: dace_nodes.Tasklet = self.tasklet
        access_node: dace_nodes.AccessNode = self.access_node
        access_desc: dace_data.Data = access_node.desc(sdfg)
        map_entry: dace_nodes.MapEntry = self.map_entry

        if graph.in_degree(tasklet) != 0:
            return False
        if graph.out_degree(tasklet) != 1:
            return False
        if tasklet.has_side_effects(sdfg):
            return False
        if tasklet.code_init.as_string:
            return False
        if tasklet.code_exit.as_string:
            return False
        if tasklet.code_global.as_string:
            return False
        if tasklet.state_fields:
            return False
        if not isinstance(access_desc, dace_data.Scalar):
            return False
        if not access_desc.transient:
            return False
        if not any(
            edge.dst_conn and edge.dst_conn.startswith("IN_")
            for edge in graph.out_edges(access_node)
            if edge.dst is map_entry
        ):
            return False
        # NOTE: We allow that the access node is used in multiple places.

        return True

    def apply(
        self,
        graph: dace.SDFGState | dace.SDFG,
        sdfg: dace.SDFG,
    ) -> None:
        tasklet: dace_nodes.Tasklet = self.tasklet
        access_node: dace_nodes.AccessNode = self.access_node
        access_desc: dace_data.Scalar = access_node.desc(sdfg)
        map_entry: dace_nodes.MapEntry = self.map_entry

        # Find _a_ connection that leads from the access node to the map.
        edge_to_map = next(
            iter(
                edge
                for edge in graph.out_edges(access_node)
                if edge.dst is map_entry and edge.dst_conn.startswith("IN_")
            )
        )
        connector_name: str = edge_to_map.dst_conn[3:]

        # This is the tasklet that we will put inside the map, note we have to do it
        #  this way to avoid some name clash stuff.
        inner_tasklet: dace_nodes.Tasklet = graph.add_tasklet(
            name=f"{tasklet.label}__clone_{str(uuid.uuid1()).replace('-', '_')}",
            outputs=tasklet.out_connectors.keys(),
            inputs=set(),
            code=tasklet.code,
            language=tasklet.language,
            debuginfo=tasklet.debuginfo,
        )
        inner_desc: dace_data.Scalar = access_desc.clone()
        inner_data_name: str = sdfg.add_datadesc(access_node.data, inner_desc, find_new_name=True)
        inner_an: dace_nodes.AccessNode = graph.add_access(
            inner_data_name, copy.copy(access_node.debuginfo)
        )

        # Connect the tasklet with the map entry and the access node.
        graph.add_nedge(map_entry, inner_tasklet, dace.Memlet())
        graph.add_edge(
            inner_tasklet,
            next(iter(inner_tasklet.out_connectors.keys())),
            inner_an,
            None,
            dace.Memlet(f"{inner_data_name}[0]"),
        )

        # Now we will reroute the edges went through the inner map, through the
        #  inner access node instead.
        for old_inner_edge in list(
            graph.out_edges_by_connector(map_entry, "OUT_" + connector_name)
        ):
            # We now modify the downstream data. This is because we no longer refer
            #  to the data outside but the one inside.
            self._modify_downstream_memlets(
                state=graph,
                edge=old_inner_edge,
                old_data=access_node.data,
                new_data=inner_data_name,
            )

            # After we have changed the properties of the MemletTree of `edge`
            #  we will now reroute it, such that the inner access node is used.
            graph.add_edge(
                inner_an,
                None,
                old_inner_edge.dst,
                old_inner_edge.dst_conn,
                old_inner_edge.data,
            )
            graph.remove_edge(old_inner_edge)
            map_entry.remove_in_connector("IN_" + connector_name)
            map_entry.remove_out_connector("OUT_" + connector_name)

        # Now we can remove the map connection between the outer/old access
        #  node and the map.
        graph.remove_edge(edge_to_map)

        # The data is no longer referenced in this state, so we can potentially
        #  remove
        if graph.out_degree(access_node) == 0:
            # TODO(phimuell): Use the pipeline to run `StateReachability` once.
            if not gtx_transformations.utils.is_accessed_downstream(
                start_state=graph,
                sdfg=sdfg,
                reachable_states=None,
                data_to_look=access_node.data,
                nodes_to_ignore={access_node},
            ):
                graph.remove_nodes_from([tasklet, access_node])
                # Needed if data is accessed in a parallel branch.
                try:
                    sdfg.remove_data(access_node.data, validate=True)
                except ValueError as e:
                    if not str(e).startswith(f"Cannot remove data descriptor {access_node.data}:"):
                        raise

    def _modify_downstream_memlets(
        self,
        state: dace.SDFGState,
        edge: dace.sdfg.graph.MultiConnectorEdge,
        old_data: str,
        new_data: str,
    ) -> None:
        """Replaces the data along on the tree defined by `edge`.

        The function will traverse the MemletTree defined by `edge`.
        Any Memlet that refers to `old_data` will be replaced with
        `new_data`.

        Args:
            state: The sate in which we operate.
            edge: The edge defining the MemletTree.
            old_data: The name of the data that should be replaced.
            new_data: The name of the new data the Memlet should refer to.
        """
        mtree: dace.memlet.MemletTree = state.memlet_tree(edge)
        for tedge in mtree.traverse_children(True):
            # Because we only change the name of the data, we do not change the
            #  direction of the Memlet, so `{src, dst}_subset` will remain the same.
            if tedge.edge.data.data == old_data:
                tedge.edge.data.data = new_data


@dace_properties.make_properties
class GT4PyMapBufferElimination(dace_transformation.SingleStateTransformation):
    """Allows to remove unneeded buffering at map output.

    The transformation matches the case `MapExit -> (T) -> (G)`, where `T` is an
    AccessNode referring to a transient and `G` an AccessNode that refers to non
    transient memory.
    If the following conditions are met then `T` is removed.
    - `T` is not used to filter computations, i.e. what is written into `G`
        is covered by what is written into `T`.
    - `T` is not used anywhere else.
    - `G` is not also an input to the map, except there is only a pointwise
        dependency in `G`, see the note below.
    - Everything needs to be at top scope.

    Notes:
        - Rule 3 of ADR18 should guarantee that any valid GT4Py program meets the
            point wise dependency in `G`, for that reason it is possible to disable
            this test by specifying `assume_pointwise`.

    Todo:
        - Implement a real pointwise test.
        - Run this inside a pipeline.
    """

    map_exit = dace_transformation.PatternNode(dace_nodes.MapExit)
    tmp_ac = dace_transformation.transformation.PatternNode(dace_nodes.AccessNode)
    glob_ac = dace_transformation.PatternNode(dace_nodes.AccessNode)

    assume_pointwise = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="Dimensions that should become the leading dimension.",
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

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.map_exit, cls.tmp_ac, cls.glob_ac)]

    def depends_on(self) -> set[type[dace_transformation.Pass]]:
        return {dace_transformation.passes.ConsolidateEdges}

    def can_be_applied(
        self,
        graph: dace.SDFGState | dace.SDFG,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        tmp_ac: dace_nodes.AccessNode = self.tmp_ac
        glob_ac: dace_nodes.AccessNode = self.glob_ac
        tmp_desc: dace_data.Data = tmp_ac.desc(sdfg)
        glob_desc: dace_data.Data = glob_ac.desc(sdfg)

        if not tmp_desc.transient:
            return False
        if glob_desc.transient:
            return False
        if graph.in_degree(tmp_ac) != 1:
            return False
        if any(gtx_transformations.utils.is_view(ac, sdfg) for ac in [tmp_ac, glob_ac]):
            return False
        if len(glob_desc.shape) != len(tmp_desc.shape):
            return False

        # Test if we are on the top scope (it is likely).
        if graph.scope_dict()[glob_ac] is not None:
            return False

        # Now perform if we are point wise
        if not self._perform_pointwise_test(graph, sdfg):
            return False

        # Test if `tmp` is only anywhere else, this is important for removing it.
        if graph.out_degree(tmp_ac) != 1:
            return False
        # TODO(phimuell): Use the pipeline system to run the `StateReachability` pass
        #  only once. Taking care of DaCe issue 1911.
        if gtx_transformations.utils.is_accessed_downstream(
            start_state=graph,
            sdfg=sdfg,
            reachable_states=None,
            data_to_look=tmp_ac.data,
            nodes_to_ignore={tmp_ac},
        ):
            return False

        # Now we ensure that `tmp` is not used to filter out some computations.
        map_to_tmp_edge = next(edge for edge in graph.in_edges(tmp_ac))
        tmp_to_glob_edge = next(edge for edge in graph.out_edges(tmp_ac))

        tmp_in_subset = map_to_tmp_edge.data.get_dst_subset(map_to_tmp_edge, graph)
        tmp_out_subset = tmp_to_glob_edge.data.get_src_subset(tmp_to_glob_edge, graph)
        glob_in_subset = tmp_to_glob_edge.data.get_dst_subset(tmp_to_glob_edge, graph)
        if tmp_in_subset is None:
            tmp_in_subset = dace_subsets.Range.from_array(tmp_desc)
        if tmp_out_subset is None:
            tmp_out_subset = dace_subsets.Range.from_array(tmp_desc)
        if glob_in_subset is None:
            return False

        # TODO(phimuell): Do we need simplify in the check.
        # TODO(phimuell): Restrict this to having the same size.
        if tmp_out_subset != tmp_in_subset:
            return False
        return True

    def _perform_pointwise_test(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> bool:
        """Test if `G` is only point wise accessed.

        This function will also consider the `assume_pointwise` property.
        """
        map_exit: dace_nodes.MapExit = self.map_exit
        map_entry: dace_nodes.MapEntry = state.entry_node(map_exit)
        glob_ac: dace_nodes.AccessNode = self.glob_ac
        glob_data: str = glob_ac.data

        # First we check if `G` is also an input to this map.
        conflicting_inputs: set[dace_nodes.AccessNode] = set()
        for in_edge in state.in_edges(map_entry):
            if not isinstance(in_edge.src, dace_nodes.AccessNode):
                continue

            # Find the source of this data, if it is a view we trace it to
            #  its origin.
            src_node: dace_nodes.AccessNode = gtx_transformations.utils.track_view(
                in_edge.src, state, sdfg
            )

            # Test if there is a conflict; We do not store the source but the
            #  actual node that is adjacent.
            if src_node.data == glob_data:
                conflicting_inputs.add(in_edge.src)

        # If there are no conflicting inputs, then we are point wise.
        #  This is an implementation detail that make life simpler.
        if len(conflicting_inputs) == 0:
            return True

        # If we can assume pointwise computations, then we do not have to do
        #  anything.
        if self.assume_pointwise:
            return True

        # Currently the only test that we do is, if we have a view, then we
        #  are not point wise.
        # TODO(phimuell): Improve/implement this.
        return any(gtx_transformations.utils.is_view(node, sdfg) for node in conflicting_inputs)

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        # Removal
        # Propagation ofthe shift.
        map_exit: dace_nodes.MapExit = self.map_exit
        tmp_ac: dace_nodes.AccessNode = self.tmp_ac
        tmp_desc: dace_data.Data = tmp_ac.desc(sdfg)
        tmp_data = tmp_ac.data
        glob_ac: dace_nodes.AccessNode = self.glob_ac
        glob_data = glob_ac.data

        map_to_tmp_edge = next(edge for edge in graph.in_edges(tmp_ac))
        tmp_to_glob_edge = next(edge for edge in graph.out_edges(tmp_ac))

        glob_in_subset = tmp_to_glob_edge.data.get_dst_subset(tmp_to_glob_edge, graph)
        tmp_out_subset = tmp_to_glob_edge.data.get_src_subset(tmp_to_glob_edge, graph)
        if tmp_out_subset is None:
            tmp_out_subset = dace_subsets.Range.from_array(tmp_desc)
        assert glob_in_subset is not None

        # Recursively visit the nested SDFGs for mapping of strides from inner to outer array
        gtx_transformations.gt_map_strides_to_src_nested_sdfg(sdfg, graph, map_to_tmp_edge, glob_ac)

        # We now remove the `tmp` node, and create a new connection between
        #  the global node and the map exit.
        new_map_to_glob_edge = graph.add_edge(
            map_exit,
            map_to_tmp_edge.src_conn,
            glob_ac,
            tmp_to_glob_edge.dst_conn,
            dace.Memlet(
                data=glob_ac.data,
                subset=copy.deepcopy(glob_in_subset),
            ),
        )
        graph.remove_edge(map_to_tmp_edge)
        graph.remove_edge(tmp_to_glob_edge)
        graph.remove_node(tmp_ac)

        # We can not unconditionally remove the data `tmp` refers to, because
        #  it could be that in a parallel branch the `tmp` is also defined.
        try:
            sdfg.remove_data(tmp_ac.data, validate=True)
        except ValueError as e:
            if not str(e).startswith(f"Cannot remove data descriptor {tmp_ac.data}:"):
                raise

        # Now we must modify the memlets inside the map scope, because
        #  they now write into `G` instead of `tmp`, which has a different
        #  offset.
        # NOTE: Assumes that `tmp_out_subset` and `tmp_in_subset` are the same.
        correcting_offset = glob_in_subset.offset_new(tmp_out_subset, negative=True)
        mtree = graph.memlet_tree(new_map_to_glob_edge)
        for tree in mtree.traverse_children(include_self=False):
            curr_edge = tree.edge
            curr_dst_subset = curr_edge.data.get_dst_subset(curr_edge, graph)
            if curr_edge.data.data == tmp_data:
                curr_edge.data.data = glob_data
            if curr_dst_subset is not None:
                curr_dst_subset.offset(correcting_offset, negative=False)
