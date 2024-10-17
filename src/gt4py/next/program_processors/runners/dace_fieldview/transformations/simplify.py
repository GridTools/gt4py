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
from typing import Any, Final, Iterable, Optional, TypeAlias, Union

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    subsets as dace_subsets,
    transformation as dace_transformation,
)
from dace.sdfg import nodes as dace_nodes
from dace.transformation import (
    dataflow as dace_dataflow,
    pass_pipeline as dace_ppl,
    passes as dace_passes,
)

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)


GT_SIMPLIFY_DEFAULT_SKIP_SET: Final[set[str]] = {"ScalarToSymbolPromotion", "ConstantPropagation"}
"""Set of simplify passes `gt_simplify()` skips by default.

The following passes are included:
- `ScalarToSymbolPromotion`: The lowering has sometimes to turn a scalar into a
    symbol or vice versa and at a later point to invert this again. However, this
    pass has some problems with this pattern so for the time being it is disabled.
- `ConstantPropagation`: Same reasons as `ScalarToSymbolPromotion`.
"""


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

    Further, the function will run the following passes in addition to DaCe simplify:
    - `GT4PyRednundantArrayElimination`: Special version of the array removal, see
        documentation of `GT4PyRednundantArrayElimination`.

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
        elimination at the begin. Thus, `gt_simplify()` might not result in a fix
        point. This is an implementation detail that will change in the future.
    """
    # Ensure that `skip` is a `set`
    skip = GT_SIMPLIFY_DEFAULT_SKIP_SET if skip is None else set(skip)

    result: Optional[dict[str, Any]] = None

    if "InlineSDFGs" not in skip:
        inline_res = gt_inline_nested_sdfg(
            sdfg=sdfg,
            multistate=True,
            permissive=False,
            validate=validate,
            validate_all=validate_all,
        )
        if inline_res is not None:
            result = inline_res

    simplify_res = dace_passes.SimplifyPass(
        validate=validate,
        validate_all=validate_all,
        verbose=False,
        skip=(skip | {"InlineSDFGs"}),
    ).apply_pass(sdfg, {})

    if simplify_res is not None:
        result = result or {}
        result.update(simplify_res)

    if "GT4PyRednundantArrayElimination" not in skip:
        array_elimination_result = sdfg.apply_transformations_repeated(
            GT4PyRednundantArrayElimination(),
            validate=validate,
            validate_all=validate_all,
        )
        if array_elimination_result is not None:
            result = result or {}
            result["GT4PyRednundantArrayElimination"] = array_elimination_result

    return result


def gt_set_iteration_order(
    sdfg: dace.SDFG,
    leading_dim: Optional[
        Union[str, gtx_common.Dimension, list[Union[str, gtx_common.Dimension]]]
    ] = None,
    validate: bool = True,
    validate_all: bool = False,
) -> Any:
    """Set the iteration order of the Maps correctly.

    Modifies the order of the Map parameters such that `leading_dim`
    is the fastest varying one, the order of the other dimensions in
    a Map is unspecific. `leading_dim` should be the dimensions were
    the stride is one.

    Args:
        sdfg: The SDFG to process.
        leading_dim: The leading dimensions.
        validate: Perform validation during the steps.
        validate_all: Perform extensive validation.
    """
    return sdfg.apply_transformations_once_everywhere(
        gtx_transformations.MapIterationOrder(
            leading_dims=leading_dim,
        ),
        validate=validate,
        validate_all=validate_all,
    )


def gt_inline_nested_sdfg(
    sdfg: dace.SDFG,
    multistate: bool = True,
    permissive: bool = False,
    validate: bool = True,
    validate_all: bool = False,
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
    first_iteration = True
    nb_preproccess_total = 0
    nb_inlines_total = 0
    while True:
        nb_preproccess = sdfg.apply_transformations_repeated(
            [dace_dataflow.PruneSymbols, dace_dataflow.PruneConnectors],
            validate=False,
            validate_all=validate_all,
        )
        nb_preproccess_total += nb_preproccess
        if (nb_preproccess == 0) and (not first_iteration):
            break

        # Create and configure the inline pass
        inline_sdfg = dace_passes.InlineSDFGs()
        inline_sdfg.progress = False
        inline_sdfg.permissive = permissive
        inline_sdfg.multistate = multistate

        # Apply the inline pass
        #  The pass returns `None` no indicate "nothing was done"
        nb_inlines = inline_sdfg.apply_pass(sdfg, {}) or 0
        nb_inlines_total += nb_inlines

        # Check result, if needed and test if we can stop
        if validate_all or validate:
            sdfg.validate()
        if nb_inlines == 0:
            break
        first_iteration = False

    result: dict[str, int] = {}
    if nb_inlines_total != 0:
        result["InlineSDFGs"] = nb_inlines_total
    if nb_preproccess_total != 0:
        result["PruneSymbols|PruneConnectors"] = nb_preproccess_total
    return result if result else None


def gt_reduce_distributed_buffering(
    sdfg: dace.SDFG,
) -> Optional[dict[dace.SDFG, dict[dace.SDFGState, set[str]]]]:
    """Removes distributed write back buffers."""
    pipeline = dace_ppl.Pipeline([DistributedBufferRelocater()])
    all_result = {}

    for rsdfg in sdfg.all_sdfgs_recursive():
        ret = pipeline.apply_pass(sdfg, {})
        if ret is not None:
            all_result[rsdfg] = ret

    return all_result


@dace_properties.make_properties
class GT4PyRednundantArrayElimination(dace_transformation.SingleStateTransformation):
    """Special version of the redundant array removal transformation.

    DaCe is not able to remove redundant arrays. This transformation is specially
    designed to remove these transient arrays. It matches two array `read` that is
    read and written into `write`. The transformation applies if:
    - `read` is a transient non view array.
    - `write` has input degree 1 and output degree zero (sink node might be lifted).
    - `read` has input degree larger than zero and output degree 1.
    - `read` does not appear in any other state; by construction in other states
        it can only be read.
    - `read` and `write` have the same dimensionality.
    - In case they have different sizes the following must hold:
        - `read` has input degree 1.
        - All subsets have the same size, i.e. everything that is written into `read`
            is also read from it.

    Then array `read` is removed from the SDFG.

    This passes takes advantages of the structure of the SDFG outlined in:
    https://github.com/GridTools/gt4py/tree/main/docs/development/ADRs/0018-Canonical_SDFG_in_GT4Py_Transformations.md
    """

    read = dace_transformation.transformation.PatternNode(dace_nodes.AccessNode)
    write = dace_transformation.transformation.PatternNode(dace_nodes.AccessNode)

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.read, cls.write)]

    def can_be_applied(
        self,
        graph: dace.SDFGState | dace.SDFG,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Tests if the requirements listed above are met."""
        read_an: dace_nodes.AccessNode = self.read
        write_an: dace_nodes.AccessNode = self.write
        read_desc: dace_data.Data = read_an.desc(sdfg)
        write_desc: dace_data.Data = write_an.desc(sdfg)

        if not read_desc.transient:
            return False
        if any(isinstance(desc, dace_data.View) for desc in [read_desc, write_desc]):
            return False
        if graph.in_degree(read_an) == 0:
            return False
        if graph.out_degree(read_an) != 1:
            return False
        if graph.out_degree(write_an) != 0:
            return False
        if graph.in_degree(write_an) != 1:
            return False
        if len(read_desc.shape) != len(write_desc.shape):
            return False

        # NOTE:
        #   We do _not_ require that the whole `read` array is read, nor this array
        #   is fully written. If it is not fully written (at this access node) then
        #   these parts will remain undefined (as the array can not be written to).
        #   If we would require that `read` is fully read, but `read` is not fully
        #   written to in the first place, we would override undefined data with
        #   undefined data.

        # By requiring the same shape, we do not have to do subset translation.
        if write_desc.shape != read_desc.shape:
            # In certain cases we allow that the two data have different shapes.
            if graph.in_degree(read_an) != 1:
                return False

            # Check if everything that is read from `read` is also defined.
            #  This is not a real requirement, it just simplify the offset computation
            #  later on. If we would remove this restriction it means that we would
            #  allow to copy around undefined data. Which might be helpful in some
            #  cases, especially in chains, but we currently do not handle it.
            read_in_edge = next(iter(graph.in_edges(read_an)))
            read_out_edge = next(iter(graph.out_edges(read_an)))
            read_isubset = read_in_edge.data.get_dst_subset(read_in_edge, graph)
            read_osubset = read_out_edge.data.get_src_subset(read_out_edge, graph)
            if read_osubset is None:
                read_osubset = dace_subsets.Range.from_array(read_desc)
            if read_isubset is None:
                read_isubset = dace_subsets.Range.from_array(read_desc)
            if not isinstance(read_isubset, dace_subsets.Range):
                return False
            if not isinstance(read_osubset, dace_subsets.Range):
                return False
            if read_isubset.size() != read_osubset.size():
                return False

        # Check if used anywhere else.
        # TODO(phimuell): Find a way to cache this information.
        read_name: str = read_an.data
        for state in sdfg.states():
            if any(
                (node is not read_an) and (read_name == node.data) for node in state.data_nodes()
            ):
                return False

        return True

    def apply(
        self,
        graph: dace.SDFGState | dace.SDFG,
        sdfg: dace.SDFG,
    ) -> None:
        """Removes the array that is read from."""
        read_an: dace_nodes.AccessNode = self.read
        write_an: dace_nodes.AccessNode = self.write
        write_name: str = write_an.data
        read_name: str = read_an.data

        read_desc: dace_data.Data = read_an.desc(sdfg)
        write_desc: dace_data.Data = write_an.desc(sdfg)

        # The destination of the new edge, is fully given by the subset of the
        #  incoming edge at `write`.
        write_in_edge = next(iter(graph.in_edges(write_an)))
        curr_dst_subset = copy.deepcopy(write_in_edge.data.get_dst_subset(write_in_edge, graph))
        if curr_dst_subset is None:
            curr_dst_subset = dace_subsets.Range.from_array(write_desc)

        if write_desc.shape == read_desc.shape:
            # The shapes are the same, so no subset translation is needed.
            for iedge in graph.in_edges(read_an):
                org_memlet: dace.Memlet = iedge.data
                src_subset: dace_subsets.Subset = copy.deepcopy(
                    org_memlet.get_src_subset(iedge, graph)
                )
                dst_subset: dace_subsets.Subset = copy.deepcopy(
                    org_memlet.get_dst_subset(iedge, graph)
                )
                new_edge = graph.add_edge(
                    iedge.src,
                    iedge.src_conn,
                    write_an,
                    write_in_edge.dst_conn,
                    copy.deepcopy(org_memlet),
                )
                # Modify the memlet, mostly adjust the subset and direction.
                new_edge.data.data = write_name
                new_edge.data.subset = dst_subset
                new_edge.data.other_subset = src_subset
                new_edge.data.try_initialize(graph.parent, graph, new_edge)
                assert src_subset is new_edge.data.src_subset
                assert dst_subset is new_edge.data.dst_subset
                graph.remove_edge(iedge)

        else:
            # The shapes different, so we have to do translation of the subset.
            #  This is a special case so we know that there is only one edge involved.
            # TODO(phimuell): extend to multiple incoming edges, however, the
            #   subset they write into must be distinct.
            assert graph.out_degree(read_an) == 1

            read_in_edge = next(iter(graph.in_edges(read_an)))
            read_out_edge = next(iter(graph.out_edges(read_an)))

            # The source subset of the new connection comes from the connection that
            #  goes into `read`. If it is `None` we might need to correct it.
            old_origin_subset = read_in_edge.data.get_src_subset(read_in_edge, graph)
            if old_origin_subset is None and isinstance(read_in_edge.src, dace_nodes.AccessNode):
                old_origin_subset = dace_subsets.Range.from_array(read_in_edge.src.desc(sdfg))

            if old_origin_subset is not None:
                read_isubset = read_in_edge.data.get_dst_subset(read_in_edge, graph)
                read_osubset = read_out_edge.data.get_src_subset(read_out_edge, graph)
                if read_osubset is None:
                    read_osubset = dace_subsets.Range.from_array(read_desc)
                if read_isubset is None:
                    read_isubset = dace_subsets.Range.from_array(read_desc)
                transfered_size = read_osubset.size()
                read_isubset_min = read_isubset.min_element()
                read_osubset_min = read_osubset.min_element()
                old_origin_subset_min = old_origin_subset.min_element()

                new_origin_subset_parts: list[tuple[Any, Any, int]] = []
                for isbs, osbs, size, origin_sbs in zip(
                    read_isubset_min, read_osubset_min, transfered_size, old_origin_subset_min
                ):
                    new_origin_start = f"(({osbs!s} - {isbs!s}) + {origin_sbs!s})"
                    # The `-1` is because in this mode of construction end is inclusive.
                    new_origin_end = f"((({new_origin_start}) + ({size!s})) - 1)"
                    new_origin_subset_parts.append((new_origin_start, new_origin_end, 1))
                new_origin_subset = dace_subsets.Range(new_origin_subset_parts)

            else:
                new_origin_subset = None

            # Now we create the new edge, we will adjust the memlet afterwards.
            new_edge = graph.add_edge(
                read_in_edge.src,
                read_in_edge.src_conn,
                write_an,
                write_in_edge.dst_conn,
                copy.deepcopy(read_in_edge.data),
            )
            # Modify the memlet, mostly adjust the subset and direction.
            new_edge.data.data = write_name
            new_edge.data.try_initialize(graph.parent, graph, new_edge)
            new_edge.data.subset = write_in_edge.data.get_dst_subset(write_in_edge, graph)
            new_edge.data.other_subset = new_origin_subset
            graph.remove_edge(read_in_edge)

        for oedge in graph.out_edges(read_an):
            graph.remove_edge(oedge)

        # Now we have to adjust all memlets in scopes.
        for iedge in graph.in_edges(write_an):
            mtree = sdfg.memlet_tree(iedge)
            for tree_edge in mtree.traverse_children(False):
                if tree_edge.edge.data.data == read_name:
                    tree_edge.edge.data.data = write_name

        assert graph.degree(read_an) == 0
        graph.remove_node(read_an)
        sdfg.remove_data(read_name)


AccessLocation: TypeAlias = tuple[dace.SDFGState, dace_nodes.AccessNode]
"""Describes an access node and the state in which it is located.
"""


@dace_properties.make_properties
class DistributedBufferRelocater(dace_transformation.Pass):
    """Moves the final write back of the results to where it is needed.

    In certain cases, especially in case where we have `if` the result is computed
    in each branch and then in the join state written back. Thus there is some
    additional storage needed.
    The transformation will look for the following situation:
    - A transient data container, called `src_cont`, is written into another
        container, called `dst_cont`, which is not transient.
    - The access node of `src_cont` has an in degree of zero and an out degree of one.
    - The access node of `dst_cont` has an in degree of of one and an
        out degree of zero (this might be lifted).
    - `src_cont` is not used afterwards.
    - `dst_cont` is only used to implement the buffering.

    The function will relocate the writing of `dst_cont` to where `src_cont` is
    written, which might be multiple locations.
    It will also remove the writing back.
    It is advised that after this transformation simplify is run again.

    Note:
        Essentially this transformation removes the double buffering of `dst_cont`.
        Because we ensure that that `dst_cont` is non transient this is okay, as our
        rule guarantees this.

    Todo: Also check if the stuff we write back to is used in between.
    """

    CATEGORY: str = "Simplification"

    def modifies(self) -> dace_ppl.Modifies:
        return dace_ppl.Modifies.Memlets | dace_ppl.Modifies.AccessNodes

    def should_reapply(self, modified: dace_ppl.Modifies) -> bool:
        return modified & (dace_ppl.Modifies.Memlets | dace_ppl.Modifies.AccessNodes)

    def depends_on(self) -> set[type[dace_transformation.Pass]]:
        return {
            dace_transformation.passes.analysis.StateReachability,
            dace_transformation.passes.analysis.AccessSets,
        }

    def apply_pass(
        self, sdfg: dace.SDFG, pipeline_results: dict[str, Any]
    ) -> Optional[dict[dace.SDFGState, set[str]]]:
        """
        Removes double buffering that is distrbuted among multiple states.

        Returns:
            TBA

        Args:
            sdfg: The SDFG to process.
            pipeline_results: Result of previous analysis passes.
        """
        reachable: dict[dace.SDFGState, set[dace.SDFGState]] = pipeline_results[
            "StateReachability"
        ][sdfg.cfg_id]
        access_sets: dict[dace.SDFGState, tuple[set[str], set[str]]] = pipeline_results[
            "AccessSets"
        ][sdfg.cfg_id]
        result: dict[dace.SDFGState, set[str]] = collections.defaultdict(set)

        to_relocate = self._find_candidates(sdfg, reachable, access_sets)
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
                def_state.add_edge(
                    def_an,
                    wb_edge.src_conn,
                    def_state.add_access(final_dest_name),
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
        access_sets: dict[dace.SDFGState, tuple[set[str], set[str]]],
    ) -> list[tuple[AccessLocation, list[AccessLocation]]]:
        """Determines all candidates that needs to be processed.

        Returns:
            A list of tuples. The first element of a tuple is an `AccessLocation`,
            describing where the write back occurs. The second element is a list,
            containing the locations where the node is originally written.
        """
        # First lets find us all nodes that might be `src_cont` candidate.
        candidate_src_cont: list[AccessLocation] = []
        for state in sdfg.states():
            candidate_dst_nodes: set[dace_nodes.AccessNode] = {
                node
                for node in state.sink_nodes()
                if isinstance(node, dace_nodes.AccessNode) and (not node.desc(sdfg).transient)
            }
            if len(candidate_dst_nodes) == 0:
                continue

            for src_cont in state.source_nodes():
                if not isinstance(src_cont, dace_nodes.AccessNode):
                    continue
                if state.out_degree(src_cont) != 1:
                    continue
                if not src_cont.desc(sdfg).transient:
                    continue
                if not all(edge.dst in candidate_dst_nodes for edge in state.out_edges(src_cont)):
                    continue
                candidate_src_cont.append((src_cont, state))

        if len(candidate_src_cont) == 0:
            return []

        # Now we have to ensure that after `src_cont` has been read is no longer in use.
        def is_not_used_after(data: str, wb_state: dace.SDFGState) -> bool:
            for down_state in reachable[wb_state]:
                if any(data in read_set for read_set, _ in access_sets[down_state]):
                    return False
            return True

        candidate_src_cont = [
            wb_location
            for wb_location in candidate_src_cont
            if is_not_used_after(wb_location[0].data, wb_location[1])
        ]

        if len(candidate_src_cont) == 0:
            return []

        # Now we have to find the place where the temporary is written.
        def find_upstream_states(state: dace.SDFGState) -> set[dace.SDFGState]:
            return {
                astate
                for astate in sdfg.states()
                if astate not in reachable[state] and astate is not state
            }

        # Now we have to find the places where the temporary sources are defined.
        result: list[tuple[AccessLocation, list[AccessLocation]]] = []
        for src_cont in candidate_src_cont:
            def_locations: list[AccessLocation] = []
            for upstream_state in find_upstream_states(src_cont[1]):
                if src_cont[0].data in access_sets[upstream_state][1]:
                    def_locations.extend(
                        (data_node, upstream_state)
                        for data_node in upstream_state.data_nodes()
                        if data_node.data == src_cont[0].data
                    )
            if len(def_locations) != 0:
                result.append((src_cont, def_locations))

        return result
