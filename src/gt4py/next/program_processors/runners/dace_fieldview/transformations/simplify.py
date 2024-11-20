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
    - `GT4PyRedundantArrayElimination`: Special version of the array removal, see
        documentation of `GT4PyRedundantArrayElimination`.
    - `GT4PyGlobalSelfCopyElimination`: Special copy pattern that in the context
        of GT4Py based SDFG behaves as a no op.

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
    skip = GT_SIMPLIFY_DEFAULT_SKIP_SET if skip is None else set(skip)

    result: Optional[dict[str, Any]] = None

    at_least_one_xtrans_run = True

    while at_least_one_xtrans_run:
        at_least_one_xtrans_run = False

        if "InlineSDFGs" not in skip:
            inline_res = gt_inline_nested_sdfg(
                sdfg=sdfg,
                multistate=True,
                permissive=False,
                validate=validate,
                validate_all=validate_all,
            )
            if inline_res is not None:
                at_least_one_xtrans_run = True
                result = result or {}
                result.update(inline_res)

        simplify_res = dace_passes.SimplifyPass(
            validate=validate,
            validate_all=validate_all,
            verbose=False,
            skip=(skip | {"InlineSDFGs"}),
        ).apply_pass(sdfg, {})

        if simplify_res is not None:
            at_least_one_xtrans_run = True
            result = result or {}
            result.update(simplify_res)

        if "GT4PyRedundantArrayElimination" not in skip:
            array_elimination_result = sdfg.apply_transformations_repeated(
                GT4PyRedundantArrayElimination(),
                validate=validate,
                validate_all=validate_all,
            )
            if array_elimination_result > 0:
                at_least_one_xtrans_run = True
                result = result or {}
                result.setdefault("GT4PyRedundantArrayElimination", 0)
                result["GT4PyRedundantArrayElimination"] += array_elimination_result

        if "GT4PyGlobalSelfCopyElimination" not in skip:
            self_copy_removal_result = sdfg.apply_transformations_repeated(
                GT4PyGlobalSelfCopyElimination(),
                validate=validate,
                validate_all=validate_all,
            )
            if self_copy_removal_result > 0:
                at_least_one_xtrans_run = True
                result = result or {}
                result.setdefault("GT4PyGlobalSelfCopyElimination", 0)
                result["GT4PyGlobalSelfCopyElimination"] += self_copy_removal_result

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
        validate: Perform validation at the end of the function.
        validate_all: Perform validation also on intermediate steps.
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


def gt_substitute_compiletime_symbols(
    sdfg: dace.SDFG,
    repl: dict[str, Any],
    validate: bool = False,
    validate_all: bool = False,
) -> None:
    """Substitutes symbols that are known at compile time with their value.

    Some symbols are known to have a constant value. This function will remove these
    symbols from the SDFG and replace them with the value.
    An example where this makes sense are strides that are known to be one.

    Args:
        sdfg: The SDFG to process.
        repl: Maps the name of the symbol to the value it should be replaced with.
        validate: Perform validation at the end of the function.
        validate_all: Perform validation also on intermediate steps.
    """

    # We will use the `replace` function of the top SDFG, however, lower levels
    #  are handled using ConstantPropagation.
    sdfg.replace_dict(repl)

    const_prop = dace_passes.ConstantPropagation()
    const_prop.recursive = True
    const_prop.progress = False

    const_prop.apply_pass(
        sdfg=sdfg,
        initial_symbols=repl,
        _=None,
    )
    gt_simplify(
        sdfg=sdfg,
        validate=validate,
        validate_all=validate_all,
    )


def gt_reduce_distributed_buffering(
    sdfg: dace.SDFG,
) -> Optional[dict[dace.SDFG, dict[dace.SDFGState, set[str]]]]:
    """Removes distributed write back buffers."""
    pipeline = dace_ppl.Pipeline([DistributedBufferRelocator()])
    all_result = {}

    for rsdfg in sdfg.all_sdfgs_recursive():
        ret = pipeline.apply_pass(sdfg, {})
        if ret is not None:
            all_result[rsdfg] = ret

    return all_result


@dace_properties.make_properties
class GT4PyRedundantArrayElimination(dace_transformation.SingleStateTransformation):
    """Special version of the redundant array removal transformation.

    DaCe is not able to remove some redundant arrays. This transformation is specially
    designed to remove these transient arrays. It matches two array `read` that is
    read and written into `write`. The transformation applies if:
    - `read` is a transient non view array.
    - `write` has input degree 1 and output degree zero (sink node might be lifted).
    - If `write` is non-transient, then `read` shall not have a upstream access node
        that refers to the same global data.
    - `read` has input degree larger than zero and output degree 1.
    - `read` does not appear in any other state that is reachable from this state.
    - `read` and `write` have the same dimensionality.
    - In case they have different sizes the following must hold:
        - `read` has input degree 1.
        - All subsets have the same size, i.e. everything that is written into `read`
            is also read from it.

    Then array `read` is removed from the SDFG.

    This pass takes advantages of the structure of the SDFG outlined in:
    docs/development/ADRs/0018-Canonical_SDFG_in_GT4Py_Transformations.md
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

        # Check if used anywhere else.
        # TODO(phimuell): Find a way to cache this information.
        if self._check_if_read_is_used_downstream(graph, sdfg):
            return False

        # If `write` is global memory, then we do not remove `read` if there if
        #  `read` is written to by another access node that also refers to `write`.
        #  Essentially this prevents that global data can write into itself. It
        #  might be possible to lift it in some cases, but not in all cases.
        #  Consider this case:
        # ```
        #   tmp = a[0:20]  # noqa: ERA001 [commented-out-code]
        #   a[10:30] = tmp  # noqa: ERA001 [commented-out-code]
        # ```
        #  which can only be solved, if the copy is done backwards (I am not
        #  sure if DaCe uses `memmove()`), but any way we fully forbid that case,
        #  since it is bother-line invalid anyway.
        if not write_desc.transient:
            for read_in_edge in graph.in_edges(read_an):
                producer_node = read_in_edge.src
                if not isinstance(producer_node, dace_nodes.AccessNode):
                    continue
                producer_desc = producer_node.desc(sdfg)
                if producer_node.data == write_an.data:
                    return False
                elif isinstance(producer_desc, dace_data.View):
                    # TODO(phimuell): Handle this case
                    return False

        # NOTE: We do _not_ require that the whole `read` array is actually read, nor
        #  that the array is fully written to. Because of our SDFG structure.
        #  What we accepts depends on if `read` and `write` have the same shape.
        if write_desc.shape == read_desc.shape:
            # If `read` and write` have the same shape, then there are only two
            #  kind of restriction. Consider the following:
            # ```
            #   read[0:100] = ...  # noqa: ERA001 [commented-out-code]
            #   write[0:50] = read[0:50]  # noqa: ERA001 [commented-out-code]
            # ```
            #  This would become:
            # ```
            #   write[0:100] = ...  # noqa: ERA001 [commented-out-code]
            # ```
            #  This is a semantic change, since now `write` is defined in the range
            #  `0:100` instead of `0:50` as before. However, because it has only one
            #  incoming edge and our SSA rule, we know that the range `50:100` could
            #  only contain invalid data, so reading from it would be wrong anyway
            #  so we can safely write to it. The only problem is if `write` is
            #  non transient data, in which case we would need to adjust the subsets.
            #  The second problem are operations such as:
            # ```
            #   read[0:100] = ...  # noqa: ERA001 [commented-out-code]
            #   write[0:50] = read[25:75]  # noqa: ERA001 [commented-out-code]
            # ```
            #  Because now we must account for offsets.

            # What is actually transferred, between `read` and `write`.
            read_out_edge = next(iter(graph.out_edges(read_an)))
            write_dst_subset = read_out_edge.data.get_dst_subset(read_out_edge, graph)
            if write_dst_subset is None:
                write_dst_subset = dace_subsets.Range.from_array(write_desc)
            write_src_subset = read_out_edge.data.get_src_subset(read_out_edge, graph)
            if write_src_subset is None:
                write_src_subset = dace_subsets.Range.from_array(read_desc)

            # To avoid that we have to take into account offsets we require that
            #  both offsets starts at zero. Note that this is not a real restriction
            #  it just makes the implementation simpler.
            # TODO(phimuell): Lift this restriction.
            zero_start = (0,) * len(write_desc.shape)
            if tuple(write_dst_subset.min_element()) != zero_start:
                return False
            if tuple(write_src_subset.min_element()) != zero_start:
                return False

            if write_desc.transient:
                # `write` is a transient, in which case we can be more permissive.
                #  Actually the tests above, to ensure that everything starts at
                #  zero are already enough in the transient case. See also the
                #  discussion above for more information.
                pass
            else:
                # `write` is global data, so we must be more careful. We have to
                #  ensure that `read` is not used to filter out some computation.
                for read_in_edge in graph.in_edges(read_an):
                    read_dst_subset = read_in_edge.data.get_dst_subset(read_in_edge, graph)
                    if read_dst_subset is None:
                        read_dst_subset = dace_subsets.Range.from_array(read_desc)
                    # Test if the edges writes something that is later not read.
                    if not write_src_subset.covers(read_dst_subset):
                        return False
        else:
            # They have different shapes, which is much more complicated to handle.
            # TODO(phimuell): revisit after we have `concat_where`.

            # For simplicity we assume that there is only one producer.
            #  And that they have the same dimensionality.
            if graph.in_degree(read_an) != 1:
                return False
            if len(write_desc.shape) != len(read_desc.shape):
                return False

            read_in_edge = next(iter(graph.in_edges(read_an)))
            read_out_edge = next(iter(graph.out_edges(read_an)))

            # Check if everything that is read from `read` is also defined.
            #  This is not a real requirement, it just simplify the offset computation
            #  later on. If we would remove this restriction it means that we would
            #  allow to copy around undefined data. Which might be helpful in some
            #  cases, especially in chains, but we currently do not handle it.
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

            # In case `read` is not written to by an AccessNode, we require that
            #  everything that is written must also be read. This is because only
            #  in cases for AccessNodes we can regulate what is further written to
            #  into `write`.
            if not isinstance(read_in_edge.src, dace_nodes.AccessNode):
                if read_isubset != read_osubset:
                    return False

            # If everything is written into `read` that is not also read later and
            #  transferred to `write`. However, we have to exclude the case if both
            #  have the same size.
            if read_osubset.covers(read_isubset) and (not read_osubset == read_isubset):
                return False

        return True

    def _check_if_read_is_used_downstream(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> bool:
        read_an: dace_nodes.AccessNode = self.read
        read_name: str = read_an.data

        to_visit: list[dace.SDFGState] = [edge.dst for edge in sdfg.out_edges(state)]
        seen: set[dace.SDFGState] = {state}

        while len(to_visit) != 0:
            state_to_inspect = to_visit.pop()
            if any(dnode.data == read_name for dnode in state_to_inspect.data_nodes()):
                return True
            to_visit.extend(
                edge.dst for edge in sdfg.out_edges(state_to_inspect) if edge.dst not in seen
            )
        return False

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
        curr_dst_subset_start = curr_dst_subset.min_element()

        if write_desc.shape == read_desc.shape:
            # The shapes are the same, so no subset translation is needed.
            for iedge in graph.in_edges(read_an):
                org_memlet: dace.Memlet = iedge.data
                src_subset: dace_subsets.Subset = copy.deepcopy(
                    org_memlet.get_src_subset(iedge, graph)
                )
                if src_subset is None and isinstance(iedge.src, dace_nodes.AccessNode):
                    src_subset = dace_subsets.Range.from_array(iedge.src.desc(sdfg))
                if src_subset is not None:
                    src_subset.offset(curr_dst_subset_start, negative=False)

                dst_subset: dace_subsets.Subset = copy.deepcopy(
                    org_memlet.get_dst_subset(iedge, graph)
                )
                if dst_subset is None and isinstance(iedge.dst, dace_nodes.AccessNode):
                    dst_subset = dace_subsets.Range.from_array(iedge.dst.desc.sdfg)
                new_edge = graph.add_edge(
                    iedge.src,
                    iedge.src_conn,
                    write_an,
                    write_in_edge.dst_conn,
                    copy.deepcopy(org_memlet),
                )
                # Modify the memlet, mostly adjust the subset and direction.
                new_edge.data.data = write_name
                new_edge.data.try_initialize(graph.parent, graph, new_edge)
                new_edge.data.subset = dst_subset
                new_edge.data.other_subset = src_subset
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
        try:
            sdfg.remove_data(read_name)
        except ValueError:
            # This might happen if the data is still used somewhere else, but the
            #  transformation has decided that it is safe to remove it.
            pass


@dace_properties.make_properties
class GT4PyGlobalSelfCopyElimination(dace_transformation.SingleStateTransformation):
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

    node_read_g = dace_transformation.transformation.PatternNode(dace_nodes.AccessNode)
    node_tmp = dace_transformation.transformation.PatternNode(dace_nodes.AccessNode)
    node_write_g = dace_transformation.transformation.PatternNode(dace_nodes.AccessNode)

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

        return gtx_transformations.util.is_accessed_downstream(
            start_state=start_state,
            sdfg=sdfg,
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
            sdfg.remove_data(tmp_node.data)


AccessLocation: TypeAlias = tuple[dace.SDFGState, dace_nodes.AccessNode]
"""Describes an access node and the state in which it is located.
"""


@dace_properties.make_properties
class DistributedBufferRelocator(dace_transformation.Pass):
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

    Todo:
        - Allow that `dst_cont` can also be transient.
        - Allow that `dst_cont` does not need to be a sink node, this is most
            likely most relevant if it is transient.
        - Check if `dst_cont` is used between where we want to place it and
            where it is currently used.
    """

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
        """Determines all temporaries that have to be relocated.

        Returns:
            A list of tuples. The first element element of the tuple is an
            `AccessLocation` that describes where the temporary is read.
            The second element is a list of `AccessLocation`s that describes
            where the temporary is defined.
        """
        # All nodes that are used as distributed buffers.
        candidate_src_cont: list[AccessLocation] = []

        # Which `src_cont` access node is written back to which global memory.
        src_cont_to_global: dict[dace_nodes.AccessNode, str] = {}

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

            for src_cont in state.source_nodes():
                if not isinstance(src_cont, dace_nodes.AccessNode):
                    continue
                if not src_cont.desc(sdfg).transient:
                    continue
                if state.out_degree(src_cont) != 1:
                    continue
                dst_candidate: dace_nodes.AccessNode = next(
                    iter(edge.dst for edge in state.out_edges(src_cont))
                )
                if dst_candidate not in candidate_dst_nodes:
                    continue
                candidate_src_cont.append((src_cont, state))
                src_cont_to_global[src_cont] = dst_candidate.data

        if len(candidate_src_cont) == 0:
            return []

        # Now we have to find the places where the temporary sources are defined.
        #  I.e. This is also the location where the original value is defined.
        result_candidates: list[tuple[AccessLocation, list[AccessLocation]]] = []

        def find_upstream_states(dst_state: dace.SDFGState) -> set[dace.SDFGState]:
            return {
                src_state
                for src_state in sdfg.states()
                if dst_state in reachable[src_state] and dst_state is not src_state
            }

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
                result_candidates.append((src_cont, def_locations))

        # This transformation removes `src_cont` by writing its content directly
        #  to `dst_cont`, at the point where it is defined.
        #  That this transformation is valid the following conditions have to be met:
        #   - Between the definition of `src_cont` and the write back to `dst_cont`,
        #       `dst_cont` can not be accessed.
        #   - Between the definitions of `src_cont` and the point where it is written
        #       back, `src_cont` can only be accessed in the range that is written back.
        #   - After the write back point, `src_cont` shall not be accessed. This
        #       restriction can be lifted, but it is super strict.
        #
        #  To keep the implementation simple, we use the conditions:
        #   - `src_cont` is only accessed were it is defined and at the write back
        #       point.
        #   - Between the definitions of `src_cont` and the write back point,
        #       `dst_cont` is not used.

        result: list[tuple[AccessLocation, list[AccessLocation]]] = []

        for wb_localation, def_locations in result_candidates:
            for def_node, def_state in def_locations:
                # Test if `src_cont` is only accessed where it is defined and
                #  where it is written back.
                if gtx_transformations.util.is_accessed_downstream(
                    start_state=def_state,
                    sdfg=sdfg,
                    data_to_look=wb_localation[0].data,
                    nodes_to_ignore={def_node, wb_localation[0]},
                ):
                    break
                # check if the global data is not used between the definition of
                #  `dst_cont` and where its written back. We allow one exception,
                #  if the global data is used in the state the distributed temporary
                #  is defined is used only for reading then it is ignored. This is
                #  allowed because of rule 3 of ADR0018.
                glob_nodes_in_def_state = {
                    dnode
                    for dnode in def_state.data_nodes()
                    if dnode.data == src_cont_to_global[wb_localation[0]]
                }
                if any(def_state.in_degree(gdnode) != 0 for gdnode in glob_nodes_in_def_state):
                    break
                if gtx_transformations.util.is_accessed_downstream(
                    start_state=def_state,
                    sdfg=sdfg,
                    data_to_look=src_cont_to_global[wb_localation[0]],
                    nodes_to_ignore=glob_nodes_in_def_state,
                    states_to_ignore={wb_localation[1]},
                ):
                    break
            else:
                result.append((wb_localation, def_locations))

        return result
