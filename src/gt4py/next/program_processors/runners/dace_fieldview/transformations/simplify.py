# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The GT4Py specific simplification pass."""

import copy
from typing import Any, Final, Iterable, Optional, Union

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    subsets as dace_subsets,
    transformation as dace_transformation,
)
from dace.sdfg import nodes as dace_nodes
from dace.transformation import dataflow as dace_dataflow, passes as dace_passes

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


@dace_properties.make_properties
class GT4PyRednundantArrayElimination(dace_transformation.SingleStateTransformation):
    """Special version of the redundant array removal transformation.

    DaCe is not able to remove redundant arrays. This transformation is specially
    designed to remove these transient arrays. It matches two array `read` that is
    read and written into `write`. The transformation applies if:
    - `read` is a transient non view array.
    - `write` has input degree 1 and output degree zero.
    - `read` has input degree larger than zero and output degree 1.
    - `read` does not appear in any other state; by construction in other states
        it can only be read.
    - They have the same size (might be lifted).
    - The content of the full array must be transferred (there might be a way to work
        around that).

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

        if not (write_desc.transient and read_desc.transient):
            return False
        if any(isinstance(desc, dace_data.View) for desc in [read_desc, write_desc]):
            return False
        if write_desc.shape != read_desc.shape:  # TODO(phimuell): Add simplify
            return False
        if graph.in_degree(read_an) == 0:
            return False
        if graph.out_degree(read_an) != 1:
            return False
        if graph.out_degree(write_an) != 0:
            return False
        if graph.in_degree(write_an) != 1:
            return False

        # Ensure that the whole array `read` is transferred to the second array.
        edge = next(iter(graph.in_edges(write_an)))
        subset = edge.data.get_src_subset(edge, graph)
        if subset is None:
            subset = edge.data.get_dst_subset(edge, graph)
        assert subset is not None

        if write_desc.shape != tuple(subset.size()):
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

        for iedge in graph.in_edges(read_an):
            org_memlet: dace.Memlet = iedge.data
            src_subset: dace_subsets.Subset = copy.deepcopy(org_memlet.get_src_subset(iedge, graph))
            dst_subset: dace_subsets.Subset = copy.deepcopy(org_memlet.get_dst_subset(iedge, graph))
            new_edge = graph.add_edge(
                iedge.src,
                iedge.src_conn,
                write_an,
                iedge.dst_conn,
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
