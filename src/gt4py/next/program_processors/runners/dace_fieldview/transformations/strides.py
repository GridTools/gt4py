# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dace
from dace import data as dace_data

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)


def gt_change_transient_strides(
    sdfg: dace.SDFG,
    gpu: bool,
) -> dace.SDFG:
    """Modifies the strides of transients.

    The function will analyse the access patterns and set the strides of
    transients in the optimal way.
    The function should run after all maps have been created.

    Args:
        sdfg: The SDFG to process.
        gpu: If the SDFG is supposed to run on the GPU.

    Note:
        Currently the function will not scan the access pattern. Instead it will
        either use FORTRAN order for GPU or C order (which is assumed to be the
        default, so it is a no ops).

    Todo:
        - Implement the estimation correctly.
        - Handle the case of nested SDFGs correctly; on the outside a transient,
            but on the inside a non transient.
    """
    # TODO(phimeull): Implement this function correctly.

    # We assume that by default we have C order which is already correct,
    #  so in this case we have a no ops
    if not gpu:
        return sdfg

    for nsdfg in sdfg.all_sdfgs_recursive():
        # TODO(phimuell): Handle the case when transient goes into nested SDFG
        #   on the inside it is a non transient, so it is ignored.
        _gt_change_transient_strides_non_recursive_impl(nsdfg)


def _gt_change_transient_strides_non_recursive_impl(
    sdfg: dace.SDFG,
) -> None:
    """Essentially this function just changes the stride to FORTRAN order."""
    for top_level_transient in _find_toplevel_transients(sdfg, only_arrays=True):
        desc: dace_data.Array = sdfg.arrays[top_level_transient]
        ndim = len(desc.shape)
        if ndim <= 1:
            continue
        # We assume that everything is in C order initially, to get FORTRAN order
        #  we simply have to reverse the order.
        new_stride_order = list(range(ndim))
        desc.set_strides_from_layout(*new_stride_order)
        for state in sdfg.states():
            for data_node in state.data_nodes():
                if data_node.data == top_level_transient:
                    for in_edge in state.in_edges(data_node):
                        gt_map_strides_to_nested_sdfg(sdfg, state, in_edge, data_node)


def _find_toplevel_transients(
    sdfg: dace.SDFG,
    only_arrays: bool = False,
) -> set[str]:
    """Find all top level transients in the SDFG.

    The function will scan the SDFG, ignoring nested one, and return the
    name of all transients that have an access node at the top level.
    However, it will ignore access nodes that refers to registers.
    """
    top_level_transients: set[str] = set()
    for state in sdfg.states():
        scope_dict = state.scope_dict()
        for dnode in state.data_nodes():
            data: str = dnode.data
            if scope_dict[dnode] is not None:
                if data in top_level_transients:
                    top_level_transients.remove(data)
                continue
            elif data in top_level_transients:
                continue
            elif gtx_transformations.util.is_view(dnode, sdfg):
                continue
            desc: dace_data.Data = dnode.desc(sdfg)

            if not desc.transient:
                continue
            elif only_arrays and not isinstance(desc, dace_data.Array):
                continue
            top_level_transients.add(data)
    return top_level_transients


def gt_map_strides_to_nested_sdfg(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    edge: dace.sdfg.graph.Edge,
    outer_node: dace.nodes.AccessNode,
) -> None:
    """Propagates the strides of the given data node to the nested SDFGs.

    This function will recursively visit the nested SDFGs connected to the given
    data node and apply mapping from inner to outer strides.

    Args:
        sdfg: The SDFG to process.
        state: The state where the data node is used.
        edge: The edge that writes to the data node.
        outer_node: The data node whose strides should be propagated.
    """
    if isinstance(edge.src, dace.nodes.MapExit):
        # Find the source of the edge entering the map exit node
        map_exit_in_conn = edge.src_conn.replace("OUT_", "IN_")
        for edge_to_map_exit_edge in state.in_edges_by_connector(edge.src, map_exit_in_conn):
            gt_map_strides_to_nested_sdfg(sdfg, state, edge_to_map_exit_edge, outer_node)
        return

    if not isinstance(edge.src, dace.nodes.NestedSDFG):
        return

    # We need to propagate the strides inside the nested SDFG on the global arrays
    nsdfg_node = edge.src
    new_strides = tuple(
        stride
        for stride, to_map_size in zip(
            outer_node.desc(sdfg).strides,
            edge.data.subset.size(),
            strict=True,
        )
        if to_map_size != 1
    )
    inner_data = edge.src_conn
    inner_desc = nsdfg_node.sdfg.arrays[inner_data]
    assert not inner_desc.transient

    if isinstance(inner_desc, dace.data.Scalar):
        assert len(new_strides) == 0
        return

    assert isinstance(inner_desc, dace.data.Array)
    inner_desc.set_shape(inner_desc.shape, new_strides)

    for stride in new_strides:
        if dace.symbolic.issymbolic(stride):
            for sym in stride.free_symbols:
                nsdfg_node.sdfg.add_symbol(str(sym), sym.dtype)
                nsdfg_node.symbol_mapping |= {str(sym): sym}

    for inner_state in nsdfg_node.sdfg.states():
        for inner_node in inner_state.data_nodes():
            if inner_node.data == inner_data:
                for inner_edge in inner_state.in_edges(inner_node):
                    gt_map_strides_to_nested_sdfg(sdfg, state, inner_edge, inner_node)
