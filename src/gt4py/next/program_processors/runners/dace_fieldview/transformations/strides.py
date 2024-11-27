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
        either use FORTRAN order for GPU or C order (which is assumed to b the
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
