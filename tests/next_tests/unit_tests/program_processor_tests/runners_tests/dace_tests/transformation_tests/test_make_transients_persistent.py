# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import dace
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _make_transients_persistent_inner_access_sdfg() -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("transients_persistent_inner_access_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abc":
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["b"].transient = True

    me: dace_nodes.MapEntry
    mx: dace_nodes.MapExit
    me, mx = state.add_map("comp", ndrange={"__i0": "0:10"})
    a, b, c = (state.add_access(name) for name in "abc")
    tsklt: dace_nodes.Tasklet = state.add_tasklet(
        "tsklt",
        inputs={"__in"},
        code="__out = __in + 1.0",
        outputs={"__out"},
    )

    me.add_in_connector("IN_A")
    state.add_edge(a, None, me, "IN_A", dace.Memlet("a[0:10]"))

    me.add_out_connector("OUT_A")
    state.add_edge(me, "OUT_A", b, None, dace.Memlet("a[__i0] -> [__i0]"))

    state.add_edge(b, None, tsklt, "__in", dace.Memlet("b[__i0]"))

    mx.add_in_connector("IN_C")
    state.add_edge(tsklt, "__out", mx, "IN_C", dace.Memlet("c[__i0]"))

    mx.add_out_connector("OUT_C")
    state.add_edge(mx, "OUT_C", c, None, dace.Memlet("c[0:10]"))
    sdfg.validate()
    return sdfg, state


def test_make_transients_persistent_inner_access():
    sdfg, state = _make_transients_persistent_inner_access_sdfg()
    assert sdfg.arrays["b"].lifetime is dace.dtypes.AllocationLifetime.Scope

    # Because `b`, the only transient, is used inside a map scope, it is not selected,
    #  although in this situation it would be possible.
    change_report: dict[int, set[str]] = gtx_transformations.gt_make_transients_persistent(
        sdfg, device=dace.DeviceType.CPU
    )
    assert len(change_report) == 1
    assert change_report[sdfg.cfg_id] == set()
