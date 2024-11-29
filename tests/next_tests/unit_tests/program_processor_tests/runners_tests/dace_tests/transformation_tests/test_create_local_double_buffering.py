# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np
import copy

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _create_sdfg_double_read_part_1(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    me: dace.nodes.MapEntry,
    mx: dace.nodes.MapExit,
    A_in: dace.nodes.AccessNode,
    nb: int,
) -> dace.nodes.Tasklet:
    tskl = state.add_tasklet(
        name=f"tasklet_1", inputs={"__in1"}, outputs={"__out"}, code="__out = __in1 + 1.0"
    )

    state.add_edge(A_in, None, me, f"IN_{nb}", dace.Memlet("A[0:10]"))
    state.add_edge(me, f"OUT_{nb}", tskl, "__in1", dace.Memlet("A[__i0]"))
    me.add_in_connector(f"IN_{nb}")
    me.add_out_connector(f"OUT_{nb}")

    state.add_edge(tskl, "__out", mx, f"IN_{nb}", dace.Memlet("A[__i0]"))
    state.add_edge(mx, f"OUT_{nb}", state.add_access("A"), None, dace.Memlet("A[0:10]"))
    mx.add_in_connector(f"IN_{nb}")
    mx.add_out_connector(f"OUT_{nb}")


def _create_sdfg_double_read_part_2(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    me: dace.nodes.MapEntry,
    mx: dace.nodes.MapExit,
    A_in: dace.nodes.AccessNode,
    nb: int,
) -> dace.nodes.Tasklet:
    tskl = state.add_tasklet(
        name=f"tasklet_2", inputs={"__in1"}, outputs={"__out"}, code="__out = __in1 + 3.0"
    )

    state.add_edge(A_in, None, me, f"IN_{nb}", dace.Memlet("A[0:10]"))
    state.add_edge(me, f"OUT_{nb}", tskl, "__in1", dace.Memlet("A[__i0]"))
    me.add_in_connector(f"IN_{nb}")
    me.add_out_connector(f"OUT_{nb}")

    state.add_edge(tskl, "__out", mx, f"IN_{nb}", dace.Memlet("B[__i0]"))
    state.add_edge(mx, f"OUT_{nb}", state.add_access("B"), None, dace.Memlet("B[0:10]"))
    mx.add_in_connector(f"IN_{nb}")
    mx.add_out_connector(f"OUT_{nb}")


def _create_sdfg_double_read(
    version: int,
) -> tuple[dace.SDFG]:
    sdfg = dace.SDFG(util.unique_name(f"double_read_version_{version}"))
    state = sdfg.add_state(is_start_block=True)
    for name in "AB":
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )
    A_in = state.add_access("A")
    me, mx = state.add_map("map", ndrange={"__i0": "0:10"})

    if version == 0:
        _create_sdfg_double_read_part_1(sdfg, state, me, mx, A_in, 0)
        _create_sdfg_double_read_part_2(sdfg, state, me, mx, A_in, 1)
    elif version == 1:
        _create_sdfg_double_read_part_1(sdfg, state, me, mx, A_in, 1)
        _create_sdfg_double_read_part_2(sdfg, state, me, mx, A_in, 0)
    else:
        raise ValueError(f"Does not know version {version}")
    sdfg.validate()
    return sdfg


def test_double_read_sdfg():
    sdfg0 = _create_sdfg_double_read(0)
    sdfg1 = _create_sdfg_double_read(1)
    args0 = {name: np.array(np.random.rand(10), dtype=np.float64, copy=True) for name in "AB"}
    args1 = copy.deepcopy(args0)

    count0 = gtx_transformations.gt_crearte_local_double_buffering(sdfg0)
    assert count0 == 1

    count1 = gtx_transformations.gt_crearte_local_double_buffering(sdfg1)
    assert count1 == 1

    sdfg0(**args0)
    sdfg1(**args1)
    for name in args0:
        assert np.allclose(args0[name], args1[name]), f"Failed verification in '{name}'."
