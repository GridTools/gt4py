# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util
from .test_move_dataflow_into_if_body import _make_if_block

import dace

def _make_map_with_conditional_blocks() -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.MapEntry, dace_nodes.MapExit
]:
    sdfg = dace.SDFG(util.unique_name("map_with_conditional_blocks"))
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_array(
        "a",
        shape=(10,),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "b",
        shape=(10,),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "c",
        shape=(10,),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "d",
        shape=(10,),
        dtype=dace.float64,
        transient=False,
    )
    a, b, c, d = (state.add_access(name) for name in "abcd")

    for tmp_name in ["tmp_a", "tmp_b", "tmp_c", "tmp_d"]:
        sdfg.add_scalar(tmp_name, dtype=dace.float64, transient=True)
    tmp_a, tmp_b, tmp_c, tmp_d = (state.add_access(f"tmp_{name}") for name in "abcd")

    sdfg.add_scalar("cond_var", dtype=dace.bool_, transient=True)
    cond_var = state.add_access("cond_var")

    me, mx = state.add_map("map_with_ifs", ndrange={"__i": "0:10"})
    me.add_in_connector("IN_a")
    me.add_out_connector("OUT_a")
    me.add_in_connector("IN_b")
    me.add_out_connector("OUT_b")
    mx.add_in_connector("IN_c")
    mx.add_out_connector("OUT_c")
    mx.add_in_connector("IN_d")
    mx.add_out_connector("OUT_d")
    state.add_edge(a, None, me, "IN_a", dace.Memlet("a[__i]"))
    state.add_edge(b, None, me, "IN_b", dace.Memlet("b[__i]"))
    state.add_edge(me, "OUT_a", tmp_a, None, dace.Memlet("a[__i]"))
    state.add_edge(me, "OUT_b", tmp_b, None, dace.Memlet("b[__i]"))

    tasklet_cond = state.add_tasklet(
        "tasklet_cond",
        inputs={"__in"},
        outputs={"__out"},
        code="__out = __in <= 0.0",
    )
    state.add_edge(tmp_a, None, tasklet_cond, "__in", dace.Memlet("tmp_a[0]"))
    state.add_edge(tasklet_cond, "__out", cond_var, None, dace.Memlet("cond_var"))

    if_block_0 = _make_if_block(state=state, outer_sdfg=sdfg)
    state.add_edge(cond_var, None, if_block_0, "__cond", dace.Memlet("cond_var"))
    state.add_edge(tmp_a, None, if_block_0, "__arg1", dace.Memlet("tmp_a[0]"))
    state.add_edge(tmp_b, None, if_block_0, "__arg2", dace.Memlet("tmp_b[0]"))
    state.add_edge(if_block_0, "__output", tmp_c, None, dace.Memlet("tmp_c[0]"))
    state.add_edge(tmp_c, None, mx, "IN_c", dace.Memlet("c[__i]"))

    if_block_1 = _make_if_block(state=state, outer_sdfg=sdfg)
    state.add_edge(cond_var, None, if_block_1, "__cond", dace.Memlet("cond_var"))
    state.add_edge(tmp_a, None, if_block_1, "__arg1", dace.Memlet("tmp_a[0]"))
    state.add_edge(tmp_b, None, if_block_1, "__arg2", dace.Memlet("tmp_b[0]"))
    state.add_edge(if_block_1, "__output", tmp_d, None, dace.Memlet("tmp_d[0]"))
    state.add_edge(tmp_d, None, mx, "IN_d", dace.Memlet("d[__i]"))

    state.add_edge(mx, "OUT_c", c, None, dace.Memlet("c[__i]"))
    state.add_edge(mx, "OUT_d", d, None, dace.Memlet("d[__i]"))

    sdfg.validate()
    return sdfg, state, me, mx

def test_fuse_horizontal_condition_blocks():
    sdfg, state, me, mx = _make_map_with_conditional_blocks()

    # sdfg.view()
    # breakpoint()

    sdfg.apply_transformations_repeated(
        gtx_transformations.FuseHorizontalConditionBlocks(),
        validate=True,
        validate_all=True,
    )

    # sdfg.view()
    # breakpoint()
