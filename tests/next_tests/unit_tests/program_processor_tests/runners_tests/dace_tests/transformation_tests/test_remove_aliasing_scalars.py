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

import dace


def _make_map_with_scalar_copies() -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.MapEntry, dace_nodes.MapExit
]:
    sdfg = dace.SDFG(gtx_transformations.utils.unique_name("scalar_elimination"))
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
        transient=True,
    )

    a, b = (state.add_access(name) for name in "ab")
    for i in range(3):
        sdfg.add_scalar(f"tmp{i}", dtype=dace.float64, transient=True)
    tmp0, tmp1, tmp2 = (state.add_access(f"tmp{i}") for i in range(3))

    me, mx = state.add_map("copy_map", ndrange={"__i": "0:10"})
    me.add_in_connector("IN_a")
    me.add_out_connector("OUT_a")
    mx.add_in_connector("IN_b")
    mx.add_out_connector("OUT_b")
    state.add_edge(a, None, me, "IN_a", dace.Memlet("a[__i]"))
    state.add_edge(me, "OUT_a", tmp0, None, dace.Memlet("a[__i]"))
    state.add_edge(tmp0, None, tmp1, None, dace.Memlet("tmp1[0]"))
    state.add_edge(tmp1, None, tmp2, None, dace.Memlet("tmp1[0]"))
    state.add_edge(tmp2, None, mx, "IN_b", dace.Memlet("[0] -> b[__i]"))
    state.add_edge(mx, "OUT_b", b, None, dace.Memlet("b[__i]"))

    sdfg.validate()
    return sdfg, state, me, mx


def test_remove_double_write_single_consumer():
    sdfg, state, me, mx = _make_map_with_scalar_copies()

    access_nodes_inside_original_map = util.count_nodes(
        state.scope_subgraph(me, include_entry=False, include_exit=False), dace_nodes.AccessNode
    )
    assert access_nodes_inside_original_map == 3

    find_single_use_data = dace_analysis.FindSingleUseData()
    single_use_data = find_single_use_data.apply_pass(sdfg, None)
    sdfg.apply_transformations_repeated(
        gtx_transformations.RemoveAliasingScalars(
            single_use_data=single_use_data,
            assume_single_use_data=False,
        ),
        validate=True,
        validate_all=True,
    )

    access_nodes_inside_new_map = util.count_nodes(
        state.scope_subgraph(me, include_entry=False, include_exit=False), dace_nodes.AccessNode
    )
    assert access_nodes_inside_new_map == 1
