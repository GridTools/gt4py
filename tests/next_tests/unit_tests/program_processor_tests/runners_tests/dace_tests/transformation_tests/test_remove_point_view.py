# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import copy

dace = pytest.importorskip("dace")

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)
from dace.sdfg import nodes as dace_nodes, propagation as dace_propagation
from sympy.core.numbers import Number

from . import util


def _make_sdfg_with_map_with_view(
    N: str | int,
) -> dace.SDFG:
    shape = (N, N)
    sdfg = dace.SDFG(util.unique_name("multiple_maps"))
    state = sdfg.add_state(is_start_block=True)

    for name in ["a", "out"]:
        sdfg.add_array(
            name=name,
            shape=shape,
            dtype=dace.float64,
            transient=False,
        )

    a, out = (
        state.add_access(name) for name in ["a", "out"]
    )
    sdfg.add_symbol("horizontal_start", dace.int32)
    sdfg.add_symbol("horizontal_end", dace.int32)

    # First independent Tasklet
    task1 = state.add_tasklet(
        "task1",
        inputs={
            "__in0",  # <- `b[i, j]`
        },
        outputs={
            "__out0",  # <- `tmp1`
        },
        code="__out0 = __in0 + 3.0",
    )

    # Now create the map using the above tasklets
    mentry, mexit = state.add_map(
        "simple_map",
        ndrange={"i": f"0:{N}", "j": "horizontal_start:horizontal_end"},
    )

    tmp_access_node = state.add_access("tmp")
    tmp_access_node.transient = True
    
    sdfg.add_view(
        "tmp",
        shape=(1,),
        dtype=dace.float64,
    )

    # tmp2 = sdfg.add_scalar("tmp2", dtype=dace.float64)

    tmp_access_node2 = state.add_transient("tmp2", shape=(1,), dtype=dace.float64)

    
    state.add_edge(a, None, mentry, "IN_a", dace.Memlet("a[i, j]"))
    # Now assemble everything.
    state.add_edge(mentry, "OUT_a", task1, "__in0", dace.Memlet("a[i, j]"))
    state.add_edge(task1, "__out0", tmp_access_node, None, dace.Memlet("tmp[0]"))
    state.add_edge(tmp_access_node, None, tmp_access_node2, None, dace.Memlet("tmp[0]"))
    state.add_edge(tmp_access_node2, None, mexit, "IN_out", dace.Memlet("tmp2[0]"))


    mentry.add_scope_connectors("a")
    mentry.add_in_connector("IN_a")
    mexit.add_in_connector("IN_out")
    mexit.add_out_connector("OUT_out")
    state.add_edge(mexit, "OUT_out", out, None, dace.Memlet("out[i, j]"))

    dace_propagation.propagate_states(sdfg)
    sdfg.validate()

    return sdfg


def test_horizontal_map_fusion():
    N = 20
    sdfg = _make_sdfg_with_map_with_view(N)

    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 4


    sdfg.apply_transformations_once_everywhere(
        gtx_transformations.RemovePointwiseViews,
        validate=True,
        validate_all=True,
    )

    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 1
    assert util.count_nodes(sdfg, dace_nodes.MapExit) == 1
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 3

