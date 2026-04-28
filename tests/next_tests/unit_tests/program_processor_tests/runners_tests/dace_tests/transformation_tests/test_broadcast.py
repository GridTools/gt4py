# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
import copy
import numpy as np

dace = pytest.importorskip("dace")

from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
    library_nodes as gtx_lib_nodes,
)

from . import util


import dace


def _make_broadcast_inline_sdfg() -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.MapEntry
]:
    sdfg = dace.SDFG(util.unique_name("broadcast_inliner"))
    state = sdfg.add_state(is_start_block=True)

    for aname in "abc":
        sdfg.add_array(
            aname,
            shape=(10,),
            dtype=dace.float64,
            transient=(aname == "b"),
        )
    for sname in "dts":
        sdfg.add_scalar(
            sname,
            dtype=dace.float64,
            transient=(sname != "d"),
        )

    # TODO: Ask Edoardo how to properly use; Maybe also modify it.
    bcast_lib = gtx_lib_nodes.Broadcast(name="bcast")

    a, b, c, d, t, s = (state.add_access(name) for name in "abcdts")

    me, mx = state.add_map("map", ndrange={"__i": "1:10"})

    tlet1, tlet2, tlet3 = [
        state.add_tasklet(
            f"tlet{i + 1}",
            inputs={"__in1", "__in2"},
            outputs={"__out"},
            code=f"__out = {op}",
        )
        for i, op in enumerate(["__in1 + __in2", "__in1 - __in1", "2.0 * __in1 + __in2"])
    ]

    state.add_edge(d, None, bcast_lib, "_inp", dace.Memlet("d[0]"))
    state.add_edge(bcast_lib, "_outp", b, None, dace.Memlet("b[0:10]"))
    state.add_edge(b, None, me, "IN_b", dace.Memlet("b[1:10]"))
    me.add_scope_connectors("b")

    state.add_edge(a, None, me, "IN_a", dace.Memlet("a[1:10]"))
    me.add_scope_connectors("a")

    state.add_edge(me, "OUT_a", tlet1, "__in1", dace.Memlet("a[__i]"))
    state.add_edge(me, "OUT_b", tlet1, "__in2", dace.Memlet("b[__i - 1]"))
    state.add_edge(tlet1, "__out", t, None, dace.Memlet("t[0]"))

    state.add_edge(me, "OUT_a", tlet2, "__in1", dace.Memlet("a[__i - 1]"))
    state.add_edge(me, "OUT_b", tlet2, "__in2", dace.Memlet("b[__i]"))
    state.add_edge(tlet2, "__out", s, None, dace.Memlet("s[0]"))

    state.add_edge(s, None, tlet3, "__in1", dace.Memlet("s[0]"))
    state.add_edge(t, None, tlet3, "__in2", dace.Memlet("t[0]"))
    state.add_edge(tlet3, "__out", mx, "IN_c", dace.Memlet("c[__i]"))
    state.add_edge(mx, "OUT_c", c, None, dace.Memlet("c[1:10]"))
    mx.add_scope_connectors("c")

    sdfg.validate()

    return sdfg, state, d, b, me


def test_broadcast():
    sdfg, state, bcast_value, bcast_result, map_entry = _make_broadcast_inline_sdfg()

    assert util.count_nodes(sdfg, gtx_lib_nodes.Broadcast) == 1
    assert bcast_result in util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert bcast_result.data in sdfg.arrays
    assert state.out_degree(map_entry) == 4
    assert not any(iedge.src is bcast_value for iedge in state.in_edges(map_entry))

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    nb_applied = sdfg.apply_transformations_repeated(
        gtx_transformations.ScalarBrodcastInliner, validate_all=True
    )
    assert nb_applied == 1
    assert util.count_nodes(sdfg, gtx_lib_nodes.Broadcast) == 0
    assert bcast_result not in util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert bcast_result.data not in sdfg.arrays
    assert sum([ie.src is bcast_value for ie in state.in_edges(map_entry)]) == 1

    util.compile_and_run_sdfg(sdfg, **res)

    assert util.compare_sdfg_res(ref=ref, res=res)


def _make_indirect_access() -> dace.SDFG:
    sdfg = dace.SDFG(util.unique_name("broadcast_indirect_access"))
    state = sdfg.add_state(is_start_block=True)

    for aname in "abc":
        sdfg.add_array(
            aname,
            shape=(10,),
            dtype=dace.float64,
            transient=(aname == "b"),
        )
    sdfg.add_array(
        "idx",
        shape=(10,),
        dtype=dace.int32,
        transient=False,
    )

    for sname in "dts":
        sdfg.add_scalar(
            sname,
            dtype=dace.float64,
            transient=(sname != "d"),
        )

    # TODO: Ask Edoardo how to properly use; Maybe also modify it.
    bcast_lib = gtx_lib_nodes.Broadcast(name="bcast")
    a, b, c, d, t, s = (state.add_access(name) for name in "abcdts")
    idx = state.add_access("idx")
    me, mx = state.add_map("map", ndrange={"__i": "0:10"})

    tlet1, tlet2 = [
        state.add_tasklet(
            f"tlet{i + 1}",
            inputs={"__in1", "__in2"},
            outputs={"__out"},
            code=f"__out = {op}",
        )
        for i, op in enumerate(["__in1 + __in2", "2.0 * __in1 + __in2"])
    ]
    tlet_idx = state.add_tasklet(
        "tlet_indirect",
        inputs={"__field", "__idx"},
        outputs={"__out"},
        code="__out = __field[__idx]",
    )

    state.add_edge(d, None, bcast_lib, "_inp", dace.Memlet("d[0]"))
    state.add_edge(bcast_lib, "_outp", b, None, dace.Memlet("b[0:10]"))

    state.add_edge(b, None, me, "IN_b", dace.Memlet("b[0:10]"))
    state.add_edge(idx, None, me, "IN_idx", dace.Memlet("idx[0:10]"))
    state.add_edge(a, None, me, "IN_a", dace.Memlet("a[0:10]"))
    me.add_scope_connectors("a")
    me.add_scope_connectors("idx")
    me.add_scope_connectors("b")

    state.add_edge(me, "OUT_b", tlet_idx, "__field", dace.Memlet("b[0:10]"))
    state.add_edge(me, "OUT_idx", tlet_idx, "__idx", dace.Memlet("idx[__i]"))
    state.add_edge(tlet_idx, "__out", t, None, dace.Memlet("t[0]"))

    state.add_edge(me, "OUT_a", tlet1, "__in1", dace.Memlet("a[__i]"))
    state.add_edge(t, None, tlet1, "__in2", dace.Memlet("t[0]"))
    state.add_edge(tlet1, "__out", s, None, dace.Memlet("s[0]"))

    state.add_edge(s, None, tlet2, "__in1", dace.Memlet("s[0]"))
    state.add_edge(me, "OUT_b", tlet2, "__in2", dace.Memlet("b[__i]"))
    state.add_edge(tlet2, "__out", mx, "IN_c", dace.Memlet("c[__i]"))
    state.add_edge(mx, "OUT_c", c, None, dace.Memlet("c[0:10]"))
    mx.add_scope_connectors("c")

    sdfg.validate()

    return sdfg


def test_indirect_access_broadcast():
    sdfg = _make_indirect_access()

    # NOTE: This pattern could be processed, however, we would need to inspect the
    #   Tasklet to make sure that it is indeed an indirect access. The safest way
    #   to do it would be to add another Library node for it.
    nb_applied = sdfg.apply_transformations_repeated(
        gtx_transformations.ScalarBrodcastInliner, validate_all=True
    )
    assert nb_applied == 0
