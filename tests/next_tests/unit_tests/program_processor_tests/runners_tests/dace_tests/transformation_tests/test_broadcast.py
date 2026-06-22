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

from dace import data as dace_data, subsets as dace_sbs
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
    library_nodes as gtx_lib_nodes,
)

from . import util


def _make_broadcast_map_substititution(
    multi_edge: bool,
) -> tuple[dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.MapEntry]:
    sdfg = dace.SDFG(gtx_transformations.utils.unique_name("broadcast_inliner"))
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
    bcast_lib = gtx_lib_nodes.Broadcast(name="bcast", broadcast_in_dims=[], params=None)

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

    if multi_edge:
        bcast_res_conn1 = "b1"
        bcast_res_conn2 = "b2"
    else:
        bcast_res_conn1 = "b"
        bcast_res_conn2 = bcast_res_conn1

    for conn in {bcast_res_conn1, bcast_res_conn2}:
        state.add_edge(b, None, me, "IN_" + conn, dace.Memlet("b[1:10]"))
        me.add_scope_connectors(conn)

    state.add_edge(a, None, me, "IN_a", dace.Memlet("a[1:10]"))
    me.add_scope_connectors("a")

    state.add_edge(me, "OUT_a", tlet1, "__in1", dace.Memlet("a[__i]"))
    state.add_edge(me, "OUT_" + bcast_res_conn1, tlet1, "__in2", dace.Memlet("b[__i - 1]"))
    state.add_edge(tlet1, "__out", t, None, dace.Memlet("t[0]"))

    state.add_edge(me, "OUT_a", tlet2, "__in1", dace.Memlet("a[__i - 1]"))
    state.add_edge(me, "OUT_" + bcast_res_conn2, tlet2, "__in2", dace.Memlet("b[__i]"))
    state.add_edge(tlet2, "__out", s, None, dace.Memlet("s[0]"))

    state.add_edge(s, None, tlet3, "__in1", dace.Memlet("s[0]"))
    state.add_edge(t, None, tlet3, "__in2", dace.Memlet("t[0]"))
    state.add_edge(tlet3, "__out", mx, "IN_c", dace.Memlet("c[__i]"))
    state.add_edge(mx, "OUT_c", c, None, dace.Memlet("c[1:10]"))
    mx.add_scope_connectors("c")

    sdfg.validate()

    return sdfg, state, d, b, me


@pytest.mark.parametrize("multi_edge", [True, False])
def test_map_replacement(
    multi_edge: bool,
):
    sdfg, state, bcast_value, bcast_result, map_entry = _make_broadcast_map_substititution(
        multi_edge=multi_edge
    )

    assert util.count_nodes(sdfg, gtx_lib_nodes.Broadcast) == 1
    assert bcast_result in util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert bcast_result.data in sdfg.arrays
    assert state.out_degree(map_entry) == 4

    if multi_edge:
        assert len(map_entry.out_connectors) == 3
    else:
        assert len(map_entry.out_connectors) == 2

    assert not any(iedge.src is bcast_value for iedge in state.in_edges(map_entry))

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    nb_applied = sdfg.apply_transformations_repeated(
        gtx_transformations.InlineBroadcastAccess, validate_all=True
    )
    assert nb_applied == 1
    assert util.count_nodes(sdfg, gtx_lib_nodes.Broadcast) == 0
    assert bcast_result not in util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert bcast_result.data not in sdfg.arrays

    # We only have one connection between the `bcast_value` node and the MapEntry
    #  because the transformation performs duplication.
    assert sum([ie.src is bcast_value for ie in state.in_edges(map_entry)]) == 1

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)


def _make_indirect_access() -> dace.SDFG:
    sdfg = dace.SDFG(gtx_transformations.utils.unique_name("broadcast_indirect_access"))
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
    bcast_lib = gtx_lib_nodes.Broadcast(name="bcast", broadcast_in_dims=[], params=None)
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
        gtx_transformations.InlineBroadcastAccess, validate_all=True
    )
    assert nb_applied == 0


def _make_access_node_chain() -> tuple[
    dace.SDFG, dace.SDFGState, gtx_lib_nodes.Broadcast, dace_nodes.AccessNode
]:
    sdfg = dace.SDFG(gtx_transformations.utils.unique_name("_broadcast_access_node_chain"))
    state = sdfg.add_state(is_start_block=True)

    for aname in "abc":
        sdfg.add_array(
            aname,
            shape=(10,),
            dtype=dace.float64,
            transient=(aname != "c"),
        )

    sdfg.add_scalar(
        "s",
        dtype=dace.float64,
        transient=False,
    )

    a, b, c, s = (state.add_access(name) for name in "abcs")

    bcast_lib = gtx_lib_nodes.Broadcast(name="bcast", broadcast_in_dims=[], params=None)

    state.add_edge(s, None, bcast_lib, "_inp", dace.Memlet("s[0]"))
    state.add_edge(bcast_lib, "_outp", a, None, dace.Memlet("a[1:10]"))
    state.add_nedge(a, b, dace.Memlet("a[2:9] -> [0:7]"))
    state.add_nedge(b, c, dace.Memlet("c[1:5] -> [2:6]"))

    sdfg.validate()

    return sdfg, state, bcast_lib, c


def test_access_node_chain():
    sdfg, state, bcast_lib, c = _make_access_node_chain()

    ac_before = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert len(ac_before) == 4
    assert c in ac_before
    assert isinstance(sdfg.arrays["s"], dace_data.Scalar)
    assert c.data in sdfg.arrays
    assert list(util.count_nodes(sdfg, gtx_lib_nodes.Broadcast, True)) == [bcast_lib]
    assert state.in_degree(c) == 1
    assert all(e.src is not bcast_lib for e in state.in_edges(c))

    nb_applied = sdfg.apply_transformations_repeated(
        gtx_transformations.InlineBroadcastAccess, validate_all=True
    )
    assert nb_applied == 2

    ac_before = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert len(ac_before) == 2
    assert c in ac_before
    assert isinstance(sdfg.arrays["s"], dace_data.Scalar)
    assert c.data in sdfg.arrays

    # In the AccessNode mode the broadcast node is copied. In this case it is not
    #  needed, but it is needed in more general cases.
    # TODO(phimuell): Once the node is finalized, check if it is copied correctly.
    bcast_libs_after = list(util.count_nodes(sdfg, gtx_lib_nodes.Broadcast, True))
    assert len(bcast_libs_after) == 1
    assert bcast_libs_after[0] is not bcast_lib

    bcast_edge = next(iter(state.in_edges(c)))
    assert state.in_degree(c) == 1
    assert bcast_edge.src is bcast_libs_after[0]
    assert bcast_edge.data.dst_subset == dace_sbs.Range.from_string("1:5")


def _make_access_node_fan_out() -> tuple[dace.SDFG, dace.SDFGState, dict[str, str]]:
    sdfg = dace.SDFG(gtx_transformations.utils.unique_name("_broadcast_access_node_fan_out"))
    state = sdfg.add_state(is_start_block=True)

    bcast_result_name = "bcast_result"
    res_names = {
        "res1": "1:10",
        "res2": "2:8",
    }
    for aname in [bcast_result_name] + list(res_names.keys()):
        sdfg.add_array(
            aname,
            shape=(10,),
            dtype=dace.float64,
            transient=(not aname.startswith("res")),
        )

    sdfg.add_scalar(
        "s",
        dtype=dace.float64,
        transient=False,
    )

    bcast_result = state.add_access(bcast_result_name)
    s = state.add_access("s")
    bcast_lib = gtx_lib_nodes.Broadcast(name="bcast", broadcast_in_dims=[], params=None)

    state.add_edge(s, None, bcast_lib, "_inp", dace.Memlet("s[0]"))
    state.add_edge(bcast_lib, "_outp", bcast_result, None, dace.Memlet(f"{bcast_result}[0:10]"))

    for dname, sbs in res_names.items():
        state.add_nedge(bcast_result, state.add_access(dname), dace.Memlet(f"{dname}[{sbs}]"))

    return sdfg, state, res_names


def test_access_node_fan_out():
    sdfg, state, res_names = _make_access_node_fan_out()

    assert util.count_nodes(sdfg, gtx_lib_nodes.Broadcast) == 1
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == (len(res_names) + 2)
    assert all(res_name in sdfg.arrays for res_name in res_names)

    nb_applied = sdfg.apply_transformations_repeated(
        gtx_transformations.InlineBroadcastAccess, validate_all=True
    )
    assert nb_applied == 1

    # The broadcast nodes are replicated in the AccessNode mode, one for each output.
    bcast_libs_after = set(util.count_nodes(sdfg, gtx_lib_nodes.Broadcast, True))
    assert len(bcast_libs_after) == len(res_names)

    ac_after = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert len(ac_after) == (1 + len(res_names))

    for res_ac in ac_after:
        if res_ac.data == "s":
            assert res_ac.data in sdfg.arrays
            assert isinstance(sdfg.arrays[res_ac.data], dace_data.Scalar)
        else:
            assert res_ac.data in res_names
            state.in_degree(res_ac) == 1
            edge = next(iter(state.in_edges(res_ac)))
            assert edge.src in bcast_libs_after
            assert edge.data.dst_subset == dace_sbs.Range.from_string(res_names[res_ac.data])
            bcast_libs_after.remove(edge.src)


def _make_access_node_multi_connection():
    sdfg = dace.SDFG(
        gtx_transformations.utils.unique_name("_broadcast_access_node_multi_connection")
    )
    state = sdfg.add_state(is_start_block=True)

    for aname in "ab":
        sdfg.add_array(
            aname,
            shape=(10,),
            dtype=dace.float64,
            transient=(aname == "a"),
        )

    sdfg.add_scalar(
        "s",
        dtype=dace.float64,
        transient=False,
    )

    a, b, s = (state.add_access(name) for name in "abs")
    bcast_lib = gtx_lib_nodes.Broadcast(name="bcast", broadcast_in_dims=[], params=None)

    state.add_edge(s, None, bcast_lib, "_inp", dace.Memlet("s[0]"))
    state.add_edge(bcast_lib, "_outp", a, None, dace.Memlet("a[1:10]"))

    state.add_nedge(a, b, dace.Memlet("a[2:5] -> [1:4]"))
    state.add_nedge(a, b, dace.Memlet("b[6:9] -> [4:7]"))

    return sdfg, state, b


def test_access_node_multi_connection():
    sdfg, state, b = _make_access_node_multi_connection()

    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 3
    assert state.in_degree(b) == 2
    bcast_libs_before = list(util.count_nodes(sdfg, gtx_lib_nodes.Broadcast, True))
    assert len(bcast_libs_before) == 1

    nb_applied = sdfg.apply_transformations_repeated(
        gtx_transformations.InlineBroadcastAccess, validate_all=True
    )
    assert nb_applied == 1

    # The broadcast library nodes are replicated and the original one is then removed.
    bcast_libs_after = set(util.count_nodes(sdfg, gtx_lib_nodes.Broadcast, True))
    assert len(bcast_libs_after) == 2
    assert bcast_libs_before[0] not in bcast_libs_after

    ac_after = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert len(ac_after) == 2
    assert b in ac_after
    assert state.in_degree(b) == 2
    assert bcast_libs_after == {e.src for e in state.in_edges(b)}

    expected_sbs = {
        dace_sbs.Range.from_string("1:4"),
        dace_sbs.Range.from_string("6:9"),
    }
    assert expected_sbs == {e.data.dst_subset for e in state.in_edges(b)}
