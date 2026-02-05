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
import dace
from dace.sdfg import nodes as dace_nodes, graph as dace_graph
from dace import data as dace_data, subsets as dace_sbs
from dace.transformation import dataflow as dace_dftrafo

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


def _make_concat_where_different_inputs() -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.MapEntry
]:
    sdfg = dace.SDFG(util.unique_name("concat_where_replacer_different_inputs"))
    state = sdfg.add_state()
    for name in "abcde":
        sdfg.add_array(
            name=name,
            shape=(10,),
            dtype=dace.float64,
            transient=(name == "c"),
        )
    a, b, c, d, e = (state.add_access(name) for name in "abcde")

    state.add_nedge(a, c, dace.Memlet("a[1:6] -> [0:5]"))
    state.add_nedge(b, c, dace.Memlet("b[3:8] -> [5:10]"))

    _, me, _ = state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in0": dace.Memlet("c[__i]"),
            "__in1": dace.Memlet("e[__i]"),
        },
        code="__out2 = __in1 + __in0",
        outputs={"__out2": dace.Memlet("d[__i]")},
        input_nodes={c, e},
        output_nodes={d},
        external_edges=True,
        schedule=dace.dtypes.ScheduleType.Sequential,
    )
    sdfg.validate()

    return sdfg, state, c, me


def test_concat_where_different_inputs():
    sdfg, state, concat_node, me = _make_concat_where_different_inputs()

    access_nodes_before = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(access_nodes_before) == 5
    assert concat_node.data == "c"
    assert [ac for ac in access_nodes_before if ac.desc(sdfg).transient] == [concat_node]
    assert util.count_nodes(state, dace_nodes.Tasklet) == 1
    assert state.in_degree(me) == 2

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    gtx_transformations.gt_replace_concat_where_node(
        state=state,
        sdfg=sdfg,
        concat_node=concat_node,
    )
    sdfg.validate()

    access_nodes_after = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(access_nodes_after) == 5
    assert concat_node not in access_nodes_after
    assert concat_node.data not in sdfg.arrays
    assert all(
        ac.data.startswith("__gt4py_concat_where_mapper_temp_c_")
        for ac in access_nodes_after
        if ac.desc(sdfg).transient
    )
    assert util.count_nodes(state, dace_nodes.Tasklet) == 2
    assert state.in_degree(me) == 3

    csdfg = util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)


def _make_concat_where_same_inputs() -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.MapEntry, dace_nodes.AccessNode
]:
    sdfg = dace.SDFG(util.unique_name("concat_where_replacer_same_inputs"))
    state = sdfg.add_state()
    for name in "abcd":
        sdfg.add_array(
            name=name,
            shape=(10,),
            dtype=dace.float64,
            transient=(name == "c"),
        )
    a, b, c, d = (state.add_access(name) for name in "abcd")

    state.add_nedge(a, c, dace.Memlet("a[1:6] -> [0:5]"))
    state.add_nedge(a, c, dace.Memlet("a[3:8] -> [5:10]"))

    _, me, _ = state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in0": dace.Memlet("c[__i]"),
            "__in1": dace.Memlet("b[__i]"),
        },
        code="__out2 = __in1 + __in0",
        outputs={"__out2": dace.Memlet("d[__i]")},
        input_nodes={c, b},
        output_nodes={d},
        external_edges=True,
        schedule=dace.dtypes.ScheduleType.Sequential,
    )
    sdfg.validate()

    return sdfg, state, c, me, a


def test_concat_where_same_inputs():
    sdfg, state, concat_node, me, a = _make_concat_where_same_inputs()

    access_nodes_before = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(access_nodes_before) == 4
    assert a in access_nodes_before
    assert state.out_degree(a) == 2
    assert all(oedge.dst is concat_node for oedge in state.out_edges(a))
    assert concat_node.data == "c"
    assert [ac for ac in access_nodes_before if ac.desc(sdfg).transient] == [concat_node]
    assert util.count_nodes(state, dace_nodes.Tasklet) == 1
    assert state.in_degree(me) == 2

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    gtx_transformations.gt_replace_concat_where_node(
        state=state,
        sdfg=sdfg,
        concat_node=concat_node,
    )
    sdfg.validate()

    access_nodes_after = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(access_nodes_after) == 4
    assert a in access_nodes_after
    assert concat_node not in access_nodes_after
    assert concat_node.data not in sdfg.arrays
    assert all(
        ac.data.startswith("__gt4py_concat_where_mapper_temp_c_")
        for ac in access_nodes_after
        if ac.desc(sdfg).transient
    )
    assert util.count_nodes(state, dace_nodes.Tasklet) == 2
    assert state.in_degree(me) == 2

    assert state.out_degree(a) == 1  # Automatic compaction.
    assert all(oedge.dst is me for oedge in state.out_edges(a))
    a_to_me_edge = next(iter(state.out_edges(a)))
    assert a_to_me_edge.dst_conn.startswith("IN_")
    inner_map_a_edges = list(state.out_edges_by_connector(me, "OUT_" + a_to_me_edge.dst_conn[3:]))
    assert len(inner_map_a_edges) == 2

    csdfg = util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)


def _make_concat_where_mixed_inputs() -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.MapEntry, dace_nodes.AccessNode
]:
    sdfg = dace.SDFG(util.unique_name("concat_where_replacer_mixed_inputs"))
    state = sdfg.add_state()
    for name in "abcd":
        sdfg.add_array(
            name=name,
            shape=(15,),
            dtype=dace.float64,
            transient=(name == "c"),
        )
    a, b, c, d = (state.add_access(name) for name in "abcd")

    state.add_nedge(a, c, dace.Memlet("a[1:6] -> [0:5]"))
    state.add_nedge(b, c, dace.Memlet("b[9:14] -> [5:10]"))
    state.add_nedge(a, c, dace.Memlet("a[3:8] -> [10:15]"))

    _, me, _ = state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i": "0:15"},
        inputs={
            "__in0": dace.Memlet("a[__i]"),
            "__in1": dace.Memlet("c[__i]"),
        },
        code="__out2 = __in1 + __in0",
        outputs={"__out2": dace.Memlet("d[__i]")},
        input_nodes={c, a},
        output_nodes={d},
        external_edges=True,
        schedule=dace.dtypes.ScheduleType.Sequential,
    )
    sdfg.validate()

    return sdfg, state, c, me, a


def test_concat_where_mixed_inputs():
    sdfg, state, concat_node, me, a = _make_concat_where_mixed_inputs()

    access_nodes_before = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(access_nodes_before) == 4
    assert a in access_nodes_before
    assert state.out_degree(a) == 3
    assert {oedge.dst for oedge in state.out_edges(a)} == {me, concat_node}
    assert concat_node.data == "c"
    assert state.in_degree(concat_node) == 3
    assert [ac for ac in access_nodes_before if ac.desc(sdfg).transient] == [concat_node]
    assert util.count_nodes(state, dace_nodes.Tasklet) == 1
    assert state.in_degree(me) == 2
    assert {
        iedge.src.data
        for iedge in state.in_edges(me)
        if isinstance(iedge.src, dace_nodes.AccessNode)
    } == {"c", "a"}

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    gtx_transformations.gt_replace_concat_where_node(
        state=state,
        sdfg=sdfg,
        concat_node=concat_node,
    )
    sdfg.validate()

    access_nodes_after = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(access_nodes_after) == 4
    assert a in access_nodes_after
    assert concat_node not in access_nodes_after
    assert concat_node.data not in sdfg.arrays
    assert all(
        ac.data.startswith("__gt4py_concat_where_mapper_temp_c_")
        for ac in access_nodes_after
        if ac.desc(sdfg).transient
    )
    assert util.count_nodes(state, dace_nodes.Tasklet) == 2
    assert state.in_degree(me) == 2  # Compaction and reusing of connections.
    assert {
        iedge.src.data
        for iedge in state.in_edges(me)
        if isinstance(iedge.src, dace_nodes.AccessNode)
    } == {"a", "b"}

    assert state.out_degree(a) == 1  # Automatic compaction.
    assert all(oedge.dst is me for oedge in state.out_edges(a))
    a_to_me_edge = next(iter(state.out_edges(a)))
    assert a_to_me_edge.dst_conn.startswith("IN_")
    inner_map_a_edges = list(state.out_edges_by_connector(me, "OUT_" + a_to_me_edge.dst_conn[3:]))
    assert len(inner_map_a_edges) == 3
    assert len({e.dst for e in inner_map_a_edges if isinstance(e.dst, dace_nodes.Tasklet)}) == 2

    csdfg = util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)


def _make_concat_where_no_mapping_reuse() -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.MapEntry, dace_nodes.AccessNode
]:
    sdfg = dace.SDFG(util.unique_name("concat_where_replacer_no_mapping_reuse"))
    state = sdfg.add_state()
    for name in "abcd":
        sdfg.add_array(
            name=name,
            shape=(10,),
            dtype=dace.float64,
            transient=(name == "c"),
        )
    a, b, c, d = (state.add_access(name) for name in "abcd")

    state.add_nedge(a, c, dace.Memlet("a[1:6] -> [0:5]"))
    state.add_nedge(b, c, dace.Memlet("b[3:8] -> [5:10]"))

    _, me, _ = state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in0": dace.Memlet("c[__i]"),
            "__in1": dace.Memlet("a[2]"),
        },
        code="__out2 = __in1 + __in0",
        outputs={"__out2": dace.Memlet("d[__i]")},
        input_nodes={c, a},
        output_nodes={d},
        external_edges=True,
        schedule=dace.dtypes.ScheduleType.Sequential,
    )
    sdfg.validate()

    return sdfg, state, c, me, a


def test_concat_where_additional_mapping_needed():
    sdfg, state, concat_node, me, a = _make_concat_where_no_mapping_reuse()

    access_nodes_before = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(access_nodes_before) == 4
    assert concat_node.data == "c"
    assert [ac for ac in access_nodes_before if ac.desc(sdfg).transient] == [concat_node]
    assert util.count_nodes(state, dace_nodes.Tasklet) == 1
    assert state.in_degree(me) == 2
    assert state.out_degree(a) == 2
    assert {iedge.src for iedge in state.in_edges(me)} == {a, concat_node}

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    gtx_transformations.gt_replace_concat_where_node(
        state=state,
        sdfg=sdfg,
        concat_node=concat_node,
    )
    sdfg.validate()

    access_nodes_after = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(access_nodes_after) == 4
    assert state.out_degree(a) == 2
    assert all(oedge.dst is me for oedge in state.out_edges(a))
    assert concat_node not in access_nodes_after
    assert concat_node.data not in sdfg.arrays
    assert all(
        ac.data.startswith("__gt4py_concat_where_mapper_temp_c_")
        for ac in access_nodes_after
        if ac.desc(sdfg).transient
    )
    assert util.count_nodes(state, dace_nodes.Tasklet) == 2
    assert state.in_degree(me) == 3
    assert {iedge.src.data for iedge in state.in_edges(me)} == {a.data, "b"}

    csdfg = util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)


def _make_concat_where_non_canonical_memlet_sdfg() -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.MapEntry, dace_nodes.AccessNode
]:
    sdfg = dace.SDFG(util.unique_name("concat_where_replacer_non_canonical_memlet"))
    state = sdfg.add_state()
    for name in "abcd":
        sdfg.add_array(
            name=name,
            shape=(10,),
            dtype=dace.float64,
            transient=(name == "c"),
        )
    sdfg.add_array(
        "t",
        shape=(2,),
        dtype=dace.float64,
        transient=True,
    )

    a, b, c, d, t = (state.add_access(name) for name in "abcdt")

    me, mx = state.add_map("comp_map", ndrange={"__i": "0:10"})
    tlet = state.add_tasklet(
        "comp_tlet",
        inputs={"__in0", "__in1"},
        outputs={"__out"},
        code="__out = __in0 + __in1",
    )

    state.add_nedge(a, c, dace.Memlet("a[1:6] -> [0:5]"))
    state.add_nedge(b, c, dace.Memlet("b[3:8] -> [5:10]"))

    state.add_edge(c, None, me, "IN_c", dace.Memlet("c[0:10]"))
    state.add_edge(b, None, me, "IN_b", dace.Memlet("b[0:10]"))
    state.add_edge(
        me,
        "OUT_c",
        t,
        None,
        dace.Memlet(
            data="t",
            subset="1",
            other_subset="__i",
        ),
    )
    state.add_edge(t, None, tlet, "__in0", dace.Memlet("t[1]"))
    state.add_edge(me, "OUT_b", tlet, "__in1", dace.Memlet("b[__i]"))
    state.add_edge(tlet, "__out", mx, "IN_d", dace.Memlet("d[__i]"))
    state.add_edge(mx, "OUT_d", d, None, dace.Memlet("d[0:10]"))
    me.add_scope_connectors("c")
    me.add_scope_connectors("b")
    mx.add_scope_connectors("d")

    sdfg.validate()

    return sdfg, state, c, me, t


def test_concat_where_non_canonical_memlet_sdfg():
    sdfg, state, concat_node, me, t = _make_concat_where_non_canonical_memlet_sdfg()

    access_nodes_before = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(access_nodes_before) == 5
    assert concat_node in access_nodes_before
    assert all(
        iedge.src is me
        and iedge.data.data == t.data
        and str(iedge.data.other_subset) == "__i"
        and str(iedge.data.subset) == "1"
        for iedge in state.in_edges(t)
    )

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    gtx_transformations.gt_replace_concat_where_node(
        state=state,
        sdfg=sdfg,
        concat_node=concat_node,
    )
    sdfg.validate()

    access_nodes_after = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(access_nodes_after) == 5
    assert concat_node not in access_nodes_after
    assert concat_node.data not in sdfg.arrays
    assert t in access_nodes_after
    assert all(
        iedge.data.data != t.data
        and str(iedge.data.other_subset) == "1"
        and str(iedge.data.subset) == "0"
        and iedge.src is not me
        for iedge in state.in_edges(t)
    )

    csdfg = util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)


def _make_concat_where_multiple_nested_consumers(
    uniform_access: bool,
) -> tuple[dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.MapEntry]:
    sdfg = dace.SDFG(util.unique_name("concat_where_replacer_multiple_consumer"))
    state = sdfg.add_state()
    for name in "abcde":
        sdfg.add_array(
            name=name,
            shape=(10,),
            dtype=dace.float64,
            transient=(name == "c"),
        )

    a, b, c = (state.add_access(name) for name in "abc")

    me, mx = state.add_map("map", ndrange={"__i": "0:10"})
    tlet1 = state.add_tasklet(
        "tlet1",
        inputs={"__in0", "__in1"},
        outputs={"__out"},
        code="__out = __in0 * __in1",
    )
    tlet2 = state.add_tasklet(
        "tlet2",
        inputs={"__in0", "__in1"},
        outputs={"__out"},
        code="__out = __in0 + __in1",
    )

    state.add_nedge(a, c, dace.Memlet("a[2:7] -> [0:5]"))
    state.add_nedge(b, c, dace.Memlet("b[3:8] -> [5:10]"))

    for ac in [a, b, c]:
        state.add_edge(ac, None, me, "IN_" + ac.data, dace.Memlet(ac.data + "[0:10]"))

    for tlet, inp, out in [(tlet1, "a", "d"), (tlet2, "b", "e")]:
        concat_access = "9 - __i" if (not uniform_access) and inp == "a" else "__i"
        state.add_edge(me, "OUT_c", tlet, "__in0", dace.Memlet(f"c[{concat_access}]"))
        state.add_edge(me, "OUT_" + inp, tlet, "__in1", dace.Memlet(inp + "[__i]"))
        state.add_edge(tlet, "__out", mx, "IN_" + out, dace.Memlet(out + "[__i]"))
        state.add_edge(mx, "OUT_" + out, state.add_access(out), None, dace.Memlet(out + "[0:10]"))
        mx.add_scope_connectors(out)
        me.add_scope_connectors(inp)
    me.add_scope_connectors("c")

    sdfg.validate()
    return sdfg, state, c, me


@pytest.mark.parametrize("uniform_access", [True, False])
def test_concat_where_multiple_nested_consumers(uniform_access: bool):
    sdfg, state, concat_node, me = _make_concat_where_multiple_nested_consumers(
        uniform_access=uniform_access
    )

    access_nodes_before = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    tasklets_before = util.count_nodes(state, dace_nodes.Tasklet, return_nodes=True)
    assert len(access_nodes_before) == 5
    assert concat_node in access_nodes_before
    assert len(tasklets_before) == 2
    assert all(all(e.src is me for e in state.in_edges(tlet)) for tlet in tasklets_before)
    assert state.out_degree(concat_node) == 1
    assert all(
        e.dst is me and e.dst_conn == f"IN_{concat_node.data}" for e in state.out_edges(concat_node)
    )
    inner_concat_edges_before = list(state.out_edges_by_connector(me, f"OUT_{concat_node.data}"))
    assert len(inner_concat_edges_before) == 2
    assert all(isinstance(e.dst, dace_nodes.Tasklet) for e in inner_concat_edges_before)

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    gtx_transformations.gt_replace_concat_where_node(
        state=state,
        sdfg=sdfg,
        concat_node=concat_node,
    )
    sdfg.validate()

    # NOTE Currently there is no distinction between uniform access and non uniform
    #   access. In the uniform case both accesses to the concat where node are on
    #   the same index, thus it would be possible to only use one of them. However,
    #   currently this is not done.
    access_nodes_after = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    tasklets_after = util.count_nodes(state, dace_nodes.Tasklet, return_nodes=True)
    assert len(access_nodes_after) == 6
    assert concat_node not in access_nodes_after
    assert len(tasklets_after) == 4
    assert all(old_tlet in tasklets_after for old_tlet in tasklets_before)
    new_tasklets = [tlet for tlet in tasklets_after if tlet not in tasklets_before]
    assert len(new_tasklets) == 2

    csdfg = util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)


def _make_concat_where_nested_scopes() -> tuple[
    dace.SDFG,
    dace.SDFGState,
    dace_nodes.AccessNode,
    dace_nodes.MapEntry,
    dace_nodes.MapEntry,
    dace_nodes.Tasklet,
]:
    sdfg = dace.SDFG(util.unique_name("concat_where_replacer_nested_consumer"))
    state = sdfg.add_state()
    for name in "abcd":
        sdfg.add_array(
            name=name,
            shape=(10, 10),
            dtype=dace.float64,
            transient=(name == "c"),
        )

    a, b, c, d = (state.add_access(name) for name in "abcd")

    me, mx = state.add_map("outer_map", ndrange={"__i": "0:10"})
    nme, nmx = state.add_map("nested_map", ndrange={"__j": "0:10"})
    tlet = state.add_tasklet(
        "comp",
        inputs={"__in0", "__in1"},
        outputs={"__out"},
        code="__out = __in0 + __in1",
    )

    state.add_nedge(a, c, dace.Memlet("a[1:6, 3:8] -> [0:5, 0:5]"))
    state.add_nedge(a, c, dace.Memlet("a[4:9, 4:9] -> [5:10, 5:10]"))
    state.add_nedge(b, c, dace.Memlet("b[0:5, 0:5] -> [0:5, 5:10]"))
    state.add_nedge(b, c, dace.Memlet("b[3:8, 4:9] -> [5:10, 0:5]"))

    for ac, conn in [(a, "__in0"), (c, "__in1")]:
        state.add_edge(ac, None, me, f"IN_{ac.data}", dace.Memlet(f"{ac.data}[0:10, 0:10]"))
        state.add_edge(
            me, f"OUT_{ac.data}", nme, f"IN_{ac.data}", dace.Memlet(f"{ac.data}[__i, 0:10]")
        )
        state.add_edge(nme, f"OUT_{ac.data}", tlet, conn, dace.Memlet(f"{ac.data}[__i, __j]"))
        me.add_scope_connectors(ac.data)
        nme.add_scope_connectors(ac.data)

    state.add_edge(tlet, "__out", nmx, "IN_d", dace.Memlet("d[__i, __j]"))
    state.add_edge(nmx, "OUT_d", mx, "IN_d", dace.Memlet("d[__i, 0:10]"))
    state.add_edge(mx, "OUT_d", d, None, dace.Memlet("d[0:10, 0:10]"))
    mx.add_scope_connectors("d")
    nmx.add_scope_connectors("d")

    sdfg.validate()

    return sdfg, state, c, me, nme, tlet


def test_concat_where_nested_scopes():
    sdfg, state, concat_node, me, nme, tlet = _make_concat_where_nested_scopes()

    access_nodes_before = util.count_nodes(state, dace_nodes.AccessNode, True)
    assert len(access_nodes_before) == 4
    assert concat_node in access_nodes_before
    assert set(util.count_nodes(state, dace_nodes.MapEntry, True)) == {me, nme}
    assert util.count_nodes(state, dace_nodes.Tasklet, True) == [tlet]
    assert state.scope_dict()[tlet] is nme
    assert all(e.src is nme for e in state.in_edges(tlet))

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    gtx_transformations.gt_replace_concat_where_node(
        state=state,
        sdfg=sdfg,
        concat_node=concat_node,
    )
    sdfg.validate()

    access_nodes_after = util.count_nodes(state, dace_nodes.AccessNode, True)
    tasklets_after = util.count_nodes(state, dace_nodes.Tasklet, True)
    assert len(access_nodes_after) == 4
    assert concat_node not in access_nodes_after
    new_ac = next(ac for ac in access_nodes_after if ac not in access_nodes_before)
    assert any(e.src is new_ac for e in state.in_edges(tlet))
    assert any(e.src is nme for e in state.in_edges(tlet))
    assert len(tasklets_after) == 2
    assert tlet in tasklets_after

    csdfg = util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)


def _make_concat_where_multiple_nested_scopes(
    access_in_both_scopes: bool,
) -> tuple[
    dace.SDFG,
    dace.SDFGState,
    dace_nodes.AccessNode,
    dace_nodes.MapEntry,
    dace_nodes.MapEntry,
    dace_nodes.Tasklet,
    dace_nodes.MapEntry,
    dace_nodes.Tasklet,
]:
    sdfg = dace.SDFG(util.unique_name("concat_where_replacer_nested_consumer"))
    state = sdfg.add_state()
    for name in "abcd":
        sdfg.add_array(
            name=name,
            shape=(10, 10),
            dtype=dace.float64,
            transient=(name == "c"),
        )

    a, b, c, d = (state.add_access(name) for name in "abcd")

    me, mx = state.add_map("outer_map", ndrange={"__i": "0:10"})

    nested_ranges = ["0:6", "6:10"]
    nmaps, tlets = [], []
    for i, ndrange in enumerate(nested_ranges):
        op_ = "*" if i % 2 == 0 else "+"
        nmaps.append(state.add_map(f"nested_map{i}", ndrange={f"__j{i}": ndrange}))
        tlets.append(
            state.add_tasklet(
                f"comp{i}",
                inputs={"__in0", "__in1"},
                outputs={"__out"},
                code=f"__out = __in0 {op_} __in1",
            )
        )

    # Perform the concat where.
    state.add_nedge(a, c, dace.Memlet("a[1:6, 3:8] -> [0:5, 0:5]"))
    state.add_nedge(a, c, dace.Memlet("a[4:9, 4:9] -> [5:10, 5:10]"))
    state.add_nedge(b, c, dace.Memlet("b[0:5, 0:5] -> [0:5, 5:10]"))
    state.add_nedge(b, c, dace.Memlet("b[3:8, 4:9] -> [5:10, 0:5]"))

    wiring_specs = [[a, c], [b, (c if access_in_both_scopes else a)]]
    for ac in [a, b, c]:
        state.add_edge(ac, None, me, f"IN_{ac.data}", dace.Memlet(f"{ac.data}[0:10, 0:10]"))
        me.add_scope_connectors(ac.data)

    for ndrange, (nme, nmx), tlet, wspec in zip(nested_ranges, nmaps, tlets, wiring_specs):
        for ac, conn in zip(wspec, ["__in0", "__in1"]):
            state.add_edge(
                me, f"OUT_{ac.data}", nme, f"IN_{ac.data}", dace.Memlet(f"{ac.data}[__i, 0:10]")
            )
            state.add_edge(
                nme,
                f"OUT_{ac.data}",
                tlet,
                conn,
                dace.Memlet(f"{ac.data}[__i, {nme.map.params[0]}]"),
            )
            nme.add_scope_connectors(ac.data)
        state.add_edge(tlet, "__out", nmx, "IN_d", dace.Memlet(f"d[__i, {nme.map.params[0]}]"))
        state.add_edge(nmx, "OUT_d", mx, "IN_d", dace.Memlet("d[__i, 0:10]"))
        nmx.add_scope_connectors("d")

    state.add_edge(mx, "OUT_d", d, None, dace.Memlet("d[0:10, 0:10]"))
    mx.add_scope_connectors("d")

    sdfg.validate()
    return sdfg, state, c, me, nmaps[0][0], tlets[0], nmaps[1][0], tlets[1]


@pytest.mark.parametrize("access_in_both_scopes", [True, False])
def test_concat_where_multiple_nested_scopes(access_in_both_scopes: bool):
    """
    Same as `test_concat_where_nested_scopes()` but with two nested Maps. Depending on
    `access_in_both_scopes` the second nested Map does not use the concat where data.
    """
    sdfg, state, concat_node, me, nme1, tlet1, nme2, tlet2 = (
        _make_concat_where_multiple_nested_scopes(access_in_both_scopes=access_in_both_scopes)
    )

    access_nodes_before = util.count_nodes(state, dace_nodes.AccessNode, True)
    assert len(access_nodes_before) == 4
    assert concat_node in access_nodes_before
    assert set(util.count_nodes(state, dace_nodes.MapEntry, True)) == {me, nme1, nme2}
    assert set(util.count_nodes(state, dace_nodes.Tasklet, True)) == {tlet1, tlet2}
    assert (state.scope_dict()[tlet1] is nme1) and (state.scope_dict()[tlet2] is nme2)
    assert all(e.src is nme1 for e in state.in_edges(tlet1))
    assert all(e.src is nme2 for e in state.in_edges(tlet2))

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    gtx_transformations.gt_replace_concat_where_node(
        state=state,
        sdfg=sdfg,
        concat_node=concat_node,
    )
    sdfg.validate()

    scope_dict = state.scope_dict()
    access_nodes_after = util.count_nodes(state, dace_nodes.AccessNode, True)
    new_acs = [ac for ac in access_nodes_after if ac not in access_nodes_before]
    tasklets_after = util.count_nodes(state, dace_nodes.Tasklet, True)
    assert concat_node not in access_nodes_after
    assert concat_node.data not in sdfg.arrays
    assert tlet1 in tasklets_after
    assert tlet2 in tasklets_after

    if access_in_both_scopes:
        assert len(tasklets_after) == 4
        assert len(access_nodes_after) == 5
        assert len(new_acs) == 2

        if scope_dict[new_acs[0]] is nme1:
            new_ac1, new_ac2 = new_acs
        else:
            new_ac2, new_ac1 = new_acs
        assert scope_dict[new_ac1] is nme1
        assert scope_dict[new_ac2] is nme2
        assert sum(1 for e in state.in_edges(tlet1) if e.src is new_ac1) == 1
        assert sum(1 for e in state.in_edges(tlet2) if e.src is new_ac2) == 1

    else:
        assert len(tasklets_after) == 3
        assert len(access_nodes_after) == 4
        assert len(new_acs) == 1
        new_ac = next(iter(new_acs))
        assert scope_dict[new_ac] is nme1
        assert sum(1 for e in state.in_edges(tlet1) if e.src is new_ac) == 1

    csdfg = util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)


def _make_concat_where_global_read(
    use_tasklet: bool, scalar_access: int
) -> tuple[
    dace.SDFG,
    dace.SDFGState,
    dace_nodes.AccessNode,
]:
    sdfg = dace.SDFG(util.unique_name("concat_where_replacer_global_consumer"))
    state = sdfg.add_state()
    for aname in "abcd":
        sdfg.add_array(
            name=aname,
            shape=(10,),
            dtype=dace.float64,
            transient=(aname == "c"),
        )
    a, b, c, d = (state.add_access(aname) for aname in "abcd")

    state.add_nedge(a, c, dace.Memlet("a[1:6] -> [0:5]"))
    state.add_nedge(b, c, dace.Memlet("b[3:8] -> [5:10]"))

    if use_tasklet:
        tlet = state.add_tasklet(
            "consumer",
            inputs={"__in0"},
            outputs={"__out"},
            code="__out = __in0 + 1.234",
        )
        state.add_edge(c, None, tlet, "__in0", dace.Memlet(f"c[{scalar_access}]"))
        state.add_edge(tlet, "__out", d, None, dace.Memlet("d[5]"))
    else:
        state.add_edge(c, None, d, None, dace.Memlet(f"c[{scalar_access}]"))

    sdfg.validate()

    return sdfg, state, c


@pytest.mark.parametrize("use_tasklet", [False, True])
@pytest.mark.parametrize("scalar_access", [2, 8])
def test_concat_where_global_read(use_tasklet: bool, scalar_access: int):
    sdfg, state, concat_node = _make_concat_where_global_read(use_tasklet, scalar_access)

    access_nodes_before = util.count_nodes(state, dace_nodes.AccessNode, True)
    assert len(access_nodes_before) == 4
    assert concat_node in access_nodes_before

    if use_tasklet:
        tasklets_before = util.count_nodes(state, dace_nodes.Tasklet, True)
        assert len(tasklets_before) == 1
        org_tlet = tasklets_before[0]
        assert all(oedge.dst is org_tlet for oedge in state.out_edges(concat_node))
    else:
        assert util.count_nodes(state, dace_nodes.Tasklet) == 0
        assert all(
            isinstance(oedge.dst, dace_nodes.AccessNode) and oedge.dst.data == "d"
            for oedge in state.out_edges(concat_node)
        )

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    gtx_transformations.gt_replace_concat_where_node(
        state=state,
        sdfg=sdfg,
        concat_node=concat_node,
    )
    sdfg.validate()

    access_nodes_after = util.count_nodes(state, dace_nodes.AccessNode, True)
    tasklets_after = util.count_nodes(state, dace_nodes.Tasklet, True)
    assert concat_node not in access_nodes_after
    assert concat_node.data not in sdfg.arrays

    if use_tasklet:
        assert len(tasklets_after) == 2
        assert org_tlet in tasklets_after
        assert len(access_nodes_after) == 4
        concat_where_tlet = next(tlet for tlet in tasklets_after if tlet is not org_tlet)
    else:
        # In this case we can remove the intermediate because we could directly
        #  write into the destination output. However, currently we do not do it.
        assert len(access_nodes_after) == 4
        assert len(tasklets_after) == 1
        concat_where_tlet = tasklets_after[0]

    assert state.in_degree(concat_where_tlet) == 2
    assert state.out_degree(concat_where_tlet) == 1


def _make_concat_where_multiple_top_level_maps(
    access_in_both_scopes: bool,
) -> tuple[
    dace.SDFG,
    dace.SDFGState,
    dace_nodes.AccessNode,
]:
    sdfg, state, concat_node, *_ = _make_concat_where_multiple_nested_scopes(
        access_in_both_scopes=access_in_both_scopes
    )

    anames = ["o1", "o2", "o3"]
    for aname in anames:
        sdfg.add_array(
            name=aname,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )

    state.add_mapped_tasklet(
        "addi_comp1",
        map_ranges={"__k": "0:10"},
        inputs={"__in": dace.Memlet(f"{concat_node}[3, __k]")},
        outputs={"__out": dace.Memlet("o1[__k]")},
        code="__out = __in + 2.0",
        input_nodes={concat_node},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "addi_comp2",
        map_ranges={"__k": "0:10"},
        inputs={"__in": dace.Memlet(f"{concat_node.data}[__k, 8]")},
        outputs={"__out": dace.Memlet("o2[__k]")},
        code="__out = __in * 5.0",
        input_nodes={concat_node},
        external_edges=True,
    )
    state.add_nedge(concat_node, state.add_access("o3"), dace.Memlet(f"{concat_node}[4, 7] -> [6]"))

    sdfg.validate()

    return sdfg, state, concat_node


@pytest.mark.parametrize("access_in_both_scopes", [True, False])
def test_concat_where_multiple_top_level_maps(access_in_both_scopes: bool):
    """
    Essentially the same as `test_concat_where_multiple_nested_scopes()` but it
    has more consumer on the top level.
    """
    sdfg, state, concat_node = _make_concat_where_multiple_top_level_maps(
        access_in_both_scopes=access_in_both_scopes
    )

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    gtx_transformations.gt_replace_concat_where_node(
        state=state,
        sdfg=sdfg,
        concat_node=concat_node,
    )
    sdfg.validate()

    access_nodes_after = util.count_nodes(state, dace_nodes.AccessNode, True)
    assert concat_node not in access_nodes_after
    assert concat_node.data not in sdfg.arrays
