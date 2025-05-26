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
from dace import symbolic as dace_symbolic, subsets as dace_sbs
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _make_distributed_split_sdfg() -> (
    tuple[
        dace.SDFG,
        dace.SDFGState,
        dace_nodes.Tasklet,
        dace_nodes.MapExit,
        dace_nodes.AccessNode,
        dace_nodes.AccessNode,
        dace.SDFGState,
        dace_nodes.AccessNode,
    ]
):
    sdfg = dace.SDFG(util.unique_name("distributed_split_sdfg"))
    state = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state)

    for name in "abt":
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=(name == "t"),
        )

    a, t = (state.add_access(name) for name in "at")
    t2 = state2.add_access("t")
    tlet, me, mx = state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i": "0:5"},
        inputs={"__in": dace.Memlet("a[__i + 5]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("t[__i + 5]")},
        input_nodes={a},
        output_nodes={t},
        external_edges=True,
    )
    state.add_nedge(a, t, dace.Memlet("a[0:5] -> [0:5]"))
    state2.add_nedge(
        t2,
        state2.add_access("b"),
        dace.Memlet("t[0:10] -> [0:10]"),
    )
    sdfg.validate()

    return sdfg, state, tlet, mx, a, t, state2, t2


def test_distributed_split():
    sdfg, state, tlet, mx, a, t, state2, t2 = _make_distributed_split_sdfg()

    org_tlet_out_memlet = list(state.out_edges(tlet))[0].data
    assert (org_tlet_out_memlet.subset[0][0] - dace_symbolic.pystr_to_symbolic("__i")) == 5

    split_description = [
        dace_sbs.Range.from_string("0:5"),
        dace_sbs.Range.from_string("5:10"),
    ]

    new_access_nodes = gtx_transformations.spliting_tools.split_node(
        state=state,
        sdfg=sdfg,
        node_to_split=t,
        split_description=split_description,
    )

    ac_nodes = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(ac_nodes) == 3
    assert all(ac.data != t for ac in ac_nodes)

    new_ac_0_to_5 = new_access_nodes[split_description[0]]
    assert state.out_degree(new_ac_0_to_5) == 0
    assert state.in_degree(new_ac_0_to_5) == 1
    assert all(iedge.src is a for iedge in state.in_edges(new_ac_0_to_5))

    new_ac_5_to_10 = new_access_nodes[split_description[1]]
    assert state.out_degree(new_ac_5_to_10) == 0
    assert state.in_degree(new_ac_5_to_10) == 1
    assert all(iedge.src is mx for iedge in state.in_edges(new_ac_5_to_10))

    tlet_out_memlet = list(state.out_edges(tlet))[0].data
    assert (tlet_out_memlet.subset[0][0] - dace_symbolic.pystr_to_symbolic("__i")) == 0

    # The second state is not handled yet, thus this test should fail as a reminder.
    assert False


def _make_split_node_simple_sdfg() -> (
    tuple[dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.MapExit, dace_nodes.MapExit]
):
    sdfg = dace.SDFG(util.unique_name("single_state_split"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abt":
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=(name == "t"),
        )

    a, b, t = (state.add_access(name) for name in "abt")
    tlet1, me1, mx1 = state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:5"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("t[__i]")},
        input_nodes={a},
        output_nodes={t},
        external_edges=True,
    )
    tlet2, me2, mx2 = state.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i": "5:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 2.0",
        outputs={"__out": dace.Memlet("t[__i]")},
        input_nodes={a},
        output_nodes={t},
        external_edges=True,
    )

    # TODO(phimuell): Modify the test once the node splitter has learned to
    #   split these kind of edges by itself.
    state.add_nedge(t, b, dace.Memlet("t[0:5] -> [0:5]"))
    state.add_nedge(t, b, dace.Memlet("t[5:10] -> [5:10]"))
    sdfg.validate()

    return sdfg, state, t, mx1, mx2


def test_simple_node_split():
    sdfg, state, t, mx1, mx2 = _make_split_node_simple_sdfg()

    assert util.count_nodes(state, dace_nodes.AccessNode) == 3
    assert util.count_nodes(state, dace_nodes.MapEntry) == 2

    ref = {
        "a": np.array(np.random.rand(10), copy=True, dtype=np.float64),
        "b": np.array(np.random.rand(10), copy=True, dtype=np.float64),
    }
    res = copy.deepcopy(ref)

    csdfg_ref = sdfg.compile()
    csdfg_ref(**ref)
    del csdfg_ref

    split_description = [
        dace_sbs.Range.from_string("0:5"),
        dace_sbs.Range.from_string("5:10"),
    ]

    new_access_nodes = gtx_transformations.spliting_tools.split_node(
        state=state,
        sdfg=sdfg,
        node_to_split=t,
        split_description=split_description,
    )
    sdfg.validate()

    ac_nodes = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(ac_nodes) == 4
    assert all(nac in ac_nodes for nac in new_access_nodes.values())
    assert all(ac.data != "t" for ac in ac_nodes)

    csdfg_res = sdfg.compile()
    csdfg_res(**res)
    del csdfg_res

    assert all(np.allclose(ref[n], res[n]) for n in ref)


def _make_split_edge_sdfg() -> (
    tuple[dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.AccessNode]
):
    sdfg = dace.SDFG(util.unique_name("split_edge"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abt":
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=(name == "t"),
        )

    t = state.add_access("t")
    b = state.add_access("b")
    state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:5"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("t[__i]")},
        output_nodes={t},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i": "5:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in - 1.0",
        outputs={"__out": dace.Memlet("t[__i]")},
        output_nodes={t},
        external_edges=True,
    )
    state.add_nedge(t, b, dace.Memlet("t[0:10] -> [0:10]"))
    sdfg.validate()

    return sdfg, state, t, b


def test_split_edge():
    sdfg, state, t, b = _make_split_edge_sdfg()

    assert state.out_degree(t) == 1
    assert state.in_degree(t) == 2
    assert state.degree(b) == 1

    edge_to_split = next(iter(state.out_edges(t)))
    assert edge_to_split.dst is b

    # Note we only specify the first half, however, the other part `5:10` will
    #  be maintained.
    split = dace_sbs.Range.from_string("0:5")

    new_edges_by_split = gtx_transformations.spliting_tools.split_copy_edge(
        state=state,
        sdfg=sdfg,
        edge_to_split=edge_to_split,
        split_description=[split],
    )
    sdfg.validate()

    explected_subsets = [
        dace_sbs.Range.from_string("0:5"),
        dace_sbs.Range.from_string("5:10"),
    ]

    assert len(new_edges_by_split) == 2
    assert state.out_degree(t) == 2
    assert state.in_degree(b) == 2
    assert {split, None} == new_edges_by_split.keys()
    assert len(new_edges_by_split[None]) == 1
    assert len(new_edges_by_split[split]) == 1
    assert all(
        all(e.src is t and e.dst is b for e in new_edges)
        for new_edges in new_edges_by_split.values()
    )
    assert new_edges_by_split[split].union(new_edges_by_split[None]) == set(state.out_edges(t))
    assert set(explected_subsets) == {oedge.data.src_subset for oedge in state.out_edges(t)}
    assert set(explected_subsets) == {iedge.data.dst_subset for iedge in state.in_edges(b)}


def test_split_edge_2d():
    sdfg = dace.SDFG(util.unique_name("split_edge_2d"))
    state = sdfg.add_state(is_start_block=True)

    for name in "ab":
        sdfg.add_array(
            name,
            shape=(10, 10),
            dtype=dace.float64,
            transient=False,
        )

    a = state.add_access("a")
    b = state.add_access("b")
    state.add_nedge(a, b, dace.Memlet("a[0:9, 0:8] -> [1:10, 2:10]"))
    assert state.degree(a) == 1
    assert state.degree(b) == 1

    edge_to_split = next(iter(state.out_edges(a)))
    assert edge_to_split.dst is b

    split = dace_sbs.Range.from_string("0:5, 0:4")

    # There will be one edge that copies the split, and two edges for the rest.
    #  However, there are two different possibilities how they are split.
    new_edges_by_split = gtx_transformations.spliting_tools.split_copy_edge(
        state=state,
        sdfg=sdfg,
        edge_to_split=edge_to_split,
        split_description=[split],
    )
    sdfg.validate()

    assert len(new_edges_by_split) == 2
    assert state.out_degree(a) == 3
    assert state.in_degree(b) == 3

    new_edges_split = new_edges_by_split[split]
    assert len(new_edges_split) == 1
    new_edge_split = next(iter(new_edges_split))
    assert new_edge_split.data.src_subset == split
    assert new_edge_split.data.dst_subset == dace_sbs.Range.from_string("1:6, 2:6")

    # We could now also inspect the subsets, but since there are multiple, ways how
    #  they are split, we simply compile the SDFG and compare with a reference.
    new_edges_None = new_edges_by_split[None]
    assert len(new_edges_None) == 2
    assert all(not new_edge.data.src_subset.intersects(split) for new_edge in new_edges_None)
    assert all(
        not new_edge.data.dst_subset.intersects(new_edge_split.data.dst_subset)
        for new_edge in new_edges_None
    )

    a = np.array(np.random.rand(10, 10), copy=True, dtype=np.float64)
    b = np.array(np.random.rand(10, 10), copy=True, dtype=np.float64)
    b_ref = copy.deepcopy(b)
    b_ref[1:10, 2:10] = a[0:9, 0:8]

    util.compile_and_run_sdfg(sdfg, a=a, b=b)
    assert np.all(b_ref == b)
