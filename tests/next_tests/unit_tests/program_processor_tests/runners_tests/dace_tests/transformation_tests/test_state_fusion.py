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
)

from . import util

import dace


def _make_simple_two_state_sdfg() -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("simple_linear_states"))
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    for name in ["a", "b", "c", "d"]:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["b"].transient = True

    state1.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("a[__i]")},
        code="__out = __in1 + 10.",
        outputs={"__out": dace.Memlet("b[__i]")},
        external_edges=True,
    )

    b2 = state2.add_access("b")
    state2.add_mapped_tasklet(
        "comp2_1",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("b[__i]")},
        code="__out = __in1 + 11.",
        outputs={"__out": dace.Memlet("c[__i]")},
        external_edges=True,
        input_nodes={b2},
    )
    state2.add_mapped_tasklet(
        "comp2_2",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("b[__i]")},
        code="__out = __in1 + 11.",
        outputs={"__out": dace.Memlet("d[__i]")},
        external_edges=True,
        input_nodes={b2},
    )
    return sdfg, state1, state2


def _make_global_in_both_read_and_write() -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    """The first state contains a read to a global and the second contains a write to the global."""
    sdfg = dace.SDFG(util.unique_name("global_in_both_states_read_and_write"))
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    for name in ["g", "t"]:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["t"].transient = True

    state1.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("g[__i]")},
        code="__out = __in1 + 10.",
        outputs={"__out": dace.Memlet("t[__i]")},
        external_edges=True,
    )
    state2.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("t[__i]")},
        code="__out = __in1 + 12.",
        outputs={"__out": dace.Memlet("g[__i]")},
        external_edges=True,
    )

    return sdfg, state1, state2


def _make_global_both_state_read() -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    """In both states the same global is read."""
    sdfg = dace.SDFG(util.unique_name("global_read_in_the_same_state"))
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    for name in ["a", "b", "c"]:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )

    state1.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("a[__i]")},
        code="__out = __in1 + 10.",
        outputs={"__out": dace.Memlet("b[__i]")},
        external_edges=True,
    )
    state2.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("a[__i]")},
        code="__out = __in1 + 12.",
        outputs={"__out": dace.Memlet("c[__i]")},
        external_edges=True,
    )

    return sdfg, state1, state2


def _make_global_both_state_write() -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    """In both states the same global is written."""
    sdfg = dace.SDFG(util.unique_name("global_write_in_the_same_state"))
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    for name in ["a", "b", "c"]:
        sdfg.add_array(
            name,
            shape=((20,) if name == "a" else (10,)),
            dtype=dace.float64,
            transient=False,
        )

    state1.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("b[__i]")},
        code="__out = __in1 + 10.",
        outputs={"__out": dace.Memlet("a[__i]")},
        external_edges=True,
    )
    state2.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("c[__i]")},
        code="__out = __in1 + 12.",
        outputs={"__out": dace.Memlet("a[__i + 10]")},
        external_edges=True,
    )

    return sdfg, state1, state2


def _make_empty_state(
    first_state_empty: bool,
) -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    sdfg = dace.SDFG(
        util.unique_name("global_" + ("first" if first_state_empty else "second") + "_state_empty")
    )
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    for name in ["a", "b"]:
        sdfg.add_array(
            name,
            shape=((20,) if name == "a" else (10,)),
            dtype=dace.float64,
            transient=False,
        )

    state = state2 if first_state_empty else state1
    state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("a[__i]")},
        code="__out = __in1 + 10.",
        outputs={"__out": dace.Memlet("b[__i]")},
        external_edges=True,
    )

    return sdfg, state1, state2


def _make_global_merge_1() -> dace.SDFG:
    """The first state writes to a global while the second state reads and writes to it."""
    sdfg = dace.SDFG(util.unique_name("global_merge_1"))
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    for name in ["a", "b"]:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )

    state1.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("b[__i]")},
        code="__out = __in1 + 10.",
        outputs={"__out": dace.Memlet("a[__i]")},
        external_edges=True,
    )
    state2.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("a[__i]")},
        code="__out = __in1 + 12.",
        outputs={"__out": dace.Memlet("a[__i]")},
        external_edges=True,
    )
    return sdfg


def _make_global_merge_2() -> dace.SDFG:
    """In both states the global data is read and written to."""
    sdfg = dace.SDFG(util.unique_name("global_merge_2"))
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    sdfg.add_array(
        "a",
        shape=(10,),
        dtype=dace.float64,
        transient=False,
    )

    state1.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("a[__i]")},
        code="__out = __in1 + 10.",
        outputs={"__out": dace.Memlet("a[__i]")},
        external_edges=True,
    )
    state2.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("a[__i]")},
        code="__out = __in1 + 12.",
        outputs={"__out": dace.Memlet("a[__i]")},
        external_edges=True,
    )
    return sdfg


def _make_swapping_sdfg() -> dace.SDFG:
    """Makes an SDFG that implements `x, y = y, x` with Memlets."""
    sdfg = dace.SDFG(util.unique_name("swapping_sdfg"))
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    for name in ["x", "y", "t1", "t2"]:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=name.startswith("t"),
        )

    state1.add_nedge(
        state1.add_access("x"), state1.add_access("t1"), dace.Memlet("x[0:10] -> [0:10]")
    )
    state1.add_nedge(
        state1.add_access("y"), state1.add_access("t2"), dace.Memlet("y[0:10] -> [0:10]")
    )

    state2.add_nedge(
        state2.add_access("t1"), state2.add_access("y"), dace.Memlet("t1[0:10] -> [0:10]")
    )
    state2.add_nedge(
        state2.add_access("t2"), state2.add_access("x"), dace.Memlet("t2[0:10] -> [0:10]")
    )

    return sdfg


def _make_non_concurrent_sdfg() -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("non_concurrent_sdfg"))
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    for name in ["x", "y", "t1", "t2"]:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=name.startswith("t"),
        )

    state1.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in1": dace.Memlet("x[__i]"),
            "__in2": dace.Memlet("y[__i]"),
        },
        code="__out1 = __in1 + __in2\n__out2 = __in1 - __in2",
        outputs={
            "__out1": dace.Memlet("t1[__i]"),
            "__out2": dace.Memlet("t2[__i]"),
        },
        external_edges=True,
    )

    state2.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in1": dace.Memlet("t1[__i]"),
            "__in2": dace.Memlet("t2[__i]"),
        },
        code="__out1 = __in1 * 2.\n__out2 = __in2 * 3.",
        outputs={
            "__out1": dace.Memlet("y[__i]"),
            "__out2": dace.Memlet("x[__i]"),
        },
        external_edges=True,
    )

    return sdfg, state1, state2


def _make_double_producer_sdfg() -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("double_producer_sdfg"))
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    for name in ["x", "y", "t1", "t2"]:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=name.startswith("t"),
        )

    state1.add_mapped_tasklet(
        "comp1_1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("x[__i]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("t1[__i]")},
        external_edges=True,
    )
    state1.add_mapped_tasklet(
        "comp1_2",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("y[__i]")},
        code="__out = __in + 2.0",
        outputs={"__out": dace.Memlet("t2[__i]")},
        external_edges=True,
    )

    state2.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in1": dace.Memlet("t1[__i]"),
            "__in2": dace.Memlet("t2[__i]"),
        },
        code="__out1 = __in1 * 2.\n__out2 = __in2 * 3.",
        outputs={
            "__out1": dace.Memlet("y[__i]"),
            "__out2": dace.Memlet("x[__i]"),
        },
        external_edges=True,
    )
    sdfg.validate()

    return sdfg, state1, state2


def _make_double_consumer_sdfg() -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("double_consumer_sdfg"))
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    for name in ["x", "y", "t1", "t2"]:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=name.startswith("t"),
        )

    x1 = state1.add_access("x")
    y1 = state1.add_access("y")

    state1.add_mapped_tasklet(
        "comp1_1",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in1": dace.Memlet("x[__i]"),
            "__in2": dace.Memlet("y[__i]"),
        },
        code="__out = __in1 * 2.",
        outputs={
            "__out": dace.Memlet("t1[__i]"),
        },
        input_nodes={x1, y1},
        external_edges=True,
    )
    state1.add_mapped_tasklet(
        "comp1_2",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in1": dace.Memlet("x[__i]"),
            "__in2": dace.Memlet("y[__i]"),
        },
        code="__out = __in1 * 3.",
        outputs={
            "__out": dace.Memlet("t2[__i]"),
        },
        input_nodes={x1, y1},
        external_edges=True,
    )

    state2.add_mapped_tasklet(
        "comp2_1",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in1": dace.Memlet("t1[__i]"),
        },
        code="__out = __in1 + 1.",
        outputs={
            "__out": dace.Memlet("x[__i]"),
        },
        external_edges=True,
    )
    state2.add_mapped_tasklet(
        "comp2_1",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in1": dace.Memlet("t2[__i]"),
        },
        code="__out = __in1 + 2.",
        outputs={
            "__out": dace.Memlet("y[__i]"),
        },
        external_edges=True,
    )
    sdfg.validate()

    return sdfg, state1, state2


def _make_hidden_double_producer_sdfg() -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("hidden_double_producer_sdfg"))
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    for name in ["a", "b", "t1", "t2"]:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=name.startswith("t"),
        )

    a1, t2_1 = state1.add_access("a"), state1.add_access("t2")
    state1.add_mapped_tasklet(
        "comp",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet("t1[__i]")},
        input_nodes={a1},
        external_edges=True,
    )
    state1.add_nedge(a1, t2_1, dace.Memlet("a[0:10] -> [0:10]"))
    state1.add_nedge(t2_1, state1.add_access("b"), dace.Memlet("t2[0:10] -> [0:10]"))

    state2.add_nedge(
        state2.add_access("t1"), state2.add_access("a"), dace.Memlet("t1[0:10] -> [0:10]")
    )
    sdfg.validate()

    return sdfg, state1, state2


def test_simple_fusion():
    sdfg, state1, state2 = _make_simple_two_state_sdfg()
    assert sdfg.start_block is state1
    assert util.count_nodes(state1, dace_nodes.AccessNode) == 2
    assert util.count_nodes(state1, dace_nodes.MapEntry) == 1
    assert util.count_nodes(state2, dace_nodes.AccessNode) == 3
    assert util.count_nodes(state2, dace_nodes.MapEntry) == 2

    nb_applied = sdfg.apply_transformations_repeated(gtx_transformations.GT4PyStateFusion)
    sdfg.validate()

    assert nb_applied == 1
    assert sdfg.start_block is state1
    assert sdfg.number_of_nodes() == 1
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 4
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 3


def test_global_in_both_read_and_write():
    sdfg, state1, state2 = _make_global_in_both_read_and_write()

    nb_applied = sdfg.apply_transformations_repeated(gtx_transformations.GT4PyStateFusion)
    sdfg.validate()

    ac_nodes = util.count_nodes(state1, dace_nodes.AccessNode, return_nodes=True)
    assert nb_applied == 1
    assert sdfg.start_block is state1
    assert sdfg.number_of_nodes() == 1
    assert len(ac_nodes) == 3
    assert {"g", "t"} == {ac.data for ac in ac_nodes}
    assert all(state1.degree(g_node) for g_node in ac_nodes if g_node.data == "g")

    t_node = next(iter(ac for ac in ac_nodes if ac.data == "t"))
    assert state1.in_degree(t_node) == 1
    assert state1.out_degree(t_node) == 1


def test_global_in_both_state_reads():
    sdfg, state1, state2 = _make_global_both_state_read()
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 4

    nb_applied = sdfg.apply_transformations_repeated(gtx_transformations.GT4PyStateFusion)
    sdfg.validate()

    ac_nodes = util.count_nodes(state1, dace_nodes.AccessNode, return_nodes=True)
    assert nb_applied == 1
    assert sdfg.start_block is state1
    assert sdfg.number_of_nodes() == 1
    assert len(ac_nodes) == 3
    assert {"a", "b", "c"} == {ac.data for ac in ac_nodes}
    assert all(
        state1.out_degree(ac) == 2 and state1.in_degree(ac) == 0
        for ac in ac_nodes
        if ac.data == "a"
    )
    assert all(
        state1.in_degree(ac) == 1 and state1.out_degree(ac) == 0
        for ac in ac_nodes
        if ac.data != "a"
    )


def test_global_in_both_states_write():
    sdfg, state1, state2 = _make_global_both_state_write()
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 4

    # Note that this case is allowed, because the second state does not read from it.
    nb_applied = sdfg.apply_transformations_repeated(gtx_transformations.GT4PyStateFusion)
    sdfg.validate()

    ac_nodes = util.count_nodes(state1, dace_nodes.AccessNode, return_nodes=True)
    assert nb_applied == 1
    assert sdfg.start_block is state1
    assert sdfg.number_of_nodes() == 1
    assert len(ac_nodes) == 3
    assert {"a", "b", "c"} == {ac.data for ac in ac_nodes}
    assert all(
        state1.in_degree(ac) == 2 and state1.out_degree(ac) == 0
        for ac in ac_nodes
        if ac.data == "a"
    )
    assert all(
        state1.out_degree(ac) == 1 and state1.in_degree(ac) == 0
        for ac in ac_nodes
        if ac.data != "a"
    )


def test_empty_first_state():
    sdfg, state1, state2 = _make_empty_state(True)
    assert state1.number_of_nodes() == 0
    assert state2.number_of_nodes() == 5
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 2
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 1

    nb_applied = sdfg.apply_transformations_repeated(gtx_transformations.GT4PyStateFusion)
    sdfg.validate()

    assert nb_applied == 1
    assert sdfg.start_block is state1
    assert sdfg.number_of_nodes() == 1
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 2
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 1


def test_empty_second_state():
    sdfg, state1, state2 = _make_empty_state(False)
    assert state1.number_of_nodes() == 5
    assert state2.number_of_nodes() == 0
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 2
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 1

    nb_applied = sdfg.apply_transformations_repeated(gtx_transformations.GT4PyStateFusion)
    sdfg.validate()

    assert nb_applied == 1
    assert sdfg.start_block is state1
    assert sdfg.number_of_nodes() == 1
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 2
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 1


def test_both_states_are_empty():
    sdfg = dace.SDFG(util.unique_name("full_empty_sdfg"))
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 0

    nb_applied = sdfg.apply_transformations_repeated(gtx_transformations.GT4PyStateFusion)
    sdfg.validate()

    assert nb_applied == 1
    assert sdfg.start_block is state1
    assert sdfg.number_of_nodes() == 1
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 0


def test_global_merge_1():
    sdfg = _make_global_merge_1()

    nb_applied = sdfg.apply_transformations_repeated(gtx_transformations.GT4PyStateFusion)
    sdfg.validate()

    assert nb_applied == 1
    assert sdfg.number_of_nodes() == 1

    state = sdfg.nodes()[0]
    source_nodes = state.source_nodes()
    assert len(source_nodes) == 1
    source_node = source_nodes[0]
    assert source_node.data == "b"

    # ensure that `b` is connected to the `comp1` map.
    assert state.out_degree(source_node) == 1
    assert all(oedge.dst.map.label == "comp1_map" for oedge in state.out_edges(source_node))

    # Ensure that the sink node is supplied from the `comp2` map.
    sink_nodes = state.sink_nodes()
    assert len(sink_nodes) == 1
    sink_node = sink_nodes[0]
    assert sink_node.data == "a"
    assert state.in_degree(sink_node) == 1
    assert all(iedge.src.map.label == "comp2_map" for iedge in state.in_edges(sink_node))

    # Ensures that the node between the two maps is also an `a` node.
    intermediate_nodes = [dnode for dnode in state.data_nodes() if state.degree(dnode) == 2]
    assert len(intermediate_nodes) == 1
    intermediate_node = intermediate_nodes[0]
    assert state.in_degree(intermediate_node) == 1

    assert all(edge.src.map.label == "comp1_map" for edge in state.in_edges(intermediate_node))
    assert all(edge.dst.map.label == "comp2_map" for edge in state.out_edges(intermediate_node))


def test_global_merge_2():
    sdfg = _make_global_merge_2()

    # The difficulty here is that the two maps are correctly merged.
    nb_applied = sdfg.apply_transformations_repeated(gtx_transformations.GT4PyStateFusion)
    sdfg.validate()

    assert nb_applied == 1
    assert sdfg.number_of_nodes() == 1

    state = sdfg.nodes()[0]
    source_nodes = state.source_nodes()
    assert len(source_nodes) == 1
    source_node = source_nodes[0]
    assert source_node.data == "a"

    # ensure that the `a` source node is connected to the `comp1` map.
    assert state.out_degree(source_node) == 1
    assert all(oedge.dst.map.label == "comp1_map" for oedge in state.out_edges(source_node))

    # Ensure that the sink node is supplied from the `comp2` map.
    sink_nodes = state.sink_nodes()
    assert len(sink_nodes) == 1
    sink_node = sink_nodes[0]
    assert sink_node.data == "a"
    assert state.in_degree(sink_node) == 1
    assert all(iedge.src.map.label == "comp2_map" for iedge in state.in_edges(sink_node))

    # Ensures that the node between the two maps is also an `a` node.
    intermediate_nodes = [dnode for dnode in state.data_nodes() if state.degree(dnode) == 2]
    assert len(intermediate_nodes) == 1
    intermediate_node = intermediate_nodes[0]
    assert state.in_degree(intermediate_node) == 1

    assert all(edge.src.map.label == "comp1_map" for edge in state.in_edges(intermediate_node))
    assert all(edge.dst.map.label == "comp2_map" for edge in state.out_edges(intermediate_node))


def test_swapping():
    sdfg = _make_swapping_sdfg()

    # The transformation does not apply. If it would apply then we would have the two
    #  independent dataflows `(x) -> (t1) -> (y)` and `(y) -> (t2) -> (x)` and because
    #  the execution is arbitrary it might or might not work.
    nb_applied = sdfg.apply_transformations_repeated(gtx_transformations.GT4PyStateFusion)
    sdfg.validate()

    assert nb_applied == 0


def test_non_concurrent_data_dependency():
    sdfg, state1, state2 = _make_non_concurrent_sdfg()

    # Because the intermediate are generate in one go, they should be considered as
    #  one and thus the transformation should apply.
    nb_applied = sdfg.apply_transformations_repeated(gtx_transformations.GT4PyStateFusion)
    sdfg.validate()

    assert nb_applied == 1
    assert sdfg.start_block is state1
    assert sdfg.number_of_nodes() == 1
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 6
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2


def test_double_producer():
    sdfg, state1, state2 = _make_double_producer_sdfg()
    assert all(
        all(
            state.degree(ac) == 1
            for ac in util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
        )
        for state in [state1, state2]
    )

    nb_applied = sdfg.apply_transformations_repeated(gtx_transformations.GT4PyStateFusion)
    sdfg.validate()

    ac_nodes: list[dace_nodes.AccessNode] = util.count_nodes(
        sdfg, dace_nodes.AccessNode, return_nodes=True
    )
    assert nb_applied == 1
    assert sdfg.start_block is state1
    assert sdfg.number_of_nodes() == 1
    assert len(ac_nodes) == 6
    assert all(state1.degree(ac) == (2 if ac.data.startswith("t") else 1) for ac in ac_nodes)


def test_double_consumer():
    sdfg, state1, state2 = _make_double_consumer_sdfg()
    nb_applied = sdfg.apply_transformations_repeated(gtx_transformations.GT4PyStateFusion)
    sdfg.validate()

    # The transformation does not apply, because it would be indeterministic, i.e.
    #  `x` could be updated before it is read to compute the final value for `y`.
    assert nb_applied == 0


def test_hidden_double_producer():
    sdfg, state1, state2 = _make_hidden_double_producer_sdfg()

    nb_applied = sdfg.apply_transformations_repeated(gtx_transformations.GT4PyStateFusion)
    sdfg.validate()

    assert nb_applied == 0
