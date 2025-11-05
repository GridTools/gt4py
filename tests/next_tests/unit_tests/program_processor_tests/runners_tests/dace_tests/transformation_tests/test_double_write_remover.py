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

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _make_double_write_single_consumer(
    slice_to_second: bool,
) -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.AccessNode, dace_nodes.MapExit
]:
    sdfg = dace.SDFG(util.unique_name("double_write_elimination_1"))
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_array(
        "a",
        shape=(10,),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "b",
        shape=(11,),
        dtype=dace.float64,
        transient=True,
    )
    sdfg.add_array(
        "c",
        shape=((15, 4) if slice_to_second else (4, 15)),
        dtype=dace.float64,
        transient=False,
    )
    a, b, c = (state.add_access(name) for name in "abc")

    _, _, mx = state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i": "3:13"},
        inputs={"__in": dace.Memlet("a[__i - 3]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("b[__i - 2]")},
        external_edges=True,
        input_nodes={a},
        output_nodes={b},
    )

    if slice_to_second:
        state.add_nedge(b, c, dace.Memlet("b[1:11] -> [0:10, 0]"))
        state.add_nedge(b, c, dace.Memlet("b[1:11] -> [1:11, 2]"))
    else:
        state.add_nedge(b, c, dace.Memlet("b[1:11] -> [0, 0:10]"))
        state.add_nedge(b, c, dace.Memlet("b[1:11] -> [2, 1:11]"))

    sdfg.validate()
    return sdfg, state, b, c, mx


@pytest.mark.parametrize("slice_to_second", [True, False])
def test_remove_double_write_single_consumer(slice_to_second: bool):
    sdfg, state, b, c, mx = _make_double_write_single_consumer(slice_to_second)

    pre_ac = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert len(pre_ac) == 3
    assert b in pre_ac
    assert c in pre_ac
    assert b.data in sdfg.arrays
    assert c.data in sdfg.arrays
    assert util.count_nodes(sdfg, dace_nodes.MapExit) == 1
    assert state.out_degree(mx) == 1
    assert state.out_degree(b) == 2
    assert state.in_degree(b) == 1

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    nb_applied = sdfg.apply_transformations_repeated(
        gtx_transformations.DoubleWriteRemover(
            single_use_data=None,
        ),
        validate_all=True,
        validate=True,
    )
    assert nb_applied == 1

    after_ac = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert b not in after_ac
    assert b.data not in sdfg.arrays
    # Because there is a new scalar one that was added.
    assert len(after_ac) == 3

    assert util.count_nodes(sdfg, dace_nodes.MapExit) == 1
    assert state.out_degree(mx) == 2
    assert all(oedge.dst is c for oedge in state.out_edges(mx))

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)


def _make_double_write_multi_consumer(
    slice_to_second1: bool,
    slice_to_second2: bool,
) -> tuple[
    dace.SDFG,
    dace.SDFGState,
    dace_nodes.AccessNode,
    dace_nodes.AccessNode,
    dace_nodes.AccessNode,
    dace_nodes.MapExit,
]:
    sdfg = dace.SDFG(util.unique_name("double_write_elimination_2"))
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
    sdfg.add_array(
        "c",
        shape=((10, 4) if slice_to_second1 else (4, 10)),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "d",
        shape=((10, 3) if slice_to_second2 else (3, 10)),
        dtype=dace.float64,
        transient=False,
    )
    a, b, c, d = (state.add_access(name) for name in "abcd")

    _, _, mx = state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("b[__i]")},
        external_edges=True,
        input_nodes={a},
        output_nodes={b},
    )

    if slice_to_second1:
        state.add_nedge(b, c, dace.Memlet("b[0:10] -> [0:10, 0]"))
        state.add_nedge(b, c, dace.Memlet("b[0:10] -> [0:10, 2]"))
    else:
        state.add_nedge(b, c, dace.Memlet("b[0:10] -> [0, 0:10]"))
        state.add_nedge(b, c, dace.Memlet("b[0:10] -> [2, 0:10]"))
    if slice_to_second2:
        state.add_nedge(b, d, dace.Memlet("b[0:10] -> [0:10, 1]"))
    else:
        state.add_nedge(b, d, dace.Memlet("b[0:10] -> [1, 0:10]"))

    sdfg.validate()
    return sdfg, state, b, c, d, mx


@pytest.mark.parametrize("slice_to_second1", [True, False])
@pytest.mark.parametrize("slice_to_second2", [True, False])
def test_remove_double_write_multi_consumer(
    slice_to_second1: bool,
    slice_to_second2: bool,
):
    sdfg, state, b, c, d, mx = _make_double_write_multi_consumer(
        slice_to_second1=slice_to_second1,
        slice_to_second2=slice_to_second2,
    )

    pre_ac = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert len(pre_ac) == 4
    assert b in pre_ac
    assert b.data in sdfg.arrays
    assert util.count_nodes(sdfg, dace_nodes.MapExit) == 1
    assert state.out_degree(mx) == 1
    assert state.out_degree(b) == 3
    assert state.in_degree(b) == 1

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    nb_applied = sdfg.apply_transformations_repeated(
        gtx_transformations.DoubleWriteRemover(
            single_use_data=None,
        ),
        validate_all=True,
        validate=True,
    )
    assert nb_applied == 1

    after_ac = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert b not in after_ac
    assert b.data not in sdfg.arrays
    assert len(after_ac) == 4

    assert util.count_nodes(sdfg, dace_nodes.MapExit) == 1
    assert state.out_degree(mx) == 3
    assert all(oedge.dst in [c, d] for oedge in state.out_edges(mx))
    assert state.in_degree(c) == 2
    assert state.in_degree(d) == 1

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)


def _make_double_write_multi_producer_map(
    slice_to_second: bool,
) -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.AccessNode, dace_nodes.MapExit
]:
    sdfg = dace.SDFG(util.unique_name("double_write_elimination_multi_producer_map"))
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
    sdfg.add_array(
        "c",
        shape=((15, 4) if slice_to_second else (4, 15)),
        dtype=dace.float64,
        transient=False,
    )
    a, b, c = (state.add_access(name) for name in "abc")

    me, mx = state.add_map("comp_map", ndrange={"__i": "0:10"})
    tlet1 = state.add_tasklet(
        "tlet1",
        inputs={"__in"},
        outputs={"__out"},
        code="__out = __in + 1.0",
    )
    tlet2 = state.add_tasklet(
        "tlet2",
        inputs={"__in"},
        outputs={"__out"},
        code="__out = __in + 2.0",
    )

    state.add_edge(a, None, me, "IN_a1", dace.Memlet("a[0:10]"))
    state.add_edge(me, "OUT_a1", tlet1, "__in", dace.Memlet("a[__i]"))
    me.add_scope_connectors("a1")
    state.add_edge(tlet1, "__out", mx, "IN_b", dace.Memlet("b[__i]"))
    state.add_edge(mx, "OUT_b", b, None, dace.Memlet("b[0:10]"))
    mx.add_scope_connectors("b")

    state.add_edge(a, None, me, "IN_a2", dace.Memlet("a[0:10]"))
    state.add_edge(me, "OUT_a2", tlet2, "__in", dace.Memlet("a[__i]"))
    me.add_scope_connectors("a2")
    state.add_edge(
        tlet2, "__out", mx, "IN_c", dace.Memlet("c[__i, 1]" if slice_to_second else "c[1, __i]")
    )
    state.add_edge(
        mx, "OUT_c", c, None, dace.Memlet("c[0:10, 1]" if slice_to_second else "c[1, 0:10]")
    )
    mx.add_scope_connectors("c")

    if slice_to_second:
        state.add_nedge(b, c, dace.Memlet("b[0:10] -> [1:11, 0]"))
        state.add_nedge(b, c, dace.Memlet("b[0:10] -> [0:10, 2]"))
    else:
        state.add_nedge(b, c, dace.Memlet("b[0:10] -> [0, 1:11]"))
        state.add_nedge(b, c, dace.Memlet("b[0:10] -> [2, 0:10]"))

    sdfg.validate()
    return sdfg, state, b, c, mx


# @pytest.mark.parametrize("slice_to_second", [True, False])
def test_remove_double_write_multi_producer():  # slice_to_second: bool):
    slice_to_second = True
    sdfg, state, b, c, mx = _make_double_write_multi_producer_map(slice_to_second)

    pre_ac = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert len(pre_ac) == 3
    assert b in pre_ac
    assert c in pre_ac
    assert b.data in sdfg.arrays
    assert c.data in sdfg.arrays
    assert util.count_nodes(sdfg, dace_nodes.MapExit) == 1
    assert state.out_degree(mx) == 2
    assert state.out_degree(b) == 2
    assert state.in_degree(b) == 1
    assert state.in_degree(c) == 3

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    nb_applied = sdfg.apply_transformations_repeated(
        gtx_transformations.DoubleWriteRemover(
            single_use_data=None,
        ),
        validate_all=True,
        validate=True,
    )
    assert nb_applied == 1

    after_ac = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert b not in after_ac
    assert b.data not in sdfg.arrays
    # Because there is a new scalar one that was added.
    assert len(after_ac) == 3

    assert util.count_nodes(sdfg, dace_nodes.MapExit) == 1
    assert state.out_degree(mx) == 3
    assert state.in_degree(c) == 3
    assert all(oedge.dst is c for oedge in state.out_edges(mx))

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)
