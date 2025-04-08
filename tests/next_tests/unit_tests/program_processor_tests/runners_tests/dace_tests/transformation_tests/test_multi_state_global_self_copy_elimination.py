# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from typing import Optional

import dace
from dace.sdfg import nodes as dace_nodes
from dace.transformation import pass_pipeline as dace_ppl

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util

import dace


def apply_distributed_self_copy_elimination(
    sdfg: dace.SDFG,
) -> Optional[dict[dace.SDFG, set[str]]]:
    return gtx_transformations.gt_multi_state_global_self_copy_elimination(sdfg=sdfg, validate=True)


def _make_not_apply_because_of_write_to_g_sdfg() -> dace.SDFG:
    """This SDFG is not eligible, because there is a write to `G`."""
    sdfg = dace.SDFG(util.unique_name("not_apply_because_of_write_to_g_sdfg"))

    # This is the `G` array.
    sdfg.add_array(name="a", shape=(10,), dtype=dace.float64, transient=False)
    # This is the `T` array.
    sdfg.add_array(name="t", shape=(5,), dtype=dace.float64, transient=True)

    # This is an unrelated array that is used as output.
    sdfg.add_array(
        name="b",
        shape=(10,),
        dtype=dace.float64,
        transient=False,
    )

    state1 = sdfg.add_state(is_start_block=True)
    state1.add_nedge(state1.add_access("a"), state1.add_access("t"), dace.Memlet("a[0:5] -> [0:5]"))

    state2 = sdfg.add_state_after(state1)
    state2.add_mapped_tasklet(
        "make_a_non_applicable",
        map_ranges={"__i": "3:8"},
        inputs={},
        code="__out = 10.",
        outputs={"__out": dace.Memlet("a[__i]")},
        external_edges=True,
    )

    state3 = sdfg.add_state_after(state2)
    a3 = state3.add_access("a")
    state3.add_nedge(state3.add_access("t"), a3, dace.Memlet("t[0:5] -> [0:5]"))
    state3.add_mapped_tasklet(
        "comp",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 1.",
        outputs={"__out": dace.Memlet("b[__i]")},
        input_nodes={a3},
        external_edges=True,
    )

    sdfg.validate()
    return sdfg


def _make_eligible_sdfg_1() -> dace.SDFG:
    """This SDFG is very similar to the one generated by `_make_not_apply_because_of_write_to_g_sdfg()`.

    The main difference is that there is no mutating write to `a` and thus the
    transformation applies.
    """
    sdfg = dace.SDFG(util.unique_name("_make_eligible_sdfg_1"))

    # This is the `G` array.
    sdfg.add_array(name="a", shape=(10,), dtype=dace.float64, transient=False)
    # This is the `T` array.
    sdfg.add_array(name="t", shape=(5,), dtype=dace.float64, transient=True)

    # These are some  unrelated arrays that is used as output.
    sdfg.add_array(name="b", shape=(10,), dtype=dace.float64, transient=False)
    sdfg.add_array(name="c", shape=(10,), dtype=dace.float64, transient=False)

    state1 = sdfg.add_state(is_start_block=True)
    state1.add_nedge(state1.add_access("a"), state1.add_access("t"), dace.Memlet("a[0:5] -> [0:5]"))

    state2 = sdfg.add_state_after(state1)
    state2.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 1.",
        outputs={"__out": dace.Memlet("b[__i]")},
        external_edges=True,
    )

    state3 = sdfg.add_state_after(state2)
    a3 = state3.add_access("a")
    state3.add_nedge(state3.add_access("t"), a3, dace.Memlet("t[0:5] -> [0:5]"))
    state3.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 1.",
        outputs={"__out": dace.Memlet("c[__i]")},
        input_nodes={a3},
        external_edges=True,
    )

    sdfg.validate()
    return sdfg


def _make_multiple_temporaries_sdfg1() -> dace.SDFG:
    """Generates an SDFG in which `G` is saved into different temporaries."""
    sdfg = dace.SDFG(util.unique_name("multiple_temporaries"))

    # This is the `G` array.
    sdfg.add_array(name="a", shape=(10,), dtype=dace.float64, transient=False)
    # This is the first `T` array.
    sdfg.add_array(name="t1", shape=(5,), dtype=dace.float64, transient=True)
    # This is the second `T` array.
    sdfg.add_array(name="t2", shape=(5,), dtype=dace.float64, transient=True)

    # This are some unrelated array that is used as output.
    sdfg.add_array(name="b", shape=(10,), dtype=dace.float64, transient=False)

    state1 = sdfg.add_state(is_start_block=True)
    a1 = state1.add_access("a")
    state1.add_nedge(a1, state1.add_access("t1"), dace.Memlet("a[0:5] -> [0:5]"))
    state1.add_nedge(a1, state1.add_access("t2"), dace.Memlet("a[5:10] -> [0:5]"))

    state2 = sdfg.add_state_after(state1)
    a2 = state2.add_access("a")

    state2.add_nedge(state2.add_access("t1"), a2, dace.Memlet("t1[0:5] -> [0:5]"))
    state2.add_nedge(state2.add_access("t2"), a2, dace.Memlet("t2[0:5] -> [5:10]"))

    state2.add_mapped_tasklet(
        "comp",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 1.",
        outputs={"__out": dace.Memlet("b[__i]")},
        input_nodes={a2},
        external_edges=True,
    )

    sdfg.validate()
    return sdfg


def _make_multiple_temporaries_sdfg2() -> dace.SDFG:
    """Generates an SDFG where there are multiple `T` used.

    The main difference between the SDFG produced by this function and the one
    generated by `_make_multiple_temporaries_sdfg()` is that the temporaries
    are used sequentially.
    """
    sdfg = dace.SDFG(util.unique_name("multiple_temporaries_sequential"))

    # This is the `G` array.
    sdfg.add_array(name="a", shape=(10,), dtype=dace.float64, transient=False)
    # This is the first `T` array.
    sdfg.add_array(name="t1", shape=(5,), dtype=dace.float64, transient=True)
    # This is the second `T` array.
    sdfg.add_array(name="t2", shape=(5,), dtype=dace.float64, transient=True)

    # This are some unrelated array that is used as output.
    sdfg.add_array(name="b", shape=(10,), dtype=dace.float64, transient=False)
    sdfg.add_array(name="c", shape=(10,), dtype=dace.float64, transient=False)

    state1 = sdfg.add_state(is_start_block=True)
    state1.add_nedge(
        state1.add_access("a"), state1.add_access("t1"), dace.Memlet("a[0:5] -> [0:5]")
    )

    state2 = sdfg.add_state_after(state1)
    a2 = state2.add_access("a")

    state2.add_nedge(state2.add_access("t1"), a2, dace.Memlet("t1[0:5] -> [0:5]"))

    state2.add_mapped_tasklet(
        "comp",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 1.",
        outputs={"__out": dace.Memlet("b[__i]")},
        input_nodes={a2},
        external_edges=True,
    )

    # This essentially repeats the same thing as above again, but this time with `t2`.
    state3 = sdfg.add_state_after(state2)
    state3.add_nedge(
        state3.add_access("a"), state3.add_access("t2"), dace.Memlet("a[5:10] -> [0:5]")
    )

    state4 = sdfg.add_state_after(state3)
    a4 = state4.add_access("a")
    state4.add_nedge(state4.add_access("t2"), a4, dace.Memlet("t2[0:5] -> [5:10]"))
    state4.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in - 1.",
        outputs={"__out": dace.Memlet("c[__i]")},
        input_nodes={a4},
        external_edges=True,
    )

    sdfg.validate()
    return sdfg


def _make_multiple_temporaries_sdfg_keep_one_1() -> dace.SDFG:
    """
    The generated SDFG is very similar to `_make_multiple_temporaries_sdfg1()` except
    that `t1` can not be removed because it is used to generate `c`.
    """
    sdfg = _make_multiple_temporaries_sdfg1()

    sdfg.add_array("c", shape=(5,), dtype=dace.float64, transient=False)

    state = sdfg.add_state_after(
        next(iter(state for state in sdfg.states() if sdfg.out_degree(state) == 0))
    )
    state.add_mapped_tasklet(
        "comp_that_needs_t1",
        map_ranges={"__j": "0:5"},
        inputs={"__in": dace.Memlet("t1[__j]")},
        code="__out = __in + 4.0",
        outputs={"__out": dace.Memlet("c[__j]")},
        external_edges=True,
    )

    sdfg.validate()
    return sdfg


def _make_multiple_temporaries_sdfg_keep_one_2() -> dace.SDFG:
    """
    The generated SDFG is very similar to `_make_multiple_temporaries_sdfg2()` except
    that `t1` can not be removed because it is used to generate `d`.
    """
    sdfg = _make_multiple_temporaries_sdfg2()

    sdfg.add_array("d", shape=(5,), dtype=dace.float64, transient=False)

    state = sdfg.add_state_after(
        next(iter(state for state in sdfg.states() if sdfg.out_degree(state) == 0))
    )
    state.add_mapped_tasklet(
        "comp_that_needs_t1",
        map_ranges={"__j": "0:5"},
        inputs={"__in": dace.Memlet("t1[__j]")},
        code="__out = __in + 4.0",
        outputs={"__out": dace.Memlet("d[__j]")},
        external_edges=True,
    )

    sdfg.validate()
    return sdfg


def _make_non_eligible_because_of_pseudo_temporary() -> dace.SDFG:
    """Generates an SDFG that that defines `T` from two souces, which is not handled.

    Note that in this particular case it would be possible, but we do not support it.
    """
    sdfg = dace.SDFG(util.unique_name("multiple_temporaries_sequential"))

    # This is the `G` array.
    sdfg.add_array(name="a", shape=(10,), dtype=dace.float64, transient=False)
    # This is the `T` array.
    sdfg.add_array(name="t", shape=(10,), dtype=dace.float64, transient=True)

    # This is the array that also writes to `T` and thus makes it inapplicable.
    sdfg.add_array(name="pg", shape=(10,), dtype=dace.float64, transient=True)

    # This are some unrelated array that is used as output.
    sdfg.add_array(name="b", shape=(10,), dtype=dace.float64, transient=False)

    state1 = sdfg.add_state(is_start_block=True)
    t1 = state1.add_access("t")
    state1.add_nedge(state1.add_access("a"), t1, dace.Memlet("a[0:5] -> [0:5]"))
    state1.add_nedge(state1.add_access("pg"), t1, dace.Memlet("pg[0:5] -> [5:10]"))

    state2 = sdfg.add_state_after(state1)
    a2 = state2.add_access("a")
    state2.add_nedge(state2.add_access("t"), a2, dace.Memlet("t[0:5] -> [0:5]"))
    state2.add_mapped_tasklet(
        "comp",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("b[__i]")},
        input_nodes={a2},
        external_edges=True,
    )

    sdfg.validate()
    return sdfg


def _make_wb_single_state_sdfg() -> dace.SDFG:
    """Generates an SDFG with the pattern `(G) -> (T) -> (G)` which is not handled.

    This pattern is handled by the `SingleStateGlobalSelfCopyElimination` transformation.
    """
    sdfg = dace.SDFG(util.unique_name("single_state_write_back_sdfg"))

    sdfg.add_array("g", shape=(10,), dtype=dace.float64, transient=False)
    sdfg.add_array("t", shape=(10,), dtype=dace.float64, transient=True)
    sdfg.add_array("b", shape=(10,), dtype=dace.float64, transient=False)

    state1 = sdfg.add_state(is_start_block=True)
    t1 = state1.add_access("t")
    state1.add_nedge(state1.add_access("g"), t1, dace.Memlet("g[0:10] -> [0:10]"))
    g1 = state1.add_access("g")
    state1.add_nedge(t1, g1, dace.Memlet("t[0:10] -> [0:10]"))

    # return sdfg

    state1.add_mapped_tasklet(
        "comp",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("g[__i]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("b[__i]")},
        input_nodes={g1},
        external_edges=True,
    )

    sdfg.validate()
    return sdfg


def _make_non_eligible_sdfg_with_branches():
    """Creates an SDFG with two different definitions of `T`."""
    sdfg = dace.SDFG(util.unique_name("non_eligible_sdfg_with_branches_sdfg"))

    # This is the `G` array, it is also used as output.
    sdfg.add_array("a", shape=(10,), dtype=dace.float64, transient=False)
    # This is the (possible) `T` array.
    sdfg.add_array("t", shape=(10,), dtype=dace.float64, transient=True)

    # This is an additional array that serves as input. In one case it defines `t`.
    sdfg.add_array("b", shape=(10,), dtype=dace.float64, transient=False)
    # This is the condition on which we switch.
    sdfg.add_scalar("cond", dtype=dace.bool, transient=False)

    # This is the init state.
    state1 = sdfg.add_state(is_start_block=True)

    # This is the state where `T` is not defined in terms of `G`.
    stateT = sdfg.add_state(is_start_block=False)
    sdfg.add_edge(state1, stateT, dace.InterstateEdge(condition="cond == True"))
    stateT.add_nedge(
        stateT.add_access("b"), stateT.add_access("t"), dace.Memlet("b[0:10] -> [0:10]")
    )

    # This is the state where `T` is defined in terms of `G`.
    stateF = sdfg.add_state(is_start_block=False)
    sdfg.add_edge(state1, stateF, dace.InterstateEdge(condition="cond != True"))
    stateF.add_nedge(
        stateF.add_access("a"), stateF.add_access("t"), dace.Memlet("a[0:10] -> [0:10]")
    )

    # Now the write back state, where `T` is written back into `G`.
    stateWB = sdfg.add_state(is_start_block=False)
    stateWB.add_nedge(
        stateWB.add_access("t"), stateWB.add_access("a"), dace.Memlet("t[0:10] -> [0:10]")
    )

    sdfg.add_edge(stateF, stateWB, dace.InterstateEdge())
    sdfg.add_edge(stateT, stateWB, dace.InterstateEdge())

    sdfg.validate()
    return sdfg


def test_not_apply_because_of_write_to_g():
    sdfg = _make_not_apply_because_of_write_to_g_sdfg()
    old_hash = sdfg.hash_sdfg()
    nb_access_nodes_initially = util.count_nodes(sdfg, dace_nodes.AccessNode)

    res = apply_distributed_self_copy_elimination(sdfg)
    nb_access_nodes_after = util.count_nodes(sdfg, dace_nodes.AccessNode)

    assert res is None
    assert nb_access_nodes_initially == nb_access_nodes_after
    assert old_hash == sdfg.hash_sdfg()


def test_eligible_sdfg_1():
    sdfg = _make_eligible_sdfg_1()
    nb_access_nodes_initially = util.count_nodes(sdfg, dace_nodes.AccessNode)

    res = apply_distributed_self_copy_elimination(sdfg)
    access_nodes_after = util.count_nodes(sdfg, dace_nodes.AccessNode, return_nodes=True)

    assert res == {"a", "t"}
    assert nb_access_nodes_initially == len(access_nodes_after) + 3
    assert not any(an.data == "t" for an in access_nodes_after)


def test_multiple_temporaries():
    sdfg = _make_multiple_temporaries_sdfg1()
    nb_access_nodes_initially = util.count_nodes(sdfg, dace_nodes.AccessNode)

    res = apply_distributed_self_copy_elimination(sdfg)
    access_nodes_after = util.count_nodes(sdfg, dace_nodes.AccessNode, return_nodes=True)

    assert res == {"a", "t1", "t2"}
    assert not any(an.data.startswith("t") for an in access_nodes_after)
    assert nb_access_nodes_initially == len(access_nodes_after) + 5


def test_multiple_temporaries_2():
    sdfg = _make_multiple_temporaries_sdfg2()
    nb_access_nodes_initially = util.count_nodes(sdfg, dace_nodes.AccessNode)

    res = apply_distributed_self_copy_elimination(sdfg)
    access_nodes_after = util.count_nodes(sdfg, dace_nodes.AccessNode, return_nodes=True)

    assert res == {"a", "t1", "t2"}
    assert not any(an.data.startswith("t") for an in access_nodes_after)
    assert nb_access_nodes_initially == len(access_nodes_after) + 6


def test_multiple_temporaries_keep_one_1():
    sdfg = _make_multiple_temporaries_sdfg_keep_one_1()
    nb_access_nodes_initially = util.count_nodes(sdfg, dace_nodes.AccessNode)

    # NOTE: The transformation will not only remove the `(t2) -> (a)` write in the
    #  second block, but also the `(t1) -> (a)` write, this is because it was
    #  concluded that this was a noops write. This might be a bit unintuitive
    #  considering that `t1` is used in the third state. However, this is why the
    #  `(a) -> (t1)` write in the first state is maintained.
    res = apply_distributed_self_copy_elimination(sdfg)
    access_nodes_after = util.count_nodes(sdfg, dace_nodes.AccessNode, return_nodes=True)
    start_block_nodes = util.count_nodes(sdfg.start_block, dace_nodes.AccessNode, return_nodes=True)

    assert res == {"a", "t2"}
    assert not any(an.data == "t2" for an in access_nodes_after)
    assert sum(an.data == "t1" for an in access_nodes_after) == 2
    assert nb_access_nodes_initially == len(access_nodes_after) + 3
    assert len(start_block_nodes) == 2
    assert {nb.data for nb in start_block_nodes} == {"a", "t1"}


def test_multiple_temporaries_keep_one_2():
    sdfg = _make_multiple_temporaries_sdfg_keep_one_2()
    nb_access_nodes_initially = util.count_nodes(sdfg, dace_nodes.AccessNode)

    res = apply_distributed_self_copy_elimination(sdfg)
    access_nodes_after = util.count_nodes(sdfg, dace_nodes.AccessNode, return_nodes=True)

    assert res == {"a", "t2"}
    assert not any(an.data == "t2" for an in access_nodes_after)
    assert sum(an.data == "t1" for an in access_nodes_after) == 2
    assert nb_access_nodes_initially == len(access_nodes_after) + 4


def test_pseudo_temporaries():
    sdfg = _make_non_eligible_because_of_pseudo_temporary()
    old_hash = sdfg.hash_sdfg()
    nb_access_nodes_initially = util.count_nodes(sdfg, dace_nodes.AccessNode)

    res = apply_distributed_self_copy_elimination(sdfg)
    nb_access_nodes_after = util.count_nodes(sdfg, dace_nodes.AccessNode)

    assert res is None
    assert nb_access_nodes_initially == nb_access_nodes_after
    assert old_hash == sdfg.hash_sdfg()


def test_single_state():
    sdfg = _make_wb_single_state_sdfg()
    old_hash = sdfg.hash_sdfg()
    nb_access_nodes_initially = util.count_nodes(sdfg, dace_nodes.AccessNode)

    res = apply_distributed_self_copy_elimination(sdfg)
    nb_access_nodes_after = util.count_nodes(sdfg, dace_nodes.AccessNode)

    assert res is None
    assert nb_access_nodes_initially == nb_access_nodes_after
    assert old_hash == sdfg.hash_sdfg()


def test_branches():
    sdfg = _make_non_eligible_sdfg_with_branches()
    old_hash = sdfg.hash_sdfg()
    nb_access_nodes_initially = util.count_nodes(sdfg, dace_nodes.AccessNode)

    res = apply_distributed_self_copy_elimination(sdfg)
    nb_access_nodes_after = util.count_nodes(sdfg, dace_nodes.AccessNode)

    assert res is None
    assert nb_access_nodes_initially == nb_access_nodes_after
    assert old_hash == sdfg.hash_sdfg()
