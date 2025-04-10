# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import copy


dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes, propagation as dace_propagation
from dace.transformation import dataflow as dace_dataflow

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _make_if_block(
    state: dace.SDFGState,
    outer_sdfg: dace.SDFG,
) -> dace_nodes.NestedSDFG:
    inner_sdfg = dace.SDFG(util.unique_name("inner_sdfg"))

    for name in ["__arg1", "__arg2", "__output", "__cond"]:
        inner_sdfg.add_scalar(
            name,
            dtype=dace.bool_ if name == "__cond" else dace.float64,
            transient=False,
        )

    if_region = dace.sdfg.state.ConditionalBlock("if")
    inner_sdfg.add_node(if_region, is_start_block=True)

    then_body = dace.sdfg.state.ControlFlowRegion("then_body", sdfg=inner_sdfg)
    tstate = then_body.add_state("true_branch", is_start_block=True)
    tstate.add_nedge(
        tstate.add_access("__arg1"),
        tstate.add_access("__output"),
        dace.Memlet("__arg1[0] -> [0]"),
    )

    else_body = dace.sdfg.state.ControlFlowRegion("else_body", sdfg=inner_sdfg)
    fstate = else_body.add_state("false_branch", is_start_block=True)
    fstate.add_nedge(
        fstate.add_access("__arg2"),
        fstate.add_access("__output"),
        dace.Memlet("__arg2[0] -> [0]"),
    )

    if_region.add_branch(dace.sdfg.state.CodeBlock("__cond"), then_body)
    if_region.add_branch(dace.sdfg.state.CodeBlock("not __cond"), else_body)

    return state.add_nested_sdfg(
        sdfg=inner_sdfg,
        parent=outer_sdfg,
        inputs={"__arg1", "__arg2", "__cond"},
        outputs={"__output"},
    )


def _perform_test(
    sdfg: dace.SDFG,
    explected_applies: int,
) -> None:
    ref = {
        name: np.array(np.random.rand(*desc.shape), copy=True, dtype=desc.dtype.as_numpy_dtype())
        for name, desc in sdfg.arrays.items()
        if not desc.transient
    }
    res = copy.deepcopy(ref)

    if explected_applies != 0:
        csdfg_ref = sdfg.compile()
        csdfg_ref(**ref)

    nb_apply = sdfg.apply_transformations_repeated(
        gtx_transformations.MoveDataflowIntoIfBody(),
        validate=True,
        validate_all=True,
    )
    assert nb_apply == explected_applies

    if explected_applies == 0:
        return

    csdfg_res = sdfg.compile()
    csdfg_res(**res)

    assert all(np.allclose(ref[name], res[name]) for name in ref.keys())


def test_if_mover_independent_branches():
    """
    Essentially tests the following situation:
    ```python
    a = foo(...)
    b = bar(...)
    c = baz(...)
    if c:
        d = a
    else:
        d = b
    ```
    """
    sdfg = dace.SDFG(util.unique_name("if_mover_independent_branches"))
    state = sdfg.add_state(is_start_block=True)

    # Inputs
    input_names = ["a", "b", "c", "d"]
    for name in input_names:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )

    # Temporaries
    temporary_names = ["a1", "a2", "b1", "b2", "c1"]
    for name in temporary_names:
        sdfg.add_scalar(
            name, dtype=dace.bool_ if name.startswith("c") else dace.float64, transient=True
        )

    a1, a2, b1, b2, c1 = (state.add_access(name) for name in temporary_names)
    me, mx = state.add_map("comp", ndrange={"__i": "0:10"})

    # Computation involving `a`:
    tasklet_a1 = state.add_tasklet(
        "tasklet_a1", inputs={"__in"}, outputs={"__out"}, code="__out = math.sin(__in)"
    )
    tasklet_a2 = state.add_tasklet(
        "tasklet_a2", inputs={"__in"}, outputs={"__out"}, code="__out = math.exp(__in)"
    )
    state.add_edge(state.add_access("a"), None, me, "IN_a", dace.Memlet("a[0:10]"))
    state.add_edge(me, "OUT_a", tasklet_a1, "__in", dace.Memlet("a[__i]"))
    state.add_edge(tasklet_a1, "__out", a1, None, dace.Memlet("a1[0]"))
    state.add_edge(a1, None, tasklet_a2, "__in", dace.Memlet("a1[0]"))
    state.add_edge(tasklet_a2, "__out", a2, None, dace.Memlet("a2[0]"))

    # Computation involving `b`:
    tasklet_b1 = state.add_tasklet(
        "tasklet_b1", inputs={"__in"}, outputs={"__out"}, code="__out = math.exp(__in)"
    )
    tasklet_b2 = state.add_tasklet(
        "tasklet_b2", inputs={"__in"}, outputs={"__out"}, code="__out = math.sin(__in)"
    )

    state.add_edge(state.add_access("b"), None, me, "IN_b", dace.Memlet("b[0:10]"))
    state.add_edge(me, "OUT_b", tasklet_b1, "__in", dace.Memlet("b[__i]"))
    state.add_edge(tasklet_b1, "__out", b1, None, dace.Memlet("b1[0]"))
    state.add_edge(b1, None, tasklet_b2, "__in", dace.Memlet("b1[0]"))
    state.add_edge(tasklet_b2, "__out", b2, None, dace.Memlet("b2[0]"))

    # Now the condition.
    tasklet_cond = state.add_tasklet(
        "tasklet_cond",
        inputs={"__in"},
        outputs={"__out"},
        code="__out = __in <= 0.5",
    )
    state.add_edge(state.add_access("c"), None, me, "IN_c", dace.Memlet("c[0:10]"))
    state.add_edge(me, "OUT_c", tasklet_cond, "__in", dace.Memlet("c[__i]"))
    state.add_edge(tasklet_cond, "__out", c1, None, dace.Memlet("c1[0]"))

    # Make the if selection.
    if_block = _make_if_block(state=state, outer_sdfg=sdfg)
    state.add_edge(a2, None, if_block, "__arg1", dace.Memlet("a2[0]"))
    state.add_edge(b2, None, if_block, "__arg2", dace.Memlet("b2[0]"))
    state.add_edge(c1, None, if_block, "__cond", dace.Memlet("c1[0]"))

    # Now handle the output.
    state.add_edge(if_block, "__output", mx, "IN_d", dace.Memlet("d[__i]"))
    state.add_edge(mx, "OUT_d", state.add_access("d"), None, dace.Memlet("d[0:10]"))

    # Now add the connectors to the Map*
    for iname in input_names:
        if iname == "d":
            continue
        me.add_in_connector(f"IN_{iname}")
        me.add_out_connector(f"OUT_{iname}")
    mx.add_in_connector("IN_d")
    mx.add_out_connector("OUT_d")
    sdfg.validate()

    _perform_test(sdfg, explected_applies=1)

    # Examine the structure of the SDFG.
    top_ac: list[dace_nodes.AccessNode] = util.count_nodes(state, dace_nodes.AccessNode, True)
    assert {ac.data for ac in top_ac} == set(input_names).union(["c1"])
    assert len(sdfg.arrays) == len(top_ac)

    top_tlet: list[dace_nodes.Tasklet] = util.count_nodes(state, dace_nodes.Tasklet, True)
    assert len(top_tlet) == 1
    assert "tasklet_cond" == top_tlet[0].label

    inner_ac: list[dace_nodes.AccessNode] = util.count_nodes(
        if_block.sdfg, dace_nodes.AccessNode, True
    )
    expected_data: set[str] = (
        set(temporary_names).union(input_names).union(["__arg1", "__arg2", "__output"])
    )
    expected_data.difference_update(["c1", "c", "d"])
    assert expected_data == {ac.data for ac in inner_ac}
    assert len([ac for ac in inner_ac if ac.data == "__output"]) == 2
    assert len(expected_data) + 1 == len(inner_ac)
    assert if_block.sdfg.arrays.keys() == expected_data.union(["__cond"])

    inner_tlet: list[dace_nodes.Tasklet] = util.count_nodes(if_block.sdfg, dace_nodes.Tasklet, True)
    assert len(inner_tlet) == 4
    expected_tlet = {tlet.label for tlet in [tasklet_a1, tasklet_a2, tasklet_b1, tasklet_b2]}
    assert {tlet.label for tlet in inner_tlet} == expected_tlet


def test_if_mover_dependent_branch_1():
    """
    Essentially tests the following situation:
    ```python
    s = buu(...)
    a = foo(s, ...)
    b = bar(s, ...)
    c = baz(...)
    if c:
        d = a
    else:
        d = b
    ```
    """
    sdfg = dace.SDFG(util.unique_name("if_mover_dependent_branches"))
    state = sdfg.add_state(is_start_block=True)

    # Inputs
    input_names = ["a", "b", "c", "d", "s"]
    for name in input_names:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )

    # Temporaries
    temporary_names = ["a1", "a2", "b1", "b2", "c1", "s1"]
    for name in temporary_names:
        sdfg.add_scalar(
            name, dtype=dace.bool_ if name.startswith("c") else dace.float64, transient=True
        )

    a1, a2, b1, b2, c1, s1 = (state.add_access(name) for name in temporary_names)
    me, mx = state.add_map("comp", ndrange={"__i": "0:10"})

    # The auxiliary computation involving `s`:
    tasklet_s1 = state.add_tasklet(
        "tasklet_s1", inputs={"__in"}, outputs={"__out"}, code="__out = - __in"
    )

    state.add_edge(state.add_access("s"), None, me, "IN_s", dace.Memlet("s[0:10]"))
    state.add_edge(me, "OUT_s", tasklet_s1, "__in", dace.Memlet("s[__i]"))
    state.add_edge(tasklet_s1, "__out", s1, None, dace.Memlet("s1[0]"))

    # Computation involving `a`:
    tasklet_a1 = state.add_tasklet(
        "tasklet_a1",
        inputs={"__in", "__in_s"},
        outputs={"__out"},
        code="__out = math.sin(__in) + __in_s",
    )
    tasklet_a2 = state.add_tasklet(
        "tasklet_a2", inputs={"__in"}, outputs={"__out"}, code="__out = math.exp(__in)"
    )
    state.add_edge(state.add_access("a"), None, me, "IN_a", dace.Memlet("a[0:10]"))
    state.add_edge(me, "OUT_a", tasklet_a1, "__in", dace.Memlet("a[__i]"))
    state.add_edge(s1, None, tasklet_a1, "__in_s", dace.Memlet("s1[0]"))
    state.add_edge(tasklet_a1, "__out", a1, None, dace.Memlet("a1[0]"))
    state.add_edge(a1, None, tasklet_a2, "__in", dace.Memlet("a1[0]"))
    state.add_edge(tasklet_a2, "__out", a2, None, dace.Memlet("a2[0]"))

    # Computation involving `b`:
    tasklet_b1 = state.add_tasklet(
        "tasklet_b1", inputs={"__in"}, outputs={"__out"}, code="__out = math.exp(__in)"
    )
    tasklet_b2 = state.add_tasklet(
        "tasklet_b2",
        inputs={"__in", "__in_s"},
        outputs={"__out"},
        code="__out = math.sin(__in) - __in_s",
    )

    state.add_edge(state.add_access("b"), None, me, "IN_b", dace.Memlet("b[0:10]"))
    state.add_edge(me, "OUT_b", tasklet_b1, "__in", dace.Memlet("b[__i]"))
    state.add_edge(tasklet_b1, "__out", b1, None, dace.Memlet("b1[0]"))
    state.add_edge(b1, None, tasklet_b2, "__in", dace.Memlet("b1[0]"))
    state.add_edge(s1, None, tasklet_b2, "__in_s", dace.Memlet("s1[0]"))
    state.add_edge(tasklet_b2, "__out", b2, None, dace.Memlet("b2[0]"))

    # Now the condition.
    tasklet_cond = state.add_tasklet(
        "tasklet_cond",
        inputs={"__in"},
        outputs={"__out"},
        code="__out = __in <= 0.5",
    )
    state.add_edge(state.add_access("c"), None, me, "IN_c", dace.Memlet("c[0:10]"))
    state.add_edge(me, "OUT_c", tasklet_cond, "__in", dace.Memlet("c[__i]"))
    state.add_edge(tasklet_cond, "__out", c1, None, dace.Memlet("c1[0]"))

    # Make the if selection.
    if_block = _make_if_block(state=state, outer_sdfg=sdfg)
    state.add_edge(a2, None, if_block, "__arg1", dace.Memlet("a2[0]"))
    state.add_edge(b2, None, if_block, "__arg2", dace.Memlet("b2[0]"))
    state.add_edge(c1, None, if_block, "__cond", dace.Memlet("c1[0]"))

    # Now handle the output.
    state.add_edge(if_block, "__output", mx, "IN_d", dace.Memlet("d[__i]"))
    state.add_edge(mx, "OUT_d", state.add_access("d"), None, dace.Memlet("d[0:10]"))

    # Now add the connectors to the Map*
    for iname in input_names:
        if iname == "d":
            continue
        me.add_in_connector(f"IN_{iname}")
        me.add_out_connector(f"OUT_{iname}")
    mx.add_in_connector("IN_d")
    mx.add_out_connector("OUT_d")
    sdfg.validate()

    _perform_test(sdfg, explected_applies=1)

    # Examine the structure of the SDFG.
    top_ac: list[dace_nodes.AccessNode] = util.count_nodes(state, dace_nodes.AccessNode, True)
    assert {ac.data for ac in top_ac} == set(input_names).union(["c1", "s1"])
    assert len(sdfg.arrays) == len(top_ac)

    top_tlet: list[dace_nodes.Tasklet] = util.count_nodes(state, dace_nodes.Tasklet, True)
    assert len(top_tlet) == 2
    assert {"tasklet_cond", "tasklet_s1"} == {tlet.label for tlet in top_tlet}

    inner_ac: list[dace_nodes.AccessNode] = util.count_nodes(
        if_block.sdfg, dace_nodes.AccessNode, True
    )
    expected_data: set[str] = (
        set(temporary_names).union(input_names).union(["__arg1", "__arg2", "__output"])
    )
    expected_data.difference_update(["c1", "c", "d", "s"])
    assert expected_data == {ac.data for ac in inner_ac}
    assert len([ac for ac in inner_ac if ac.data == "s1"]) == 2
    assert len([ac for ac in inner_ac if ac.data == "__output"]) == 2
    assert len(expected_data) + 2 == len(inner_ac)
    assert if_block.sdfg.arrays.keys() == expected_data.union(["__cond"])

    inner_tlet: list[dace_nodes.Tasklet] = util.count_nodes(if_block.sdfg, dace_nodes.Tasklet, True)
    assert len(inner_tlet) == 4
    expected_tlet = {tlet.label for tlet in [tasklet_a1, tasklet_a2, tasklet_b1, tasklet_b2]}
    assert {tlet.label for tlet in inner_tlet} == expected_tlet


def test_if_mover_dependent_branch_2():
    """
    Essentially tests the following situation:
    ```python
    a1 = foo(a, b)
    b1 = bar(a)
    c = baz(...)
    if c:
        d = a1
    else:
        d = b1
    ```
    """
    sdfg = dace.SDFG(util.unique_name("if_mover_dependent_branches_2"))
    state = sdfg.add_state(is_start_block=True)

    # Inputs
    input_names = ["a", "b", "c", "d"]
    for name in input_names:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )

    # Temporaries
    temporary_names = ["a1", "b1", "c1"]
    for name in temporary_names:
        sdfg.add_scalar(
            name, dtype=dace.bool_ if name.startswith("c") else dace.float64, transient=True
        )

    a1, b1, c1 = (state.add_access(name) for name in temporary_names)
    a, b, c, d = (state.add_access(name) for name in input_names)
    me, mx = state.add_map("comp", ndrange={"__i": "0:10"})

    # Computing of `a1`:
    tasklet_a1 = state.add_tasklet(
        "tasklet_a1",
        inputs={"__in1", "__in2"},
        outputs={"__out"},
        code="__out = math.sin(__in1) + __in2",
    )
    state.add_edge(a, None, me, "IN_a", dace.Memlet("a[0:10]"))
    state.add_edge(me, "OUT_a", tasklet_a1, "__in1", dace.Memlet("a[__i]"))
    state.add_edge(b, None, me, "IN_b", dace.Memlet("b[0:10]"))
    state.add_edge(me, "OUT_b", tasklet_a1, "__in2", dace.Memlet("b[__i]"))
    state.add_edge(tasklet_a1, "__out", a1, None, dace.Memlet("a1[0]"))

    # Computing of `b1`:
    tasklet_b1 = state.add_tasklet(
        "tasklet_b1", inputs={"__in"}, outputs={"__out"}, code="__out = math.exp(__in)"
    )

    state.add_edge(me, "OUT_b", tasklet_b1, "__in", dace.Memlet("b[__i]"))
    state.add_edge(tasklet_b1, "__out", b1, None, dace.Memlet("b1[0]"))

    # Now the condition.
    tasklet_cond = state.add_tasklet(
        "tasklet_cond",
        inputs={"__in"},
        outputs={"__out"},
        code="__out = __in <= 0.5",
    )
    state.add_edge(c, None, me, "IN_c", dace.Memlet("c[0:10]"))
    state.add_edge(me, "OUT_c", tasklet_cond, "__in", dace.Memlet("c[__i]"))
    state.add_edge(tasklet_cond, "__out", c1, None, dace.Memlet("c1[0]"))

    # Make the if selection.
    if_block = _make_if_block(state=state, outer_sdfg=sdfg)
    state.add_edge(a1, None, if_block, "__arg1", dace.Memlet("a1[0]"))
    state.add_edge(b1, None, if_block, "__arg2", dace.Memlet("b1[0]"))
    state.add_edge(c1, None, if_block, "__cond", dace.Memlet("c1[0]"))

    # Now handle the output.
    state.add_edge(if_block, "__output", mx, "IN_d", dace.Memlet("d[__i]"))
    state.add_edge(mx, "OUT_d", d, None, dace.Memlet("d[0:10]"))

    # Now add the connectors to the Map*
    for iname in input_names:
        if iname == "d":
            continue
        me.add_in_connector(f"IN_{iname}")
        me.add_out_connector(f"OUT_{iname}")
    mx.add_in_connector("IN_d")
    mx.add_out_connector("OUT_d")

    _perform_test(sdfg, explected_applies=1)

    # Examine the structure of the SDFG.
    top_ac: list[dace_nodes.AccessNode] = util.count_nodes(state, dace_nodes.AccessNode, True)
    assert {ac.data for ac in top_ac} == set(input_names).union(["c1"])
    assert len(sdfg.arrays) == len(top_ac)

    assert set(if_block.in_connectors.keys()) == set(input_names).union(["__cond"]).difference(
        ["d", "c"]
    )
    assert set(if_block.out_connectors.keys()) == {"__output"}

    inner_ac: list[dace_nodes.AccessNode] = util.count_nodes(
        if_block.sdfg, dace_nodes.AccessNode, True
    )
    expected_data: set[str] = (
        set(temporary_names).union(input_names).union(["__arg1", "__arg2", "__output", "__cond"])
    ).difference(["c1", "d", "c"])

    assert {ac.data for ac in inner_ac} == expected_data.difference(["__cond"])
    # `__output` & `b` in both branches, but there is no AccessNode for `__cond`.
    assert len(expected_data) + 2 - 1 == len(inner_ac)
    assert if_block.sdfg.arrays.keys() == expected_data


def test_if_mover_no_ops():
    """
    Essentially tests the following situation:
    ```python
    d = a if c else b
    ```
    I.e. there is no gain from moving something inside the body.
    """
    sdfg = dace.SDFG(util.unique_name("if_mover_no_ops"))
    state = sdfg.add_state(is_start_block=True)

    # Inputs
    input_names = ["a", "b", "c", "d"]
    for name in input_names:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.add_scalar("c1", dtype=dace.bool_, transient=True)
    c1 = state.add_access("c1")

    tlet_to_delete, me, mx = state.add_mapped_tasklet(
        "comp",
        map_ranges={"__i": "0:10"},
        inputs={
            "__arg1": dace.Memlet("a[__i]"),
            "__arg2": dace.Memlet("b[__i]"),
            "__cond": dace.Memlet("c[__i]"),
        },
        code="",  # Tasklet will be replaced with the `if`.
        outputs={"__output": dace.Memlet("d[__i]")},
        external_edges=True,
    )
    state.remove_node(tlet_to_delete)

    if_block = _make_if_block(state, sdfg)

    state.add_edge(me, "OUT_a", if_block, "__arg1", dace.Memlet("a[__i]"))
    state.add_edge(me, "OUT_b", if_block, "__arg2", dace.Memlet("b[__i]"))

    # The condition needs its own tasklet.
    tasklet_c1 = state.add_tasklet(
        "tasklet_c1", inputs={"__in"}, outputs={"__out"}, code="__out = __in < 0.0"
    )
    state.add_edge(me, "OUT_c", tasklet_c1, "__in", dace.Memlet("c[__i]"))
    state.add_edge(tasklet_c1, "__out", c1, None, dace.Memlet("c1[0]"))
    state.add_edge(c1, None, if_block, "__cond", dace.Memlet("c1[0]"))

    # The output.
    state.add_edge(if_block, "__output", mx, "IN_d", dace.Memlet("d[__i]"))
    sdfg.validate()

    # This might change if we will move the read fully inside the branches.
    _perform_test(sdfg, explected_applies=0)


def test_if_mover_chain():
    """
    Essentially tests the following situation:
    ```python
    a = foo(...)
    b = bar(...)
    c = baz(...)
    d = a if c else b
    cc = baz2(d, ...)
    aa = foo2(...)
    bb = bar2(...)
    e = aa if cc else bb
    ```
    """
    assert False


def test_if_mover_one_branch_is_nothing():
    """
    Essentially tests the following situation:
    ```python
    a = foo(...)
    d = a if c else b
    ```
    I.e. in one case something can be moved in but there is nothing to move for the
    other branch.
    """
    assert False
