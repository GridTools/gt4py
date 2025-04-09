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
    sdfg = dace.SDFG(util.unique_name("if_mover_independent_branches"))
    state = sdfg.add_state(is_start_block=True)

    # Inputs
    for name in ["a", "b", "c", "d"]:
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
    for iname in ["a", "b", "c"]:
        me.add_in_connector(f"IN_{iname}")
        me.add_out_connector(f"OUT_{iname}")
    mx.add_in_connector("IN_d")
    mx.add_out_connector("OUT_d")

    sdfg.validate()

    _perform_test(sdfg, explected_applies=1)

    sdfg.view()
