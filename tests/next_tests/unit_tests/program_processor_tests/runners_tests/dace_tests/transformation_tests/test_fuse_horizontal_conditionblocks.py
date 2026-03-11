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
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util
from .test_move_dataflow_into_if_body import _make_if_block

import dace


def _make_if_block_with_tasklet(
    state: dace.SDFGState,
    b1_name: str = "__arg1",
    b2_name: str = "__arg2",
    cond_name: str = "__cond",
    output_name: str = "__output",
    b1_type: dace.typeclass = dace.float64,
    b2_type: dace.typeclass = dace.float64,
    output_type: dace.typeclass = dace.float64,
) -> dace_nodes.NestedSDFG:
    inner_sdfg = dace.SDFG(gtx_transformations.utils.unique_name("if_stmt_"))

    types = {b1_name: b1_type, b2_name: b2_type, cond_name: dace.bool_, output_name: output_type}
    for name in {b1_name, b2_name, cond_name, output_name}:
        inner_sdfg.add_scalar(
            name,
            dtype=types[name],
            transient=False,
        )

    if_region = dace.sdfg.state.ConditionalBlock(gtx_transformations.utils.unique_name("if"))
    inner_sdfg.add_node(if_region, is_start_block=True)

    then_body = dace.sdfg.state.ControlFlowRegion("then_body", sdfg=inner_sdfg)
    tstate = then_body.add_state("true_branch_0_1_2_3_4", is_start_block=True)
    inner_sdfg.add_symbol("multiplier", dace.float64)

    tasklet = tstate.add_tasklet(
        "true_tasklet",
        inputs={"__tasklet_in"},
        outputs={"__tasklet_out"},
        code="__tasklet_out = __tasklet_in * multiplier",
    )
    tstate.add_edge(
        tstate.add_access(b1_name),
        None,
        tasklet,
        "__tasklet_in",
        dace.Memlet(f"{b1_name}[0]"),
    )
    tstate.add_edge(
        tasklet,
        "__tasklet_out",
        tstate.add_access(output_name),
        None,
        dace.Memlet(f"{output_name}[0]"),
    )

    else_body = dace.sdfg.state.ControlFlowRegion("else_body", sdfg=inner_sdfg)
    fstate = else_body.add_state("false_branch_0_1_2_3_4", is_start_block=True)
    fstate.add_nedge(
        fstate.add_access(b2_name),
        fstate.add_access(output_name),
        dace.Memlet(f"{b2_name}[0] -> [0]"),
    )

    if_region.add_branch(dace.sdfg.state.CodeBlock(cond_name), then_body)
    if_region.add_branch(dace.sdfg.state.CodeBlock(f"not {cond_name}"), else_body)

    nested_sdfg = state.add_nested_sdfg(
        sdfg=inner_sdfg,
        inputs={b1_name, b2_name, cond_name},
        outputs={output_name},
    )
    nested_sdfg.symbol_mapping["multiplier"] = 2.0
    return nested_sdfg


def _make_map_with_conditional_blocks() -> dace.SDFG:
    sdfg = dace.SDFG(gtx_transformations.utils.unique_name("map_with_conditional_blocks"))
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
        transient=False,
    )
    sdfg.add_array(
        "c",
        shape=(10,),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "d",
        shape=(10,),
        dtype=dace.float64,
        transient=False,
    )
    a, b, c, d = (state.add_access(name) for name in "abcd")

    for tmp_name in ["tmp_a", "tmp_b", "tmp_c", "tmp_d"]:
        sdfg.add_scalar(tmp_name, dtype=dace.float64, transient=True)
    tmp_a, tmp_b, tmp_c, tmp_d = (state.add_access(f"tmp_{name}") for name in "abcd")

    sdfg.add_scalar("cond_var", dtype=dace.bool_, transient=True)
    cond_var = state.add_access("cond_var")

    me, mx = state.add_map("map_with_ifs", ndrange={"__i": "0:10"})
    me.add_in_connector("IN_a")
    me.add_out_connector("OUT_a")
    me.add_in_connector("IN_b")
    me.add_out_connector("OUT_b")
    mx.add_in_connector("IN_c")
    mx.add_out_connector("OUT_c")
    mx.add_in_connector("IN_d")
    mx.add_out_connector("OUT_d")
    state.add_edge(a, None, me, "IN_a", dace.Memlet("a[__i]"))
    state.add_edge(b, None, me, "IN_b", dace.Memlet("b[__i]"))
    state.add_edge(me, "OUT_a", tmp_a, None, dace.Memlet("a[__i]"))
    state.add_edge(me, "OUT_b", tmp_b, None, dace.Memlet("b[__i]"))

    tasklet_cond = state.add_tasklet(
        "tasklet_cond",
        inputs={"__in"},
        outputs={"__out"},
        code="__out = __in <= 0.5",
    )
    state.add_edge(tmp_a, None, tasklet_cond, "__in", dace.Memlet("tmp_a[0]"))
    state.add_edge(tasklet_cond, "__out", cond_var, None, dace.Memlet("cond_var"))

    if_block_0 = _make_if_block(state=state, outer_sdfg=sdfg)
    state.add_edge(cond_var, None, if_block_0, "__cond", dace.Memlet("cond_var"))
    state.add_edge(tmp_a, None, if_block_0, "__arg1", dace.Memlet("tmp_a[0]"))
    state.add_edge(tmp_b, None, if_block_0, "__arg2", dace.Memlet("tmp_b[0]"))
    state.add_edge(if_block_0, "__output", tmp_c, None, dace.Memlet("tmp_c[0]"))
    state.add_edge(tmp_c, None, mx, "IN_c", dace.Memlet("c[__i]"))

    if_block_1 = _make_if_block_with_tasklet(state=state)
    state.add_edge(cond_var, None, if_block_1, "__cond", dace.Memlet("cond_var"))
    state.add_edge(tmp_a, None, if_block_1, "__arg1", dace.Memlet("tmp_a[0]"))
    state.add_edge(tmp_b, None, if_block_1, "__arg2", dace.Memlet("tmp_b[0]"))
    state.add_edge(if_block_1, "__output", tmp_d, None, dace.Memlet("tmp_d[0]"))
    state.add_edge(tmp_d, None, mx, "IN_d", dace.Memlet("d[__i]"))

    state.add_edge(mx, "OUT_c", c, None, dace.Memlet("c[__i]"))
    state.add_edge(mx, "OUT_d", d, None, dace.Memlet("d[__i]"))

    sdfg.validate()
    return sdfg


def test_fuse_horizontal_condition_blocks():
    sdfg = _make_map_with_conditional_blocks()

    conditional_blocks = [
        n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.sdfg.state.ConditionalBlock)
    ]
    assert len(conditional_blocks) == 2

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    sdfg.apply_transformations_repeated(
        gtx_transformations.FuseHorizontalConditionBlocks(),
        validate=True,
        validate_all=True,
    )

    new_conditional_blocks = [
        n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.sdfg.state.ConditionalBlock)
    ]
    assert len(new_conditional_blocks) == 1
    conditional_block = new_conditional_blocks[0]
    assert (
        len(conditional_block.sdfg.symbols) == 1 and "multiplier" in conditional_block.sdfg.symbols
    )

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)
