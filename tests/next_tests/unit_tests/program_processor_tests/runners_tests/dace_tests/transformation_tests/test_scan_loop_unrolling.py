# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

dace = pytest.importorskip("dace")
from dace.memlet import Memlet
from dace.sdfg import nodes as dace_nodes
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.properties import CodeBlock

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


def _make_sdfg_with_LoopRegion(l: int) -> dace.SDFG:
    sdfg = dace.SDFG("LoopRegionUnroll")

    for_cfg = LoopRegion(
        label="scan_loop",
        condition_expr=CodeBlock(f"i < {l}"),
        loop_var="i",
        initialize_expr=CodeBlock("i = 0"),
        update_expr=CodeBlock("i = i + 1"),
        sdfg=sdfg,
    )

    sdfg.add_node(for_cfg, is_start_block=True)

    body = ControlFlowRegion(label="for_body", sdfg=sdfg, parent=for_cfg)
    for_cfg.add_node(body, is_start_block=True)

    s1 = body.add_state(label="s1", is_start_block=True)

    c1 = ConditionalBlock(label="cond1", sdfg=sdfg, parent=body)
    c_body = ControlFlowRegion(label="if_body", sdfg=sdfg, parent=c1)

    c1.add_branch(condition=CodeBlock("a_sym > 0.0"), branch=c_body)

    body.add_node(c1, is_start_block=False)
    body.add_edge(s1, c1, InterstateEdge(assignments={"a_sym": "A[i]"}))

    s2 = c_body.add_state(label="s2", is_start_block=True)

    b_an = s2.add_access("B")
    a_an = s2.add_access("A")
    t = s2.add_tasklet(name="assign", inputs={"_in"}, outputs={"_out"}, code="_out = _in")
    s2.add_edge(t, "_out", b_an, None, Memlet(expr="B[i]"))
    s2.add_edge(a_an, None, t, "_in", Memlet(expr="A[i]"))

    sdfg.add_array("A", shape=(5,), dtype=dace.float64)
    sdfg.add_array("B", shape=(5,), dtype=dace.float64)

    sdfg.validate()
    return sdfg


def test_scan_loop_unrolling():
    sdfg = _make_sdfg_with_LoopRegion(4)

    # Ensure that the loop region is present.
    loop_regions = [n for n in sdfg.all_control_flow_regions() if isinstance(n, LoopRegion)]
    assert len(loop_regions) == 1
    loop_region = loop_regions[0]
    assert loop_region.unroll == False, "Initial unroll property should be False."

    # Apply the transformation.
    applied = sdfg.apply_transformations_once_everywhere(
        gtx_transformations.ScanLoopUnrolling(unroll_factor=2),
        validate=True,
        validate_all=True,
    )
    assert applied

    # Verify that the loop has been unrolled.
    unrolled_loop_regions = [
        n for n in sdfg.all_control_flow_regions() if isinstance(n, LoopRegion)
    ]
    assert len(unrolled_loop_regions) == 1
    unrolled_loop_region = unrolled_loop_regions[0]
    assert unrolled_loop_region.unroll == True, (
        "Unroll property should be True after transformation."
    )
    assert unrolled_loop_region.unroll_factor == 2, "Unroll factor should be updated to 2."
