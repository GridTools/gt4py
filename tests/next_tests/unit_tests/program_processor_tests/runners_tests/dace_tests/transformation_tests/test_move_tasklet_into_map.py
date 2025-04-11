# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest


dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes, propagation as dace_propagation
from dace.transformation import dataflow as dace_dataflow

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


def _make_movable_tasklet(
    outer_tasklet_code: str,
) -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.Tasklet, dace_nodes.AccessNode, dace_nodes.MapEntry
]:
    sdfg = dace.SDFG(util.unique_name("gpu_promotable_sdfg"))
    state = sdfg.add_state("state", is_start_block=True)

    sdfg.add_scalar("outer_scalar", dtype=dace.float64, transient=True)
    for name in "AB":
        sdfg.add_array(name, shape=(10, 10), dtype=dace.float64, transient=False)
    A, B, outer_scalar = (state.add_access(name) for name in ["A", "B", "outer_scalar"])

    outer_tasklet = state.add_tasklet(
        name="outer_tasklet",
        inputs=set(),
        outputs={"__out"},
        code=f"__out = {outer_tasklet_code}",
    )
    state.add_edge(outer_tasklet, "__out", outer_scalar, None, dace.Memlet("outer_scalar[0]"))

    _, me, _ = state.add_mapped_tasklet(
        "map",
        map_ranges={"__i0": "0:10", "__i1": "0:10"},
        inputs={
            "__in0": dace.Memlet("A[__i0, __i1]"),
            "__in1": dace.Memlet("outer_scalar[0]"),
        },
        code="__out = __in0 + __in1",
        outputs={"__out": dace.Memlet("B[__i0, __i1]")},
        external_edges=True,
        input_nodes={outer_scalar, A},
        output_nodes={B},
    )
    sdfg.validate()

    return sdfg, state, outer_tasklet, outer_scalar, me


def test_move_tasklet_inside_trivial_memlet_tree():
    sdfg, state, outer_tasklet, outer_scalar, me = _make_movable_tasklet(
        outer_tasklet_code="1.2",
    )

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyMoveTaskletIntoMap,
        validate_all=True,
    )
    assert count == 1

    A = np.array(np.random.rand(10, 10), dtype=np.float64, copy=True)
    B = np.array(np.random.rand(10, 10), dtype=np.float64, copy=True)
    ref = A + 1.2

    csdfg = sdfg.compile()
    csdfg(A=A, B=B)
    assert np.allclose(B, ref)


def test_move_tasklet_inside_non_trivial_memlet_tree():
    sdfg, state, outer_tasklet, outer_scalar, me = _make_movable_tasklet(
        outer_tasklet_code="1.2",
    )
    # By expanding the maps, we the memlet tree is no longer trivial.
    sdfg.apply_transformations_repeated(dace_dataflow.MapExpansion)
    assert util.count_nodes(state, dace_nodes.MapEntry) == 2
    me = None

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyMoveTaskletIntoMap,
        validate_all=True,
    )
    assert count == 1

    A = np.array(np.random.rand(10, 10), dtype=np.float64, copy=True)
    B = np.array(np.random.rand(10, 10), dtype=np.float64, copy=True)
    ref = A + 1.2

    csdfg = sdfg.compile()
    csdfg(A=A, B=B)
    assert np.allclose(B, ref)


def test_move_tasklet_inside_two_inner_connector():
    sdfg, state, outer_tasklet, outer_scalar, me = _make_movable_tasklet(
        outer_tasklet_code="32.2",
    )
    mapped_tasklet = next(
        iter(e.dst for e in state.out_edges(me) if isinstance(e.dst, dace_nodes.Tasklet))
    )

    state.add_edge(
        me,
        f"OUT_{outer_scalar.data}",
        mapped_tasklet,
        "__in2",
        dace.Memlet(f"{outer_scalar.data}[0]"),
    )
    mapped_tasklet.add_in_connector("__in2")
    mapped_tasklet.code.as_string = "__out = __in0 + __in1 + __in2"

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyMoveTaskletIntoMap,
        validate_all=True,
    )
    assert count == 1

    A = np.array(np.random.rand(10, 10), dtype=np.float64, copy=True)
    B = np.array(np.random.rand(10, 10), dtype=np.float64, copy=True)
    ref = A + 2 * (32.2)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B)
    assert np.allclose(B, ref)


def test_move_tasklet_inside_outer_scalar_used_outside():
    sdfg, state, outer_tasklet, outer_scalar, me = _make_movable_tasklet(
        outer_tasklet_code="22.6",
    )
    sdfg.add_array("C", shape=(1,), dtype=dace.float64, transient=False)
    state.add_edge(outer_scalar, None, state.add_access("C"), None, dace.Memlet("C[0]"))

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyMoveTaskletIntoMap,
        validate_all=True,
    )
    assert count == 1

    A = np.array(np.random.rand(10, 10), dtype=np.float64, copy=True)
    B = np.array(np.random.rand(10, 10), dtype=np.float64, copy=True)
    C = np.array(np.random.rand(1), dtype=np.float64, copy=True)
    ref_C = 22.6
    ref_B = A + ref_C

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C)
    assert np.allclose(B, ref_B)
    assert np.allclose(C, ref_C)
