# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests the redundant array removal transformation."""

import pytest
import numpy as np

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)

from . import util


dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes


def test_gt4py_redundant_array_elimination():
    sdfg: dace.SDFG = dace.SDFG(util.unique_name("test_gt4py_redundant_array_elimination"))
    sdfg.add_array("a", (10, 10), dace.float64, transient=False)
    sdfg.add_array("b", (20, 20), dace.float64, transient=False)
    sdfg.add_array("c", (2, 2), dace.float64, transient=False)
    sdfg.add_array("d", (2, 2), dace.float64, transient=False)
    sdfg.add_array("tmp1", (20, 20), dace.float64, transient=True)
    sdfg.add_array("tmp2", (20, 20), dace.float64, transient=True)

    pre_state: dace.SDFGState = sdfg.add_state(is_start_block=True)
    pre_state.add_mapped_tasklet(
        "set_tmp2_to_zero",
        map_ranges={"__i0": "0:20", "__i1": "0:20"},
        inputs={},
        code="__out = 0.0",
        outputs={"__out": dace.Memlet("tmp2[__i0, __i1]")},
        external_edges=True,
    )

    state: dace.SDFGState = sdfg.add_state_after(pre_state)
    a, tmp1, tmp2 = (state.add_access(name) for name in ["a", "tmp1", "tmp2"])
    state.add_mapped_tasklet(
        "compute",
        map_ranges={"__i0": "0:10", "__i1": "0:10"},
        inputs={"__in": dace.Memlet("a[__i0, __i1]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("tmp1[5 + __i0, 6 + __i1]")},
        input_nodes={a},
        output_nodes={tmp1},
        external_edges=True,
    )
    state.add_nedge(
        tmp1,
        tmp2,
        dace.Memlet("tmp1[0:20, 0:20]"),
    )
    state.add_nedge(
        state.add_access("c"),
        tmp1,
        dace.Memlet("c[0:2, 0:2] -> 18:20, 17:19"),
    )
    state.add_nedge(
        state.add_access("d"),
        tmp1,
        dace.Memlet("tmp1[0:2, 1:3]"),
    )

    state2: dace.SDFGState = sdfg.add_state_after(state)
    state2.add_nedge(
        state2.add_access("tmp2"),
        state2.add_access("b"),
        dace.Memlet("tmp2[0:20, 0:20]"),
    )
    sdfg.validate()

    count = sdfg.apply_transformations(
        gtx_transformations.GT4PyRednundantArrayElimination(),
        validate_all=True,
    )
    assert count == 1

    a = np.random.rand(10, 10).astype(np.float64)
    c = np.random.rand(2, 2).astype(np.float64)
    d = np.random.rand(2, 2).astype(np.float64)
    b = np.zeros((20, 20), dtype=np.float64)

    # Compose the reference solution
    b_ref = np.zeros((20, 20), dtype=np.float64)
    b_ref[5:15, 6:16] = a + 1.0
    b_ref[18:20, 17:19] = c
    b_ref[0:2, 1:3] = d

    sdfg(a=a, b=b, c=c, d=d)
    assert np.allclose(b, b_ref)


def test_gt4py_redundant_array_elimination_unequal_shape():
    """Tests the array elimination when the arrays have different shapes."""
    sdfg: dace.SDFG = dace.SDFG(util.unique_name("test_gt4py_redundant_array_elimination"))
    state: dace.SDFGState = sdfg.add_state(is_start_block=True)

    sdfg.add_array("origin", (30,), dace.float64, transient=False)
    sdfg.add_array("t", (20,), dace.float64, transient=True)
    sdfg.add_array("v", (10,), dace.float64, transient=False)

    origin, t, v = (state.add_access(name) for name in ["origin", "t", "v"])

    state.add_nedge(
        origin,
        t,
        dace.Memlet(data="origin", subset="5:20", other_subset="2:17"),
    )
    state.add_nedge(
        t,
        v,
        dace.Memlet(data="t", subset="3:11", other_subset="1:9"),
    )
    sdfg.validate()

    origin = np.array(np.random.rand(30), dtype=np.float64, copy=True)
    v_ref = np.array(np.random.rand(10), dtype=np.float64, copy=True)
    v_ref[1:9] = 0.0
    v_res = v_ref.copy()

    csdfg_org = sdfg.compile()
    csdfg_org(origin=origin, v=v_ref)

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyRednundantArrayElimination(),
        validate=True,
        validate_all=True,
    )
    assert count == 1, f"Transformation was applied {count}, but expected once."

    csdfg_opt = sdfg.compile()
    csdfg_opt(origin=origin, v=v_res)
    assert np.allclose(v_ref, v_res), f"Expected {v_ref}, but got {v_res}."
