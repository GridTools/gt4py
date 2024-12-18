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

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)

from . import pytestmark
from . import util


def test_serial_map_promotion():
    """Tests the serial Map promotion transformation."""
    N = 10
    shape_1d = (N,)
    shape_2d = (N, N)
    sdfg = dace.SDFG(util.unique_name("serial_promotable_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    # 1D Arrays
    for name in ["a", "tmp"]:
        sdfg.add_array(
            name=name,
            shape=shape_1d,
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["tmp"].transient = True

    # 2D Arrays
    for name in ["b", "c"]:
        sdfg.add_array(
            name=name,
            shape=shape_2d,
            dtype=dace.float64,
            transient=False,
        )
    tmp = state.add_access("tmp")

    _, map_entry_1d, _ = state.add_mapped_tasklet(
        name="one_d_map",
        map_ranges=[("__i0", f"0:{N}")],
        inputs={"__in0": dace.Memlet("a[__i0]")},
        code="__out = __in0 + 1.0",
        outputs={"__out": dace.Memlet("tmp[__i0]")},
        output_nodes={"tmp": tmp},
        external_edges=True,
    )

    _, map_entry_2d, _ = state.add_mapped_tasklet(
        name="two_d_map",
        map_ranges=[("__i0", f"0:{N}"), ("__i1", f"0:{N}")],
        input_nodes={"tmp": tmp},
        inputs={"__in0": dace.Memlet("tmp[__i0]"), "__in1": dace.Memlet("b[__i0, __i1]")},
        code="__out = __in0 + __in1",
        outputs={"__out": dace.Memlet("c[__i0, __i1]")},
        external_edges=True,
    )

    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2
    assert len(map_entry_1d.map.params) == 1
    assert len(map_entry_1d.map.range) == 1
    assert len(map_entry_2d.map.params) == 2
    assert len(map_entry_2d.map.range) == 2

    # Now apply the promotion
    sdfg.apply_transformations(
        gtx_transformations.SerialMapPromoter(
            promote_all=True,
        ),
        validate=True,
        validate_all=True,
    )

    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2
    assert len(map_entry_1d.map.params) == 2
    assert len(map_entry_1d.map.range) == 2
    assert len(map_entry_2d.map.params) == 2
    assert len(map_entry_2d.map.range) == 2
    assert set(map_entry_1d.map.params) == set(map_entry_2d.map.params)
    assert all(
        rng_1d == rng_2d
        for rng_1d, rng_2d in zip(map_entry_1d.map.range.ranges, map_entry_2d.map.range.ranges)
    )
