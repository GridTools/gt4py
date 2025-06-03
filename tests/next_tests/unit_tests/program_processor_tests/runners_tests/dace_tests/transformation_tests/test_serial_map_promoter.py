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

import copy


from . import util


def _make_serial_map_promotion_sdfg() -> (
    tuple[dace.SDFG, dace.SDFGState, dace_nodes.MapEntry, dace_nodes.MapEntry]
):
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

    return sdfg, state, map_entry_1d, map_entry_2d


def test_serial_map_promotion_only_promote():
    sdfg, state, map_entry_1d, map_entry_2d = _make_serial_map_promotion_sdfg()

    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2
    assert len(map_entry_1d.map.params) == 1
    assert len(map_entry_1d.map.range) == 1
    assert len(map_entry_2d.map.params) == 2
    assert len(map_entry_2d.map.range) == 2

    # Now apply the promotion
    count = sdfg.apply_transformations(
        gtx_transformations.SerialMapPromoter(
            promote_all=True,
            # Do not fuse to inspect that the promotion worked.
            fuse_after_promotion=False,
        ),
        validate=True,
        validate_all=True,
    )

    assert count == 1
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


def test_serial_map_promotion_promote_and_merge():
    sdfg, state, map_entry_1d, map_entry_2d = _make_serial_map_promotion_sdfg()

    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2
    assert len(map_entry_1d.map.params) == 1
    assert len(map_entry_1d.map.range) == 1
    assert len(map_entry_2d.map.params) == 2
    assert len(map_entry_2d.map.range) == 2

    original_2d_params = list(map_entry_2d.map.params)
    original_2d_range = copy.deepcopy(map_entry_2d.map.range)

    # Now apply the promotion
    count = sdfg.apply_transformations(
        gtx_transformations.SerialMapPromoter(
            promote_all=True,
            fuse_after_promotion=True,
        ),
        validate=True,
        validate_all=True,
    )

    assert count == 1
    mes = util.count_nodes(sdfg, dace_nodes.MapEntry, True)

    assert len(mes) == 1
    me = mes[0]

    assert len(me.map.params) == 2
    assert me.map.params == original_2d_params
    assert len(me.map.range) == 2
    assert me.map.range == original_2d_range
