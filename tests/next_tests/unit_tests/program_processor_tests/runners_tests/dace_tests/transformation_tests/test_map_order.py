# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
import numpy as np

dace = pytest.importorskip("dace")

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


def _perform_reorder_test(
    sdfg: dace.SDFG,
    unit_strides_dim: list[str],
    expected_order: list[str],
) -> None:
    """Performs the reorder transformation and test it.

    If `expected_order` is the empty list, then the transformation should not apply.
    """
    map_entries: list[dace.nodes.MapEntry] = util.count_nodes(sdfg, dace.nodes.MapEntry, True)
    assert len(map_entries) == 1
    map_entry: dace.nodes.MapEntry = map_entries[0]
    old_map_params = map_entry.map.params.copy()

    apply_count = sdfg.apply_transformations_repeated(
        gtx_transformations.MapIterationOrder(
            unit_strides_dims=unit_strides_dim,
        ),
        validate=True,
        validate_all=True,
    )
    new_map_params = map_entry.map.params.copy()

    if len(expected_order) == 0:
        assert (
            apply_count == 0
        ), f"Expected that the transformation was not applied. New map order: {map_entry.map.params}"
        return
    else:
        assert (
            apply_count > 0
        ), f"Expected that the transformation was applied. Old map order: {map_entry.map.params}; Expected order: {expected_order}"
        assert len(expected_order) == len(new_map_params)

    assert (
        expected_order == new_map_params
    ), f"Expected map order {expected_order} but got {new_map_params} instead."


def _make_test_sdfg(map_params: list[str]) -> dace.SDFG:
    """Generate an SDFG for the test."""
    sdfg = dace.SDFG(util.unique_name("gpu_promotable_sdfg"))
    state: dace.SDFGState = sdfg.add_state("state", is_start_block=True)
    dim = len(map_params)
    for aname in ["a", "b"]:
        sdfg.add_array(aname, shape=((4,) * dim), dtype=dace.float64, transient=False)

    state.add_mapped_tasklet(
        "mapped_tasklet",
        map_ranges=[(map_param, "0:4") for map_param in map_params],
        inputs={"__in": dace.Memlet("a[" + ",".join(map_params) + "]")},
        code="__out = __in + 1",
        outputs={"__out": dace.Memlet("b[" + ",".join(map_params) + "]")},
        external_edges=True,
    )
    sdfg.validate()

    return sdfg


def test_map_order_unit_strides_kind_1():
    sdfg = _make_test_sdfg(["EDim_horizontal", "KDim_vertical", "VDim_horizontal"])
    map_entrie: dace.nodes.MapEntry = util.count_nodes(sdfg, dace.nodes.MapEntry, True)[0]
    original_map_order = map_entrie.map.params.copy()
    assert not original_map_order[0].endswith("_vertical")

    gtx_transformations.gt_set_iteration_order(
        sdfg=sdfg,
        unit_strides_kind=gtx_common.DimensionKind.HORIZONTAL,
    )
    new_map_order = map_entrie.map.params.copy()

    # The order of the horizontal is unspecific, but vertical must be at
    #  position 0, because position 1 and 2 are occupied by the other two.
    assert new_map_order[0].endswith("_vertical")
    assert all(map_param.endswith("_horizontal") for map_param in new_map_order[1:])
    assert set(original_map_order) == set(new_map_order)
    assert len(new_map_order) == len(original_map_order)


def test_map_order_unit_strides_kind_2():
    sdfg = _make_test_sdfg(["EDim_horizontal", "VDim_horizontal"])
    map_entrie: dace.nodes.MapEntry = util.count_nodes(sdfg, dace.nodes.MapEntry, True)[0]
    original_map_order = map_entrie.map.params.copy()
    assert not original_map_order[0].endswith("_vertical")

    nb_applies = gtx_transformations.gt_set_iteration_order(
        sdfg=sdfg,
        unit_strides_kind=gtx_common.DimensionKind.VERTICAL,
    )
    new_map_order = map_entrie.map.params.copy()

    # Because there is no vertical, the transformation should not apply.
    assert nb_applies == 0
    assert original_map_order == new_map_order


def test_map_order_unit_strides_dim_1():
    sdfg = _make_test_sdfg(["EDim", "KDim", "VDim"])
    _perform_reorder_test(sdfg, ["EDim", "VDim"], ["KDim", "VDim", "EDim"])


def test_map_order_unit_strides_dim_2():
    sdfg = _make_test_sdfg(["VDim", "KDim"])
    _perform_reorder_test(sdfg, ["EDim", "VDim"], ["KDim", "VDim"])


def test_map_order_unit_strides_dim_3():
    sdfg = _make_test_sdfg(["EDim", "KDim"])
    _perform_reorder_test(sdfg, ["EDim", "VDim"], ["KDim", "EDim"])


def test_map_order_unit_strides_dim_4():
    sdfg = _make_test_sdfg(["CDim", "KDim"])
    _perform_reorder_test(sdfg, ["EDim", "VDim"], [])
