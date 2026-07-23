# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

dace = pytest.importorskip("dace")


def _make_sdfg_with_top_level_transient(storage):
    sdfg = dace.SDFG("external_mode_scalar_test")
    state = sdfg.add_state("state", is_start_block=True)
    sdfg.add_array("a", [10], dace.float64, transient=False, storage=storage)
    sdfg.add_array("tmp_arr", [10], dace.float64, transient=True, storage=storage)
    sdfg.add_scalar("tmp_scalar", dace.float64, transient=True)
    sdfg.add_array("b", [10], dace.float64, transient=False, storage=storage)

    a = state.add_access("a")
    tmp_arr = state.add_access("tmp_arr")
    tmp_scalar = state.add_access("tmp_scalar")
    b = state.add_access("b")
    state.add_nedge(a, tmp_arr, dace.Memlet("a[0:10]"))
    state.add_nedge(tmp_arr, b, dace.Memlet("tmp_arr[0:10]"))
    init_scalar = state.add_tasklet(
        "init_scalar",
        inputs={},
        outputs={"out"},
        code="out = 1.0",
    )
    state.add_edge(init_scalar, "out", tmp_scalar, None, dace.Memlet("tmp_scalar"))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize(
    "lifetime",
    [
        dace.AllocationLifetime.Persistent,
        dace.AllocationLifetime.External,
    ],
)
@pytest.mark.parametrize(
    "storage",
    [
        dace.StorageType.Default,
        dace.StorageType.GPU_Global,
    ],
)
def test_configure_transient_lifetime(lifetime, storage):
    sdfg = _make_sdfg_with_top_level_transient(storage)

    result = gtx_transformations.gt_configure_transient_lifetime(sdfg, lifetime)
    candidates = next(iter(result.values()))
    assert candidates == {"tmp_arr", "tmp_scalar"}

    assert sdfg.arrays["tmp_arr"].lifetime == lifetime
    assert sdfg.arrays["tmp_arr"].storage == (
        dace.StorageType.CPU_Heap if storage == dace.StorageType.Default else storage
    )
    assert sdfg.arrays["tmp_scalar"].lifetime == lifetime
    assert sdfg.arrays["tmp_scalar"].storage == dace.StorageType.Default
