# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import pytest


dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace_fieldview.transformations import (
    gpu_utils as gtx_dace_fieldview_gpu_utils,
)

from . import pytestmark
from . import util


def _get_trivial_gpu_promotable(
    tasklet_code: str,
) -> tuple[dace.SDFG, dace_nodes.MapEntry, dace_nodes.MapEntry]:
    """Returns an SDFG that is suitable to test the `TrivialGPUMapPromoter` promoter.

    The first map is a trivial map (`Map[__trival_gpu_it=0]`) containing a Tasklet,
    that does not have an output, but writes a scalar value into `tmp` (output
    connector `__out`), the body of this Tasklet can be controlled through the
    `tasklet_code` argument.
    The second map (`Map[__i0=0:N]`) contains a Tasklet that computes the sum of its
    two inputs, the first input is the scalar value inside `tmp` and the second one
    is `a[__i0]`, the result is stored in `b[__i0]`.

    Returns:
        A tuple, the first element is the SDFG, the second element is the map entry
        of the trivial map and the last element is the map entry of the second map.

    Args:
        tasklet_code: The body of the Tasklet inside the trivial map.
    """
    sdfg = dace.SDFG(util.unique_name("gpu_promotable_sdfg"))
    state = sdfg.add_state("state", is_start_block=True)
    sdfg.add_symbol("N", dace.int32)

    storage_array = dace.dtypes.StorageType.GPU_Global
    storage_scalar = dace.dtypes.StorageType.Register
    schedule = dace.dtypes.ScheduleType.GPU_Device

    sdfg.add_scalar("tmp", dace.float64, transient=True)
    sdfg.add_array("a", shape=("N",), dtype=dace.float64, transient=False, storage=storage_array)
    sdfg.add_array("b", shape=("N",), dtype=dace.float64, transient=False, storage=storage_array)
    a, b, tmp = (state.add_access(name) for name in ["a", "b", "tmp"])

    _, trivial_map_entry, _ = state.add_mapped_tasklet(
        "trivail_top_tasklet",
        map_ranges={"__trivial_gpu_it": "0"},
        inputs={},
        code=tasklet_code,
        outputs={"__out": dace.Memlet("tmp[0]")},
        output_nodes={"tmp": tmp},
        external_edges=True,
        schedule=schedule,
    )
    _, second_map_entry, _ = state.add_mapped_tasklet(
        "non_trivial_tasklet",
        map_ranges={"__i0": "0:N"},
        inputs={
            "__in0": dace.Memlet("a[__i0]"),
            "__in1": dace.Memlet("tmp[0]"),
        },
        code="__out = __in0 + __in1",
        outputs={"__out": dace.Memlet("b[__i0]")},
        input_nodes={"a": a, "tmp": tmp},
        output_nodes={"b": b},
        external_edges=True,
        schedule=schedule,
    )
    return sdfg, trivial_map_entry, second_map_entry


def test_trivial_gpu_map_promoter():
    """Tests if the GPU map promoter works.

    By using a body such as `__out = 3.0`, the transformation will apply.
    """
    sdfg, trivial_map_entry, second_map_entry = _get_trivial_gpu_promotable("__out = 3.0")
    org_second_map_params = list(second_map_entry.map.params)
    org_second_map_ranges = copy.deepcopy(second_map_entry.map.range)

    nb_runs = sdfg.apply_transformations_once_everywhere(
        gtx_dace_fieldview_gpu_utils.TrivialGPUMapPromoter(do_not_fuse=True),
        validate=True,
        validate_all=True,
    )
    assert (
        nb_runs == 1
    ), f"Expected that 'TrivialGPUMapPromoter' applies once but it applied {nb_runs}."
    trivial_map_params = trivial_map_entry.map.params
    trivial_map_ranges = trivial_map_ranges.map.range
    second_map_params = second_map_entry.map.params
    second_map_ranges = second_map_entry.map.range

    assert (
        second_map_params == org_second_map_params
    ), "The transformation modified the parameter of the second map."
    assert all(
        org_rng == rng for org_rng, rng in zip(org_second_map_ranges, second_map_ranges)
    ), "The transformation modified the range of the second map."
    assert all(
        t_rng == s_rng for t_rng, s_rng in zip(trivial_map_ranges, second_map_ranges, strict=True)
    ), "Expected that the ranges are the same; trivial '{trivial_map_ranges}'; second '{second_map_ranges}'."
    assert (
        trivial_map_params == second_map_params
    ), f"Expected the trivial map to have parameters '{second_map_params}', but it had '{trivial_map_params}'."
    assert sdfg.is_valid()


def test_trivial_gpu_map_promoter():
    """Test if the GPU promoter does not fuse a special trivial map.

    By using a body such as `__out = __trivial_gpu_it` inside the
    Tasklet's body, the map parameter is now used, and thus can not be fused.
    """
    sdfg, trivial_map_entry, second_map_entry = _get_trivial_gpu_promotable(
        "__out = __trivial_gpu_it"
    )
    org_trivial_map_params = list(trivial_map_entry.map.params)
    org_second_map_params = list(second_map_entry.map.params)

    nb_runs = sdfg.apply_transformations_once_everywhere(
        gtx_dace_fieldview_gpu_utils.TrivialGPUMapPromoter(do_not_fuse=True),
        validate=True,
        validate_all=True,
    )
    assert (
        nb_runs == 0
    ), f"Expected that 'TrivialGPUMapPromoter' does not apply but it applied {nb_runs}."
    trivial_map_params = trivial_map_entry.map.params
    second_map_params = second_map_entry.map.params
    assert (
        trivial_map_params == org_trivial_map_params
    ), f"Expected the trivial map to have parameters '{org_trivial_map_params}', but it had '{trivial_map_params}'."
    assert (
        second_map_params == org_second_map_params
    ), f"Expected the trivial map to have parameters '{org_trivial_map_params}', but it had '{trivial_map_params}'."
    assert sdfg.is_valid()
