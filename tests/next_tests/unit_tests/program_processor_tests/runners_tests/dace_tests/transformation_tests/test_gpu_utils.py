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

from gt4py.next.program_processors.runners.dace.transformations import (
    gpu_utils as gtx_dace_fieldview_gpu_utils,
)

from . import pytestmark
from . import util


def _get_trivial_gpu_promotable(
    tasklet_code: str,
    trivial_map_range: str = "0",
) -> tuple[dace.SDFG, dace_nodes.MapEntry, dace_nodes.MapEntry]:
    """Returns an SDFG that is suitable to test the `TrivialGPUMapElimination` promoter.

    The first map is a trivial map (`Map[__trival_gpu_it=0]`) containing a Tasklet,
    that does not have an input, but writes a scalar value into `tmp` (output
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
        trivial_map_range: Range of the trivial map, defaults to `"0"`.
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
        map_ranges={"__trivial_gpu_it": trivial_map_range},
        inputs={},
        code=tasklet_code,
        outputs={"__out": dace.Memlet("tmp[0]")},
        output_nodes={tmp},
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
        input_nodes={a, tmp},
        output_nodes={b},
        external_edges=True,
        schedule=schedule,
    )
    return sdfg, trivial_map_entry, second_map_entry


def test_trivial_gpu_map_promoter_1():
    """Tests if the GPU map promoter works.

    By using a body such as `__out = 3.0`, the transformation will apply.
    """
    sdfg, trivial_map_entry, second_map_entry = _get_trivial_gpu_promotable("__out = 3.0")
    org_second_map_params = list(second_map_entry.map.params)
    org_second_map_ranges = copy.deepcopy(second_map_entry.map.range)

    nb_runs = sdfg.apply_transformations_once_everywhere(
        gtx_dace_fieldview_gpu_utils.TrivialGPUMapElimination(do_not_fuse=True),
        validate=True,
        validate_all=True,
    )
    assert (
        nb_runs == 1
    ), f"Expected that 'TrivialGPUMapElimination' applies once but it applied {nb_runs}."
    trivial_map_params = trivial_map_entry.map.params
    trivial_map_ranges = trivial_map_entry.map.range
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


def test_trivial_gpu_map_promoter_2():
    """Test if the GPU promoter does not fuse a special trivial map.

    By using a body such as `__out = __trivial_gpu_it` inside the
    Tasklet's body, the map parameter must now be replaced inside
    the Tasklet's body.
    """
    sdfg, trivial_map_entry, second_map_entry = _get_trivial_gpu_promotable(
        tasklet_code="__out = __trivial_gpu_it",
        trivial_map_range="2",
    )
    state: dace.SDFGStae = sdfg.nodes()[0]
    trivial_tasklet: dace_nodes.Tasklet = next(
        iter(
            out_edge.dst
            for out_edge in state.out_edges(trivial_map_entry)
            if isinstance(out_edge.dst, dace_nodes.Tasklet)
        )
    )

    nb_runs = sdfg.apply_transformations_once_everywhere(
        gtx_dace_fieldview_gpu_utils.TrivialGPUMapElimination(do_not_fuse=True),
        validate=True,
        validate_all=True,
    )
    assert nb_runs == 1

    expected_trivial_code = "__out = 2"
    assert trivial_tasklet.code == expected_trivial_code


def test_set_gpu_properties():
    """Tests the `gtx_dace_fieldview_gpu_utils.gt_set_gpu_blocksize()`."""
    sdfg = dace.SDFG("gpu_properties_test")
    state = sdfg.add_state(is_start_block=True)

    map_entries: dict[int, dace_nodes.MapEntry] = {}
    for dim in [1, 2, 3]:
        shape = (10,) * dim
        sdfg.add_array(
            f"A_{dim}", shape=shape, dtype=dace.float64, storage=dace.StorageType.GPU_Global
        )
        sdfg.add_array(
            f"B_{dim}", shape=shape, dtype=dace.float64, storage=dace.StorageType.GPU_Global
        )
        _, me, _ = state.add_mapped_tasklet(
            f"map_{dim}",
            map_ranges={f"__i{i}": f"0:{s}" for i, s in enumerate(shape)},
            inputs={"__in": dace.Memlet(f"A_{dim}[{','.join(f'__i{i}' for i in range(dim))}]")},
            code="__out = math.cos(__in)",
            outputs={"__out": dace.Memlet(f"B_{dim}[{','.join(f'__i{i}' for i in range(dim))}]")},
            external_edges=True,
        )
        map_entries[dim] = me

    sdfg.apply_gpu_transformations()
    sdfg.validate()

    gtx_dace_fieldview_gpu_utils.gt_set_gpu_blocksize(
        sdfg=sdfg,
        block_size=(10, "11", 12),
        launch_factor_2d=2,
        block_size_2d=(2, 2, 2),
        launch_bounds_3d=200,
    )

    map1, map2, map3 = (map_entries[d].map for d in [1, 2, 3])

    assert len(map1.params) == 1
    assert map1.gpu_block_size == [10, 1, 1]
    assert map1.gpu_launch_bounds == "0"

    assert len(map2.params) == 2
    assert map2.gpu_block_size == [2, 2, 1]
    assert map2.gpu_launch_bounds == "8"

    assert len(map3.params) == 3
    assert map3.gpu_block_size == [10, 11, 12]
    assert map3.gpu_launch_bounds == "200"
