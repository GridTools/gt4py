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
    amd_block_heuristic as gtx_amd_block_heuristic,
    gpu_utils as gtx_dace_fieldview_gpu_utils,
)

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
    assert nb_runs == 1, (
        f"Expected that 'TrivialGPUMapElimination' applies once but it applied {nb_runs}."
    )
    trivial_map_params = trivial_map_entry.map.params
    trivial_map_ranges = trivial_map_entry.map.range
    second_map_params = second_map_entry.map.params
    second_map_ranges = second_map_entry.map.range

    assert second_map_params == org_second_map_params, (
        "The transformation modified the parameter of the second map."
    )
    assert all(org_rng == rng for org_rng, rng in zip(org_second_map_ranges, second_map_ranges)), (
        "The transformation modified the range of the second map."
    )
    assert all(
        t_rng == s_rng for t_rng, s_rng in zip(trivial_map_ranges, second_map_ranges, strict=True)
    ), (
        "Expected that the ranges are the same; trivial '{trivial_map_ranges}'; second '{second_map_ranges}'."
    )
    assert trivial_map_params == second_map_params, (
        f"Expected the trivial map to have parameters '{second_map_params}', but it had '{trivial_map_params}'."
    )
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


@pytest.mark.parametrize("method", [0, 1])
def test_set_gpu_properties(method: int):
    """Tests the `gtx_dace_fieldview_gpu_utils.gt_set_gpu_blocksize()`."""
    sdfg = dace.SDFG(util.unique_name("gpu_properties_test"))
    state = sdfg.add_state(is_start_block=True)

    map_entries: dict[int, dace_nodes.MapEntry] = {}
    for dim in [1, 2, 3, 4]:
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
    sdfg.validate()

    if method == 0:
        sdfg.apply_gpu_transformations()
        gtx_dace_fieldview_gpu_utils.gt_set_gpu_blocksize(
            sdfg=sdfg,
            block_size=(9, "11", 12),
            launch_factor_2d=2,
            block_size_2d=(2, "12", 2),
            launch_bounds_3d=200,
        )

    elif method == 1:
        gtx_dace_fieldview_gpu_utils.gt_gpu_transformation(
            sdfg,
            gpu_block_size=(9, "11", 12),
            # Same logic as in `gt_auto_optimizer()`.
            gpu_block_size_spec={
                "launch_factor_2d": 2,
                "block_size_2d": (2, "12", 2),
                "launch_bounds_3d": 200,
            },
        )

    else:
        raise ValueError(f"Unknown method {method}")

    map1, map2, map3, map4 = (map_entries[d].map for d in [1, 2, 3, 4])

    # It takes the normal block size and does not regulate anything.
    assert len(map1.params) == 1
    assert map1.gpu_block_size == [9, 1, 1]
    assert map1.gpu_launch_bounds == "0"

    # It takes the specialization for the 2d version, but it modifies the y dimension.
    #  The value of the launch bounds are not affected from this modification, so they
    #  are still based in `(2, 12, 1)` and then multiplied with 2.
    assert len(map2.params) == 2
    assert map2.gpu_block_size == [2, 10, 1]
    assert map2.gpu_launch_bounds == "48"

    # It takes normal block size but regulates the y and z dimension.
    assert len(map3.params) == 3
    assert map3.gpu_block_size == [9, 10, 10]
    assert map3.gpu_launch_bounds == "200"

    # It takes the normal block size and regulates the y dimension, but because
    #  there are more than three dim, it will not regulate the z block.
    #  Even though `map4` is a 4D Map, the launch bound value for 3D map is used.
    assert len(map4.params) == 4
    assert map4.gpu_block_size == [9, 10, 12]
    assert map4.gpu_launch_bounds == "200"


def test_set_gpu_properties_1D():
    """Tests the `gtx_dace_fieldview_gpu_utils.gt_set_gpu_blocksize()` with 1D maps."""
    sdfg = dace.SDFG(util.unique_name("gpu_properties_test"))
    state = sdfg.add_state(is_start_block=True)

    map_entries: dict[int, dace_nodes.MapEntry] = {}
    for dim in [1, 2, 3, 4, 5]:
        shape = (10,) + (dim - 1) * (1,)
        if dim == 5:
            shape = tuple(reversed(shape))

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
    sdfg.validate()

    sdfg.apply_gpu_transformations()
    gtx_dace_fieldview_gpu_utils.gt_set_gpu_blocksize(
        sdfg=sdfg,
        block_size_1d=(64, 1, 1),
        block_size=(32, 8, 2),
    )

    map1, map2, map3, map4, map5 = (map_entries[d].map for d in sorted(map_entries.keys()))

    # Set the `x` block size to 64, but the map size is 10, so it regulates it to 10.
    assert len(map1.params) == 1
    assert map1.gpu_block_size == [10, 1, 1]
    assert map1.gpu_launch_bounds == "0"

    # Set the `y` block size to 64, but the map size is 10, so it regulates it to 10.
    assert len(map2.params) == 2
    assert map2.gpu_block_size == [1, 10, 1]
    assert map2.gpu_launch_bounds == "0"

    # Set the `z` block size to 64, but the map size is 10, so it regulates it to 10.
    assert len(map3.params) == 3
    assert map3.gpu_block_size == [1, 1, 10]
    assert map3.gpu_launch_bounds == "0"

    # NOTE: One could expect `[1, 1, 10]` here. However, because we handle degenerated
    #   1d cases for Maps with less than 4 parameters. Furthermore in that case the
    #   block size from the last dimension is simply used without modification.
    assert len(map4.params) == 4
    assert map4.gpu_block_size == [1, 1, 2]
    assert map4.gpu_launch_bounds == "0"

    # NOTE: The reason why the z block size is 2 and not 1 is because for more than
    #   three dimensions the z dimensions absorbs all additional dimensions and is
    #   not regulated.
    assert len(map5.params) == 5
    assert map5.gpu_launch_bounds == "0"
    assert map5.gpu_block_size == [10, 1, 2]


def test_set_gpu_properties_2D_3D():
    """Tests the `gtx_dace_fieldview_gpu_utils.gt_set_gpu_blocksize()` with 2D, 3D and 4D maps."""
    sdfg = dace.SDFG(util.unique_name("gpu_properties_test"))
    state = sdfg.add_state(is_start_block=True)

    map_entries: dict[int, dace_nodes.MapEntry] = {}
    for dim in [2, 3, 4]:
        shape = (10,) * (dim - 1) + (1,)
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
    sdfg.validate()

    sdfg.apply_gpu_transformations()
    gtx_dace_fieldview_gpu_utils.gt_set_gpu_blocksize(
        sdfg=sdfg,
        block_size_1d=(128, 1, 1),
        block_size_2d=(64, 2, 1),
        block_size_3d=(2, 2, 32),
        block_size=(32, 4, 1),
    )

    map2, map3, map4 = (map_entries[d].map for d in [2, 3, 4])

    # Set the 1D block size to 128, thus the block size of `y` dimension of the map is set to 128, however the map size in this dimension is 10, so it regulates it to 10.
    assert len(map2.params) == 2
    assert map2.gpu_block_size == [1, 10, 1]
    assert map2.gpu_launch_bounds == "0"

    # Set the `x` block size to 2, but the map size in this dimension is 1, so it regulates it to 1.
    # Set the `y` block size to 2.
    # Set the `z` block size to 32, but the map size in this dimension is 1, so it regulates it to 1.
    assert len(map3.params) == 3
    assert map3.gpu_block_size == [1, 2, 10]
    assert map3.gpu_launch_bounds == "0"

    # Set the `x` block size to 2, but the map size in this dimension is 1, so it regulates it to 1.
    # Set the `y` block size to 2.
    # Set the `z` block size to 32 (the product of the two last dimensions is 100, so we pick the max(block_size_3d.z, product(map4.range.size()[2:]))).
    assert len(map4.params) == 4
    assert map4.gpu_block_size == [1, 2, 32]
    assert map4.gpu_launch_bounds == "0"


def test_set_gpu_maxnreg():
    """Tests if gpu_maxnreg property is set correctly to GPU maps."""
    sdfg = dace.SDFG(util.unique_name("gpu_maxnreg_test"))
    state = sdfg.add_state(is_start_block=True)
    dim = 2
    shape = (10,) * (dim - 1) + (1,)
    sdfg.add_array(f"A_{dim}", shape=shape, dtype=dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array(f"B_{dim}", shape=shape, dtype=dace.float64, storage=dace.StorageType.GPU_Global)
    _, me, _ = state.add_mapped_tasklet(
        f"map_{dim}",
        map_ranges={f"__i{i}": f"0:{s}" for i, s in enumerate(shape)},
        inputs={"__in": dace.Memlet(f"A_{dim}[{','.join(f'__i{i}' for i in range(dim))}]")},
        code="__out = math.cos(__in)",
        outputs={"__out": dace.Memlet(f"B_{dim}[{','.join(f'__i{i}' for i in range(dim))}]")},
        external_edges=True,
    )
    sdfg.validate()
    sdfg.apply_gpu_transformations()
    gtx_dace_fieldview_gpu_utils.gt_set_gpu_blocksize(
        sdfg=sdfg,
        block_size_1d=(128, 1, 1),
        block_size_2d=(64, 2, 1),
        block_size_3d=(2, 2, 32),
        block_size=(32, 4, 1),
        gpu_maxnreg=128,
    )
    assert me.gpu_maxnreg == 128


def _build_amd_heuristic_gpu_map(
    n_vert: int | str,
    n_horiz: int | str,
    num_dependent_inputs: int = 1,
) -> tuple[dace.SDFG, dace_nodes.MapEntry]:
    """Builds a 2D GPU map suitable for exercising the AMD block-size heuristic.

    The map iterates `i0` (vertical, size `n_vert`) x `i1` (horizontal, size
    `n_horiz`). It reads one input that does not depend on `i0` (`indep[i1]`)
    and `num_dependent_inputs` inputs that do (`dep_k[i0, i1]`), each
    `float64` (8 bytes), and writes to `out[i0, i1]`.
    """
    sdfg = dace.SDFG(util.unique_name("amd_heuristic_gpu_sdfg"))
    state = sdfg.add_state(is_start_block=True)
    storage = dace.dtypes.StorageType.GPU_Global

    if isinstance(n_vert, str):
        sdfg.add_symbol(n_vert, dace.int32)
    if isinstance(n_horiz, str):
        sdfg.add_symbol(n_horiz, dace.int32)

    sdfg.add_array("indep", shape=(n_horiz,), dtype=dace.float64, storage=storage)
    sdfg.add_array("out", shape=(n_vert, n_horiz), dtype=dace.float64, storage=storage)
    inputs = {"__in_indep": dace.Memlet("indep[i1]")}
    code_terms = ["__in_indep"]
    for k in range(num_dependent_inputs):
        name = f"dep_{k}"
        sdfg.add_array(name, shape=(n_vert, n_horiz), dtype=dace.float64, storage=storage)
        inputs[f"__in_{name}"] = dace.Memlet(f"{name}[i0, i1]")
        code_terms.append(f"__in_{name}")

    _, me, _ = state.add_mapped_tasklet(
        "amd_heuristic_tasklet",
        map_ranges={"i0": f"0:{n_vert}", "i1": f"0:{n_horiz}"},
        inputs=inputs,
        code=f"__out = {' + '.join(code_terms)}",
        outputs={"__out": dace.Memlet("out[i0, i1]")},
        external_edges=True,
        schedule=dace.dtypes.ScheduleType.GPU_Device,
    )
    sdfg.validate()
    return sdfg, me


@pytest.mark.parametrize(
    "n_vert, expected_config",
    [
        (2, gtx_amd_block_heuristic.FLAT),
        (5, gtx_amd_block_heuristic.HBLK),
    ],
)
def test_amd_heuristic_small_horizontal(
    n_vert: int, expected_config: gtx_amd_block_heuristic.Config
):
    """For small `n_horiz` the choice between `FLAT` and `HBLK` depends only on `n_vert`."""
    sdfg, me = _build_amd_heuristic_gpu_map(n_vert=n_vert, n_horiz=300)

    configured_maps = gtx_dace_fieldview_gpu_utils.gt_set_gpu_blocksize(
        sdfg=sdfg, block_size=None, amd_heuristic=True
    )

    assert configured_maps == 1
    assert me.map.gpu_block_size == [expected_config.block_x, expected_config.block_y, 1]


@pytest.mark.parametrize(
    "num_dependent_inputs, expected_config",
    [
        (3, gtx_amd_block_heuristic.HBLK),  # ratio = 8 / (8 + 3 * 8) = 0.25 < 0.30
        (1, gtx_amd_block_heuristic.DEEP),  # ratio = 8 / (8 + 8) = 0.5 >= 0.30
    ],
)
def test_amd_heuristic_large_horizontal_ratio(
    num_dependent_inputs: int, expected_config: gtx_amd_block_heuristic.Config
):
    """For large, tile-aligned `n_horiz` the choice between `HBLK` and `DEEP` depends on the
    independent/total input-byte ratio.
    """
    n_horiz = gtx_amd_block_heuristic.TILE * 500  # > LARGE_NHORIZ, divisible by TILE
    sdfg, me = _build_amd_heuristic_gpu_map(
        n_vert=20, n_horiz=n_horiz, num_dependent_inputs=num_dependent_inputs
    )

    configured_maps = gtx_dace_fieldview_gpu_utils.gt_set_gpu_blocksize(
        sdfg=sdfg, block_size=None, amd_heuristic=True
    )

    assert configured_maps == 1
    assert me.map.gpu_block_size == [expected_config.block_x, expected_config.block_y, 1]


def test_amd_heuristic_priority_over_explicit_block_size():
    """The heuristic overrides an explicitly configured `block_size_2d`."""
    sdfg, me = _build_amd_heuristic_gpu_map(n_vert=2, n_horiz=300)

    gtx_dace_fieldview_gpu_utils.gt_set_gpu_blocksize(
        sdfg=sdfg,
        block_size=None,
        block_size_2d=(64, 8, 1),
        amd_heuristic=True,
    )

    assert me.map.gpu_block_size == [
        gtx_amd_block_heuristic.FLAT.block_x,
        gtx_amd_block_heuristic.FLAT.block_y,
        1,
    ]


def test_amd_heuristic_falls_back_for_non_2d_map():
    """The heuristic does not engage for a 1D map; the existing logic still applies."""
    sdfg = dace.SDFG(util.unique_name("amd_heuristic_1d_sdfg"))
    state = sdfg.add_state(is_start_block=True)
    storage = dace.dtypes.StorageType.GPU_Global
    sdfg.add_array("a", shape=(10,), dtype=dace.float64, storage=storage)
    sdfg.add_array("b", shape=(10,), dtype=dace.float64, storage=storage)
    _, me, _ = state.add_mapped_tasklet(
        "map_1d",
        map_ranges={"i0": "0:10"},
        inputs={"__in": dace.Memlet("a[i0]")},
        code="__out = __in",
        outputs={"__out": dace.Memlet("b[i0]")},
        external_edges=True,
        schedule=dace.dtypes.ScheduleType.GPU_Device,
    )
    sdfg.validate()

    gtx_dace_fieldview_gpu_utils.gt_set_gpu_blocksize(
        sdfg=sdfg, block_size_1d=(64, 1, 1), block_size=None, amd_heuristic=True
    )

    # Clamped from 64 down to the map's actual size (10) by the (unmodified) existing logic.
    assert me.map.gpu_block_size == [10, 1, 1]


def test_amd_heuristic_falls_back_for_unresolved_symbolic_range():
    """The heuristic does not engage when the Map's range still has free symbols."""
    sdfg, me = _build_amd_heuristic_gpu_map(n_vert="N", n_horiz="M")

    gtx_dace_fieldview_gpu_utils.gt_set_gpu_blocksize(
        sdfg=sdfg, block_size_2d=(64, 2, 1), block_size=None, amd_heuristic=True
    )

    # The heuristic cannot resolve `N`/`M` to concrete ints, so it falls back
    # to the explicitly configured `block_size_2d`.
    assert me.map.gpu_block_size == [64, 2, 1]
