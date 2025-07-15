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

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace import (
    gtir_to_sdfg_utils as gtx_sdfg_utils,
    transformations as gtx_transformations,
)

import copy


from . import util

N = 10


def _make_serial_map_promotion_sdfg(
    use_symbolic_range: bool = False,
) -> tuple[dace.SDFG, dace.SDFGState, dace_nodes.MapEntry, dace_nodes.MapEntry]:
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
        map_ranges=[("__i0", f"0:{N}"), ("__i1", "R0:R1" if use_symbolic_range else f"0:{N}")],
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
    count = sdfg.apply_transformations_repeated(
        gtx_transformations.SerialMapPromoter(
            promote_everything=True,
            # Do not fuse to inspect that the promotion worked.
            fuse_after_promotion=False,
            single_use_data={sdfg: {"tmp"}},
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


@pytest.mark.parametrize(
    "use_symbolic_range",
    [False, True],
)
@pytest.mark.parametrize(
    "single_use_data",
    [{"tmp"}, {}],
)
def test_serial_map_promotion_promote_and_merge(use_symbolic_range, single_use_data):
    sdfg, state, map_entry_1d, map_entry_2d = _make_serial_map_promotion_sdfg(use_symbolic_range)

    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2
    assert len(map_entry_1d.map.params) == 1
    assert len(map_entry_1d.map.range) == 1
    assert len(map_entry_2d.map.params) == 2
    assert len(map_entry_2d.map.range) == 2

    original_2d_params = list(map_entry_2d.map.params)
    original_2d_range = copy.deepcopy(map_entry_2d.map.range)

    # Now apply the promotion
    count = sdfg.apply_transformations_repeated(
        gtx_transformations.SerialMapPromoter(
            promote_everything=True,
            fuse_after_promotion=True,
            single_use_data={sdfg: single_use_data},
        ),
        validate=True,
        validate_all=True,
    )

    mes = util.count_nodes(sdfg, dace_nodes.MapEntry, True)

    if use_symbolic_range and len(single_use_data) == 0:
        # The range of the second map is purely symbolic, therefore it is unknown
        # whether the range is empty or not at compile time. Besides, the temporary
        # produced by the first map (1d) is not single-use data, thus it is needed
        # in another state of the SDFG. Since the temporary is needed in another
        # location and we are not sure that the extended range will be not-empty,
        # map promotion should not be applied.
        assert count == 0
        assert len(mes) == 2
    else:
        assert count == 1
        assert len(mes) == 1
        me = mes[0]

        assert len(me.map.params) == 2
        assert me.map.params == original_2d_params
        assert len(me.map.range) == 2
        assert me.map.range == original_2d_range


@pytest.mark.parametrize(
    "use_symbolic_range",
    [False, True],
)
def test_serial_map_promotion_on_symbolic_range(use_symbolic_range):
    sdfg, state, map_entry_1d, map_entry_2d = _make_serial_map_promotion_sdfg(use_symbolic_range)

    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2
    assert len(map_entry_1d.map.params) == 1
    assert len(map_entry_1d.map.range) == 1
    assert len(map_entry_2d.map.params) == 2
    assert len(map_entry_2d.map.range) == 2

    # Now add a nother map that consumes the temporary array
    S = dace.symbol("S", dace.int32)
    tmp_nodes = list(
        iedge.src
        for iedge in state.in_edges(map_entry_2d)
        if isinstance(iedge.src, dace_nodes.AccessNode) and iedge.src.desc(sdfg).transient
    )
    assert len(tmp_nodes) == 1
    sdfg.add_array(
        name="d",
        shape=(N, S),
        dtype=dace.float64,
        transient=False,
    )
    state.add_mapped_tasklet(
        name="another_two_d_map",
        map_ranges=[("__i0", f"0:{N}"), ("__i1", f"0:{S}")],
        input_nodes={"tmp": tmp_nodes[0]},
        inputs={"__in0": dace.Memlet("tmp[__i0]"), "__in1": dace.Memlet("b[__i0, __i1]")},
        code="__out = __in0 + __in1",
        outputs={"__out": dace.Memlet("d[__i0, __i1]")},
        external_edges=True,
    )

    count = sdfg.apply_transformations(
        gtx_transformations.SerialMapPromoter(
            promote_everything=True,
            fuse_after_promotion=True,
            single_use_data={sdfg: {"tmp"}},
        ),
        validate=True,
        validate_all=True,
    )

    mes = util.count_nodes(sdfg, dace_nodes.MapEntry, True)
    if use_symbolic_range:
        # The promotion should not apply because there are two consumer maps, both
        # with symbolic range. If the symbolic range happens to be empty, there won't
        # be valid data in the temporay for the other consumer map.
        assert count == 0
        assert len(mes) == 3
    else:
        # One the of the two consumer maps has compile-time size, with not-empty range.
        # We can apply map promotion because we know that the temporary will be computed.
        assert count == 1
        assert len(mes) == 2


def test_serial_map_promotion_2d_top_1d_bottom():
    """Does not apply because the bottom map needs to be promoted."""

    sdfg = dace.SDFG(util.unique_name("serial_map_promoter_2d_map_on_top"))
    state = sdfg.add_state(is_start_block=True)

    # 2D Arrays
    for name in ["a", "t"]:
        sdfg.add_array(name=name, shape=(20, 10), dtype=dace.float64, transient=(name == "t"))

    # 1D Arrays
    sdfg.add_array(
        name="b",
        shape=(20,),
        dtype=dace.float64,
        transient=False,
    )

    t = state.add_access("t")
    state.add_mapped_tasklet(
        "comp_2d",
        map_ranges={
            "__i": "0:20",
            "__j": "0:10",
        },
        inputs={"__in": dace.Memlet("a[__i, __j]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("t[__i, __j]")},
        output_nodes={t},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "comp_1d",
        map_ranges={"__i": "0:20"},
        inputs={"__in": dace.Memlet("t[__i, 2]")},
        code="__out = __in + 3.0",
        outputs={"__out": dace.Memlet("b[__i]")},
        input_nodes={t},
        external_edges=True,
    )
    sdfg.validate()

    map_promoter = gtx_transformations.SerialMapPromoter(
        promote_everything=True,
        fuse_after_promotion=False,
        single_use_data={sdfg: {"t"}},
    )
    # Because of the near nonsensical SDFG we are bypassing the test.
    map_promoter._bypass_fusion_test = True

    count = sdfg.apply_transformations_repeated(
        map_promoter,
        validate_all=True,
        validate=True,
    )

    assert count == 0


def _make_horizontal_promoter_sdfg(
    d1_map_is_vertical: bool,
) -> tuple[dace.SDFG, dace.SDFGState, dace_nodes.MapEntry, dace_nodes.MapEntry]:
    sdfg = dace.SDFG(util.unique_name("serial_map_promoter_tester"))
    state = sdfg.add_state(is_start_block=True)

    h_idx = gtx_sdfg_utils.get_map_variable(gtx_common.Dimension("boden"))
    v_idx = gtx_sdfg_utils.get_map_variable(
        gtx_common.Dimension("K", gtx_common.DimensionKind.VERTICAL)
    )

    if d1_map_is_vertical:
        d1_shape = (10,)
        d1_idx = v_idx
        d1_range = "0:10"
    else:
        d1_shape = (20,)
        d1_idx = h_idx
        d1_range = "0:20"

    # 1D Arrays
    for name in ["a", "t"]:
        sdfg.add_array(
            name=name,
            shape=d1_shape,
            dtype=dace.float64,
            transient=(name == "t"),
        )

    # 2D Arrays
    sdfg.add_array(
        name="b",
        shape=(20, 10),
        dtype=dace.float64,
        transient=False,
    )

    t = state.add_access("t")
    _, me1d, _ = state.add_mapped_tasklet(
        "pure_vertical_computation",
        map_ranges={d1_idx: d1_range},
        inputs={"__in": dace.Memlet(f"a[{d1_idx}]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet(f"t[{d1_idx}]")},
        output_nodes={t},
        external_edges=True,
    )
    _, me2d, _ = state.add_mapped_tasklet(
        "h_and_v_computation",
        map_ranges={
            v_idx: "0:10",
            h_idx: "0:20",
        },
        inputs={"__in": dace.Memlet(f"t[{d1_idx}]")},
        code=f"__out = __in + float({h_idx}) * float({v_idx})",
        outputs={"__out": dace.Memlet(f"b[{h_idx}, {v_idx}]")},
        input_nodes={t},
        external_edges=True,
    )
    sdfg.validate()

    return sdfg, state, me1d, me2d


@pytest.mark.parametrize("d1_map_is_vertical", [True, False])
def test_horizonal_promotion_only_promotion(d1_map_is_vertical: bool):
    sdfg, state, me1d, me2d = _make_horizontal_promoter_sdfg(d1_map_is_vertical=d1_map_is_vertical)
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 3
    assert set(util.count_nodes(sdfg, dace_nodes.MapEntry, True)) == {me1d, me2d}
    assert len(me2d.map.params) == 2
    assert len(me1d.map.params) == 1

    original_2d_params = list(me2d.map.params)
    original_2d_range = copy.deepcopy(me2d.map.range)

    # If we do not allow promotion in horizontal it will not work.
    count = sdfg.apply_transformations_repeated(
        gtx_transformations.SerialMapPromoter(
            promote_horizontal=False,
            promote_vertical=False,
            promote_local=True,
            fuse_after_promotion=False,
            single_use_data={sdfg: {"t"}},
        ),
        validate=True,
        validate_all=True,
    )
    assert count == 0

    # We have to allow the promotion of horizontal explicitly.
    count = sdfg.apply_transformations_repeated(
        gtx_transformations.SerialMapPromoter(
            promote_horizontal=d1_map_is_vertical,
            promote_vertical=not d1_map_is_vertical,
            promote_local=False,
            fuse_after_promotion=False,
            single_use_data={sdfg: {"t"}},
        ),
        validate=True,
        validate_all=True,
    )
    assert count == 1

    assert len(me1d.map.params) == 2
    assert len(me2d.map.range) == 2
    assert me1d.map.params == original_2d_params
    assert me1d.map.range == original_2d_range


@pytest.mark.parametrize("d1_map_is_vertical", [True, False])
def test_horizonal_promotion_promotion_and_merge(d1_map_is_vertical: bool):
    sdfg, state, me1d, me2d = _make_horizontal_promoter_sdfg(d1_map_is_vertical=d1_map_is_vertical)
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 3
    assert set(util.count_nodes(sdfg, dace_nodes.MapEntry, True)) == {me1d, me2d}
    assert len(me2d.map.params) == 2
    assert len(me1d.map.params) == 1

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.SerialMapPromoter(
            promote_horizontal=d1_map_is_vertical,
            promote_vertical=not d1_map_is_vertical,
            promote_local=False,
            single_use_data={sdfg: {"t"}},
        ),
        validate=True,
        validate_all=True,
    )
    assert count == 1

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)


def _make_sdfg_different_1d_map_name(
    d1_map_param: str,
) -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("serial_map_promoter_different_names_" + d1_map_param))
    state = sdfg.add_state(is_start_block=True)

    # 1D Arrays
    for name in ["a", "t"]:
        sdfg.add_array(
            name=name,
            shape=(10,),
            dtype=dace.float64,
            transient=(name == "t"),
        )

    # 2D Arrays
    sdfg.add_array(
        name="b",
        shape=(20, 10),
        dtype=dace.float64,
        transient=False,
    )

    t = state.add_access("t")
    state.add_mapped_tasklet(
        "pure_vertical_computation",
        map_ranges={d1_map_param: "0:10"},
        inputs={"__in": dace.Memlet(f"a[{d1_map_param}]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet(f"t[{d1_map_param}]")},
        output_nodes={t},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "h_and_v_computation",
        map_ranges={
            "__j": "0:20",
            "__i": "0:10",
        },
        inputs={"__in": dace.Memlet("t[__i]")},
        code=f"__out = __in + float(__i) * float(__j)",
        outputs={"__out": dace.Memlet(f"b[__j, __i]")},
        input_nodes={t},
        external_edges=True,
    )
    return sdfg, state


def test_map_promotion_different_parameter_names():
    # The transformation will not apply because the first Map has the parameter
    #  `__k` which is not used by the second Map.
    # NOTE: The design of `gt_auto_optimizer()` depends on that.
    sdfg_k, _ = _make_sdfg_different_1d_map_name("__k")
    assert util.count_nodes(sdfg_k, dace_nodes.MapEntry) == 2
    count = sdfg_k.apply_transformations_repeated(
        gtx_transformations.SerialMapPromoter(
            promote_everything=True,
            fuse_after_promotion=True,
            single_use_data={sdfg_k: {"t"}},
        ),
        validate=True,
        validate_all=True,
    )
    assert count == 0

    # However, if they have the same name, then we can merge them.
    sdfg_i, _ = _make_sdfg_different_1d_map_name("__i")
    assert util.count_nodes(sdfg_i, dace_nodes.MapEntry) == 2
    count = sdfg_i.apply_transformations_repeated(
        gtx_transformations.SerialMapPromoter(
            promote_everything=True,
            fuse_after_promotion=True,
            single_use_data={sdfg_i: {"t"}},
        ),
        validate=True,
        validate_all=True,
    )
    assert count == 1
    assert util.count_nodes(sdfg_i, dace_nodes.MapEntry) == 1
