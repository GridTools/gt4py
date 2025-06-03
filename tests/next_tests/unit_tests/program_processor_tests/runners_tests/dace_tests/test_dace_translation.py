# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test the translation stage of the dace backend workflow."""

import pytest

dace = pytest.importorskip("dace")

from gt4py._core import definitions as core_defs
from gt4py.next import common as gtx_common, config
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.runners.dace.workflow import (
    translation as dace_translation_stage,
)
from gt4py.next.type_system import type_specifications as ts

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    IDim,
    KDim,
    Vertex,
    skip_value_mesh,
)


FloatType = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
FieldType = ts.FieldType(dims=[IDim], dtype=FloatType)
IntType = ts.ScalarType(kind=ts.ScalarKind.INT32)


@pytest.mark.parametrize("has_unit_stride", [False, True])
def test_find_constant_symbols(has_unit_stride):
    config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE = has_unit_stride
    SKIP_VALUE_MESH = skip_value_mesh(None)

    ir = itir.Program(
        id=f"find_constant_symbols_{int(has_unit_stride)}",
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=ts.FieldType(dims=[Vertex], dtype=FloatType)),
            itir.Sym(id="y", type=ts.FieldType(dims=[Vertex, KDim], dtype=FloatType)),
            itir.Sym(id="h_size", type=IntType),
            itir.Sym(id="v_size", type=IntType),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.op_as_fieldop("plus")("x", 1.0),
                domain=im.domain(
                    gtx_common.GridType.UNSTRUCTURED,
                    ranges={Vertex: (0, "h_size"), KDim: (0, "v_size")},
                ),
                target=itir.SymRef(id="y"),
            )
        ],
    )

    sdfg = dace.SDFG(ir.id)
    x_size_0 = dace.symbol("__x_size_0", dace.int32)
    x_stride_0 = dace.symbol("__x_stride_0", dace.int32)
    y_size_0 = dace.symbol("__y_size_0", dace.int32)
    y_size_1 = dace.symbol("__y_size_1", dace.int32)
    y_stride_0 = dace.symbol("__y_stride_0", dace.int32)
    y_stride_1 = dace.symbol("__y_stride_1", dace.int32)
    gt_conn_V2E_size_0 = dace.symbol("__gt_conn_V2E_size_0", dace.int32)
    gt_conn_V2E_size_1 = SKIP_VALUE_MESH.offset_provider_type["V2E"].max_neighbors
    gt_conn_V2E_stride_0 = dace.symbol("__gt_conn_V2E_stride_0", dace.int32)
    gt_conn_V2E_stride_1 = dace.symbol("__gt_conn_V2E_stride_1", dace.int32)
    sdfg.add_array("x", [x_size_0], dace.float64, strides=[x_stride_0])
    sdfg.add_array("y", [y_size_0, y_size_1], dace.float64, strides=[y_stride_0, y_stride_1])
    sdfg.add_array(
        "gt_conn_V2E",
        [gt_conn_V2E_size_0, gt_conn_V2E_size_1],
        dace.int32,
        strides=[gt_conn_V2E_stride_0, gt_conn_V2E_stride_1],
    )

    for i, data in enumerate(["x", "y"]):
        assert len(ir.params[i].type.dims) == len(sdfg.arrays[data].shape)

    constant_symbols = dace_translation_stage.find_constant_symbols(
        ir, sdfg, offset_provider_type=SKIP_VALUE_MESH.offset_provider_type
    )
    if has_unit_stride:
        assert all(sym in sdfg.free_symbols for sym in constant_symbols.keys())
        assert constant_symbols == {
            "__x_stride_0": 1,
            "__y_stride_0": 1,
            "__gt_conn_V2E_stride_0": 1,
        }
    else:
        assert len(constant_symbols) == 0


def _is_present_async_sdfg_init_code(sdfg: dace.SDFG) -> bool:
    async_sdfg_init_code = "__dace_gpu_set_all_streams(__state, cudaStreamDefault);"

    if "cuda" not in sdfg.init_code:
        return False

    n_match_lines = len(
        [line for line in sdfg.init_code["cuda"].code.splitlines() if line == async_sdfg_init_code]
    )
    assert n_match_lines <= 1
    return n_match_lines == 1


def _check_sdfg_with_async_call(sdfg: dace.SDFG) -> None:
    assert len(sdfg.states()) == 1
    st = sdfg.states()[0]
    assert st.nosync == True

    assert _is_present_async_sdfg_init_code(sdfg)


def _check_sdfg_without_async_call(sdfg: dace.SDFG) -> None:
    assert len(sdfg.states()) == 1
    st = sdfg.states()[0]
    assert st.nosync == False

    assert not _is_present_async_sdfg_init_code(sdfg)


@pytest.mark.parametrize(
    "make_async_sdfg_call",
    [False, True],
)
def test_generate_sdfg_async_call(make_async_sdfg_call):
    """Verify that the flag `async_sdfg_call` takes effect on the SDFG generation."""
    program_name = "field_ir_{}_async_call".format("with" if make_async_sdfg_call else "without")

    program = itir.Program(
        id=program_name,
        declarations=[],
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=FieldType),
            itir.Sym(id="y", type=FieldType),
            itir.Sym(id="N", type=IntType),
        ],
        body=[
            itir.SetAt(
                expr=im.op_as_fieldop("plus")("x", 1.0),
                domain=im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "N")}),
                target=itir.SymRef(id="y"),
            ),
        ],
    )

    sdfg = dace_translation_stage.DaCeTranslator(
        device_type=core_defs.CUPY_DEVICE_TYPE,
        auto_optimize=False,
        async_sdfg_call=make_async_sdfg_call,
    ).generate_sdfg(program, offset_provider={}, column_axis=None)

    if make_async_sdfg_call:
        _check_sdfg_with_async_call(sdfg)
    else:
        _check_sdfg_without_async_call(sdfg)


def test_generate_sdfg_async_call_no_map():
    """Verify that the flag `async_sdfg_call=True` has no effect on an SDFG that does not contain any GPU map."""
    program_name = "scalar_ir_with_async_call"

    program = itir.Program(
        id=program_name,
        declarations=[],
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=FieldType),
            itir.Sym(id="y", type=FieldType),
            itir.Sym(id="N", type=IntType),
        ],
        body=[
            itir.SetAt(
                expr=itir.SymRef(id="x"),
                domain=im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "N")}),
                target=itir.SymRef(id="y"),
            ),
        ],
    )

    sdfg = dace_translation_stage.DaCeTranslator(
        device_type=core_defs.CUPY_DEVICE_TYPE,
        auto_optimize=False,
        async_sdfg_call=True,
    ).generate_sdfg(program, offset_provider={}, column_axis=None)

    _check_sdfg_without_async_call(sdfg)


def _make_multi_state_sdfg_0(
    sdfg_name: str = "async_call_multi_state_0",
) -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    """Make an SDFG with two states, no data descriptor is accessed on the InterState edge."""
    sdfg = dace.SDFG(sdfg_name)
    R = sdfg.add_symbol("R", dace.int32)
    X_size = sdfg.add_symbol("X_size", dace.int32)
    T, _ = sdfg.add_scalar("T", dace.int32)
    X, _ = sdfg.add_array("X", [X_size], dace.int32)

    first_state = sdfg.add_state()
    twrite = first_state.add_tasklet(
        "write",
        code="val = 10",
        inputs={},
        outputs={"val"},
    )
    first_state.add_edge(
        twrite, "val", first_state.add_access(T), None, dace.Memlet(data=T, subset="0")
    )
    second_state = sdfg.add_state_after(first_state)
    second_state.add_mapped_tasklet(
        "compute",
        map_ranges=dict(i=f"0:{R}"),
        code="val = 1.0",
        inputs={},
        outputs={"val": dace.Memlet(data=X, subset="i")},
        external_edges=True,
    )
    sdfg.out_edges(first_state)[0].data.assignments["R"] = 1
    return sdfg, first_state, second_state


def _make_multi_state_sdfg_1(
    sdfg_name: str = "async_call_multi_state_1",
) -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    """
    Make an SDFG with two states, the data descriptor 'T' is assigned to 'R'
    on the InterState edge.
    """
    sdfg, first_state, second_state = _make_multi_state_sdfg_0(sdfg_name)
    sdfg.out_edges(first_state)[0].data.assignments["R"] = "T"
    return sdfg, first_state, second_state


def _make_multi_state_sdfg_2(
    sdfg_name: str = "async_call_multi_state_2",
) -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    """
    Make an SDFG with two states, the data descriptor 'T' is used in comparison condition
    on the InterState edge.
    """
    sdfg, first_state, second_state = _make_multi_state_sdfg_0(sdfg_name)
    sdfg.out_edges(first_state)[0].data.condition = dace.properties.CodeBlock("R > T")
    return sdfg, first_state, second_state


@pytest.mark.parametrize(
    "multi_state_config",
    [
        (True, _make_multi_state_sdfg_0),
        (False, _make_multi_state_sdfg_1),
        (False, _make_multi_state_sdfg_2),
    ],
)
def test_generate_sdfg_async_call_multi_state(multi_state_config):
    """
    Verify that states are not made async when a data descriptor is accessed
    on an outgoing InterState edge.
    """
    expect_async_sdfg_call_on_first_state, make_multi_state_sdfg = multi_state_config
    sdfg, first_state, second_state = make_multi_state_sdfg()

    dace_translation_stage.make_sdfg_async(sdfg)

    assert _is_present_async_sdfg_init_code(sdfg)

    if expect_async_sdfg_call_on_first_state:
        assert first_state.nosync == True
        assert second_state.nosync == True
    else:
        assert first_state.nosync == False
        assert second_state.nosync == True
