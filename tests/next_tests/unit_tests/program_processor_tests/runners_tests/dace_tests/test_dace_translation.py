# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test the translation stage of the dace backend workflow."""

import pytest

import re
from typing import Callable

dace = pytest.importorskip("dace")

from gt4py._core import definitions as core_defs
from gt4py.next import common as gtx_common, config
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.runners.dace.workflow import (
    translation as dace_translation_stage,
    common as dace_wf_common,
)
from gt4py.next.type_system import type_specifications as ts

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    IDim,
    KDim,
    V2EDim,
    Vertex,
    skip_value_mesh,
)

from dace import nodes as dace_nodes


FloatType = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
FieldType = ts.FieldType(dims=[IDim], dtype=FloatType)
IntType = ts.ScalarType(kind=ts.ScalarKind.INT32)


@pytest.fixture(
    params=[core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM, core_defs.DeviceType.CPU]
)
def device_type(request) -> str:
    return request.param


@pytest.mark.parametrize("has_unit_stride", [False, True])
def test_find_constant_symbols(has_unit_stride):
    config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE = has_unit_stride
    SKIP_VALUE_MESH = skip_value_mesh(None)

    ir = itir.Program(
        id=f"find_constant_symbols_{int(has_unit_stride)}",
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=ts.FieldType(dims=[Vertex, V2EDim, KDim], dtype=FloatType)),
            itir.Sym(id="y", type=ts.FieldType(dims=[Vertex, KDim], dtype=FloatType)),
            itir.Sym(id="h_size", type=IntType),
            itir.Sym(id="v_size", type=IntType),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("it")(im.reduce("plus", im.literal_from_value(1.0))(im.deref("it"))),
                )("x"),
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
    x_size_1 = SKIP_VALUE_MESH.offset_provider_type["V2E"].max_neighbors
    x_size_2 = dace.symbol("__x_size_0", dace.int32)
    x_stride_0 = dace.symbol("__x_stride_0", dace.int32)
    x_stride_1 = dace.symbol("__x_stride_1", dace.int32)
    x_stride_2 = dace.symbol("__x_stride_2", dace.int32)
    y_size_0 = dace.symbol("__y_size_0", dace.int32)
    y_size_1 = dace.symbol("__y_size_1", dace.int32)
    y_stride_0 = dace.symbol("__y_stride_0", dace.int32)
    y_stride_1 = dace.symbol("__y_stride_1", dace.int32)
    gt_conn_V2E_size_0 = dace.symbol("__gt_conn_V2E_size_0", dace.int32)
    gt_conn_V2E_size_1 = SKIP_VALUE_MESH.offset_provider_type["V2E"].max_neighbors
    gt_conn_V2E_stride_0 = dace.symbol("__gt_conn_V2E_stride_0", dace.int32)
    gt_conn_V2E_stride_1 = dace.symbol("__gt_conn_V2E_stride_1", dace.int32)
    sdfg.add_array(
        "x",
        [x_size_0, x_size_1, x_size_2],
        dace.float64,
        strides=[x_stride_0, x_stride_1, x_stride_2],
    )
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


def _are_streams_set_to_default_stream(sdfg: dace.SDFG) -> bool:
    if "cuda" not in sdfg.init_code:  # Here 'cuda' equals 'GPU backend'.
        return False

    return (
        re.match(
            r"__dace_gpu_set_all_streams\(__state\s*,\s*(cuda|hip)StreamDefault\);",
            sdfg.init_code["cuda"].as_string,
        )
        is not None
    )


def _check_sdfg_with_async_call(sdfg: dace.SDFG) -> None:
    # Because we are using the default stream, the launch is asynchronous. Thus we
    #  have to check if there is no synchronization state. However, we will do a
    #  stronger test. Instead we will make sure that there are no synchronization
    #  calls in the _entire_ generated code.
    # NOTE: Even in asynchronous launch, there might be some need for synchronization,
    #   for example if something is computed in a kernel and used on an interstate
    #   edge. However, we do not have that case.

    assert not any(
        state.label == "sync_state"
        for state in sdfg.sink_nodes()
        if isinstance(state, dace.SDFGState)
    )
    # The synchronization calls are in the CPU not the GPU code.
    cpu_code = sdfg.generate_code()[0].clean_code
    assert re.match(r"\b(cuda|hip)StreamSynchronize\b", cpu_code) is None
    assert _are_streams_set_to_default_stream(sdfg)


def _check_sdfg_without_async_call(sdfg: dace.SDFG) -> None:
    states = sdfg.states()
    sink_states = sdfg.sink_nodes()

    # Test if the distinctive sink node is present.
    assert len(sink_states) == 1
    assert len(sink_states) < len(states)
    sync_state = sink_states[0]
    assert isinstance(sync_state, dace.SDFGState)
    assert sync_state.label == "sync_state"
    assert sync_state.nosync == True  # Because sync is done through the tasklet.
    assert sync_state.number_of_nodes() == 1

    sync_tlet = next(iter(sync_state.nodes()))
    assert isinstance(sync_tlet, dace_nodes.Tasklet)
    assert sync_tlet.side_effects
    assert sync_tlet.label == "sync_tlet"

    assert re.match(r"(cuda|hip)StreamSynchronize\(\1StreamDefault\)", sync_tlet.code.as_string)
    assert _are_streams_set_to_default_stream(sdfg)


def _check_cpu_sdfg_call(sdfg: dace.SDFG) -> None:
    # CPU is always synchron execution, thus we check that there is no sync state.
    assert not any(
        state.label == "sync_state"
        for state in sdfg.sink_nodes()
        if isinstance(state, dace.SDFGState)
    )
    cpu_code = sdfg.generate_code()[0].clean_code
    assert re.match(r"\b(cuda|hip)StreamSynchronize\b", cpu_code) is None


@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    "make_async_sdfg_call",
    [False, True],
)
def test_generate_sdfg_async_call(make_async_sdfg_call: bool, device_type: core_defs.DeviceType):
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
        device_type=device_type,
        auto_optimize=False,
        async_sdfg_call=make_async_sdfg_call,
    ).generate_sdfg(program, offset_provider={}, column_axis=None)

    if device_type == core_defs.DeviceType.CPU:
        _check_cpu_sdfg_call(sdfg)
    elif make_async_sdfg_call:
        _check_sdfg_with_async_call(sdfg)
    else:
        _check_sdfg_without_async_call(sdfg)


@pytest.mark.requires_gpu
def test_generate_sdfg_async_call_no_map(device_type: core_defs.DeviceType):
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
        device_type=device_type,
        auto_optimize=False,
        async_sdfg_call=True,
    ).generate_sdfg(program, offset_provider={}, column_axis=None)

    if device_type == core_defs.DeviceType.CPU:
        _check_cpu_sdfg_call(sdfg)
    else:
        _check_sdfg_with_async_call(sdfg)


def _make_multi_state_sdfg_0(
    sdfg_name: str = "async_call_multi_state_0",
) -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    """Make an SDFG with two states, no data descriptor is accessed on the InterState edge."""
    sdfg = dace.SDFG(sdfg_name)
    R = sdfg.add_symbol("R", dace.int32)
    X_size = sdfg.add_symbol("X_size", dace.int32)
    sdfg.add_scalar("T_GPU", dace.int32, storage=dace.StorageType.GPU_Global)
    sdfg.add_scalar("T", dace.int32, transient=True)

    X, _ = sdfg.add_array("X", [X_size], dace.int32)

    first_state = sdfg.add_state()
    t_gpu_1 = first_state.add_access("T_GPU")
    t_cpu_1 = first_state.add_access("T")

    first_state.add_mapped_tasklet(
        "write",
        map_ranges={"i": "0"},
        inputs={},
        code="val = 10",
        outputs={"val": dace.Memlet("T_GPU[i]")},
        output_nodes={t_gpu_1},
        external_edges=True,
        schedule=dace.ScheduleType.GPU_Device,
    )
    first_state.add_nedge(t_gpu_1, t_cpu_1, dace.Memlet("T_GPU[0] -> [0]"))

    # The second map does not need to be on GPU, it just has to use the value that is
    #  computed by the first GPU Map.
    second_state = sdfg.add_state_after(first_state)
    second_state.add_mapped_tasklet(
        "compute",
        map_ranges=dict(i=f"0:{R}"),
        code="val = 1.0",
        inputs={},
        outputs={"val": dace.Memlet(data=X, subset="i")},
        external_edges=True,
    )
    sdfg.out_edges(first_state)[0].data.assignments["R"] = "1"
    sdfg.out_edges(first_state)[0].data.assignments["S"] = "False"
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
    sdfg.out_edges(first_state)[0].data.assignments["S"] = "False"
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


def _make_multi_state_sdfg_3(
    sdfg_name: str = "async_call_multi_state_3",
) -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    """
    Essentially like `async_call_multi_state_0`, but this time the CPU data is also
    used inside the first state, thus a sync after the call is needed.
    """
    sdfg, first_state, second_state = _make_multi_state_sdfg_0(sdfg_name)
    sdfg.add_array("U", shape=(1,), dtype=dace.int32, transient=False)

    t_cpu_1 = next(iter(dnode for dnode in first_state.data_nodes() if dnode.data == "T"))
    u_cpu_1 = first_state.add_access("U")
    tlet1 = first_state.add_tasklet(
        "cpu_computation", inputs={"__in"}, code="__out = __in + 1", outputs={"__out"}
    )

    first_state.add_edge(t_cpu_1, None, tlet1, "__in", dace.Memlet("T[0]"))
    first_state.add_edge(tlet1, "__out", u_cpu_1, None, dace.Memlet("U[0]"))

    return sdfg, first_state, second_state


@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    "multi_state_config",
    [
        (True, _make_multi_state_sdfg_0),
        (False, _make_multi_state_sdfg_1),
        (False, _make_multi_state_sdfg_2),
        (False, _make_multi_state_sdfg_3),
    ],
)
def test_generate_sdfg_async_call_multi_state(
    multi_state_config: tuple[bool, Callable], device_type: core_defs.DeviceType
):
    """
    Verify that states are not made async when a data descriptor is accessed
    on an outgoing InterState edge.
    """
    expect_async_sdfg_call_on_first_state, make_multi_state_sdfg = multi_state_config
    sdfg, first_state, second_state = make_multi_state_sdfg()

    # NOTE: Here we should use a configuration context. But because of
    #   [DaCe issue#2125](https://github.com/spcl/dace/issues/2125) this is not possible.
    dace_wf_common.set_dace_config(device_type=device_type)
    dace_translation_stage.make_sdfg_call_async(sdfg, device_type != core_defs.DeviceType.CPU)
    if device_type != core_defs.DeviceType.CPU:
        assert _are_streams_set_to_default_stream(sdfg)

    # No synchronization state is added.
    assert sdfg.number_of_nodes() == 2
    assert sdfg.out_degree(first_state) == 1 and sdfg.in_degree(first_state) == 0
    assert sdfg.out_degree(second_state) == 0 and sdfg.in_degree(second_state) == 1

    # We do never a sync.
    assert first_state.nosync == False
    assert second_state.nosync == False

    if device_type == core_defs.DeviceType.CPU:
        _check_cpu_sdfg_call(sdfg)
    elif expect_async_sdfg_call_on_first_state:
        # NOTE: This test is plain wrong! Because there is a dependency between the first and the
        #   second state. This is because the Map in the first state computes something that is
        #   used on the Interstate edge. Thus there should be a sync at the end of the first
        #   state. But as the test bellow shows, there is no sync in the enter CPU code (syncs
        #   are never inside the GPU code). This is plain wrong, but we should not be affected
        #   by this. See https://github.com/spcl/dace/issues/2120 for more.
        #   In the case of `_make_multi_state_sdfg_3()` there would be a sync after the Map, before
        #   the Tasklet, if the default stream was not used!
        cpu_code = sdfg.generate_code()[0].clean_code
        assert re.match(r"(cuda|hip)StreamSynchronize\b", cpu_code) is None
    else:
        # There is no dependency between the states, so no sync.
        cpu_code = sdfg.generate_code()[0].clean_code
        assert re.match(r"(cuda|hip)StreamSynchronize\b", cpu_code) is None
