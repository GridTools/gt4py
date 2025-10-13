# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test the translation stage of the dace backend workflow."""

from unittest import mock
import pytest

import re
from typing import Callable

dace = pytest.importorskip("dace")

from gt4py._core import definitions as core_defs
from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.runners.dace.workflow import (
    translation as dace_wf_translation,
    common as dace_wf_common,
)
from gt4py.next.type_system import type_specifications as ts

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    V2E,
    Edge,
    IDim,
    Vertex,
    skip_value_mesh,
)

from dace import nodes as dace_nodes


FLOAT_TYPE = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
IFTYPE = ts.FieldType(dims=[IDim], dtype=FLOAT_TYPE)
EFTYPE = ts.FieldType(dims=[Edge], dtype=FLOAT_TYPE)
VFTYPE = ts.FieldType(dims=[Vertex], dtype=FLOAT_TYPE)


@pytest.fixture(
    params=[
        pytest.param(core_defs.DeviceType.CPU),
        pytest.param(core_defs.DeviceType.CUDA, marks=[pytest.mark.requires_gpu]),
        pytest.param(core_defs.DeviceType.ROCM, marks=[pytest.mark.requires_gpu]),
    ]
)
def device_type(request) -> str:
    return request.param


def _translate_gtir_to_sdfg(
    ir: itir.Program,
    offset_provider: gtx_common.OffsetProvider,
    device_type: core_defs.DeviceType,
    auto_optimize: bool,
    async_sdfg_call: bool,
    use_metrics: bool = False,
) -> dace.SDFG:
    with dace.config.set_temporary("cache", value="hash"):
        # we use the SDFG hash in build cache to avoid clashes between CPU and GPU SDFGs
        return dace_wf_translation.DaCeTranslator(
            device_type=device_type,
            auto_optimize=auto_optimize,
            auto_optimize_args=None,
            async_sdfg_call=async_sdfg_call,
            use_metrics=use_metrics,
        ).generate_sdfg(ir, offset_provider=offset_provider, column_axis=None)


@pytest.mark.parametrize("has_unit_stride", [False, True])
@pytest.mark.parametrize("disable_field_origin", [False, True])
def test_find_constant_symbols(has_unit_stride, disable_field_origin):
    SKIP_VALUE_MESH = skip_value_mesh(None)

    ir = itir.Program(
        id="find_constant_symbols_sdfg",
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=EFTYPE),
            itir.Sym(id="y", type=VFTYPE),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("it")(im.reduce("plus", im.literal_from_value(1.0))(im.deref("it")))
                )(im.as_fieldop_neighbors(V2E.value, "x")),
                domain=im.get_field_domain(gtx_common.GridType.UNSTRUCTURED, "y", VFTYPE.dims),
                target=itir.SymRef(id="y"),
            )
        ],
    )

    with mock.patch("gt4py.next.config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE", has_unit_stride):
        sdfg = _translate_gtir_to_sdfg(
            ir=ir,
            offset_provider=SKIP_VALUE_MESH.offset_provider,
            device_type=core_defs.DeviceType.CPU,
            auto_optimize=False,
            async_sdfg_call=False,
        )

        constant_symbols = dace_wf_translation.find_constant_symbols(
            ir, sdfg, SKIP_VALUE_MESH.offset_provider_type, disable_field_origin
        )

    expected = {}
    if has_unit_stride:
        expected |= {
            "__x_stride_0": 1,
            "__y_stride_0": 1,
            "__gt_conn_V2E_stride_0": 1,
        }
    if disable_field_origin:
        expected |= {
            "__x_Edge_range_0": 0,
            "__y_Vertex_range_0": 0,
        }
    assert constant_symbols == expected


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


def _are_streams_synchronized(sdfg: dace.SDFG) -> bool:
    re_stream_sync = re.compile(r"\b(cuda|hip)StreamSynchronize\b")
    # The synchronization calls are in the CPU not the GPU code.
    return any(
        re_stream_sync.match(code.clean_code)
        for code in sdfg.generate_code()
        if code.language == "cpp"
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
    assert not _are_streams_synchronized(sdfg)
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
    # CPU is always synchronous execution, thus we check that there is no sync state.
    assert not any(
        state.label == "sync_state"
        for state in sdfg.sink_nodes()
        if isinstance(state, dace.SDFGState)
    )
    assert not _are_streams_synchronized(sdfg)


@pytest.mark.parametrize(
    "make_async_sdfg_call",
    [False, True],
)
def test_generate_sdfg_async_call(make_async_sdfg_call: bool, device_type: core_defs.DeviceType):
    """Verify that the flag `async_sdfg_call` takes effect on the SDFG generation."""
    program_name = "field_ir_{}_async_call".format("with" if make_async_sdfg_call else "without")

    ir = itir.Program(
        id=program_name,
        declarations=[],
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=IFTYPE),
            itir.Sym(id="y", type=IFTYPE),
        ],
        body=[
            itir.SetAt(
                expr=im.op_as_fieldop("plus")("x", 1.0),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "y", IFTYPE.dims),
                target=itir.SymRef(id="y"),
            ),
        ],
    )

    sdfg = _translate_gtir_to_sdfg(
        ir=ir,
        offset_provider={},
        device_type=device_type,
        auto_optimize=False,
        async_sdfg_call=make_async_sdfg_call,
    )

    if device_type == core_defs.DeviceType.CPU:
        _check_cpu_sdfg_call(sdfg)
    elif make_async_sdfg_call:
        _check_sdfg_with_async_call(sdfg)
    else:
        _check_sdfg_without_async_call(sdfg)


def test_generate_sdfg_async_call_no_map(device_type: core_defs.DeviceType):
    """Verify that the flag `async_sdfg_call=True` has no effect on an SDFG that does not contain any GPU map."""

    ir = itir.Program(
        id="scalar_ir_with_async_call",
        declarations=[],
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=IFTYPE),
            itir.Sym(id="y", type=IFTYPE),
        ],
        body=[
            itir.SetAt(
                expr=itir.SymRef(id="x"),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "y", IFTYPE.dims),
                target=itir.SymRef(id="y"),
            ),
        ],
    )

    sdfg = _translate_gtir_to_sdfg(
        ir=ir,
        offset_provider={},
        device_type=device_type,
        auto_optimize=False,
        async_sdfg_call=True,
    )

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
    on_gpu = device_type == core_defs.CUPY_DEVICE_TYPE
    expect_async_sdfg_call_on_first_state, make_multi_state_sdfg = multi_state_config
    sdfg, first_state, second_state = make_multi_state_sdfg()

    # NOTE: Here we should use a configuration context. But because of
    #   [DaCe issue#2125](https://github.com/spcl/dace/issues/2125) this is not possible.
    with dace_wf_common.dace_context(device_type=device_type):
        dace_wf_translation.make_sdfg_call_async(sdfg, on_gpu)

    if on_gpu:
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
        assert not _are_streams_synchronized(sdfg)
    else:
        # There is no dependency between the states, so no sync.
        assert not _are_streams_synchronized(sdfg)
