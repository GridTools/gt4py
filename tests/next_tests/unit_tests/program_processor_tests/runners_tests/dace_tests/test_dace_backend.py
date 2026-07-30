# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test the bindings stage of the dace backend workflow."""

import pytest
import unittest.mock as mock

dace = pytest.importorskip("dace")

from gt4py import next as gtx
from gt4py._core import definitions as core_defs
from gt4py.next.otf import arguments, artifacts, runners
from gt4py.next.program_processors.runners.dace.workflow import (
    backend as dace_wf_backend,
)
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.type_system import type_specifications as ts

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases_utils import KDim


@pytest.fixture(
    params=[
        pytest.param(core_defs.DeviceType.CPU),
        pytest.param(core_defs.CUPY_DEVICE_TYPE, marks=pytest.mark.requires_gpu),
    ],
    ids=["CPU", "GPU"],
)
def device_type(request) -> str:
    return request.param


@pytest.mark.parametrize("auto_optimize", [False, True])
def test_make_backend(auto_optimize, device_type, monkeypatch):
    on_gpu = device_type == core_defs.CUPY_DEVICE_TYPE

    @gtx.field_operator
    def testee_op(a: cases.IField, b: cases.IField) -> cases.IField:
        return a + b

    @gtx.program
    def testee(a: cases.IField, b: cases.IField, out: cases.IField):
        testee_op(a, b, out=out)

    mock_top_level_dataflow_hook1 = mock.create_autospec(gtx_transformations.GT4PyAutoOptHookStage)
    mock_top_level_dataflow_hook2 = mock.create_autospec(gtx_transformations.GT4PyAutoOptHookStage)

    if not auto_optimize:
        optimization_args = {}
    elif on_gpu:
        optimization_args = {
            "make_persistent": False,
            "gpu_block_size": (32, 8, 1),
            "gpu_block_size_2d": (20, 20),
            "gpu_memory_pool": True,
            "optimization_hooks": {
                gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowPost: mock_top_level_dataflow_hook1,
            },
        }
    else:
        optimization_args = {
            "make_persistent": True,
            "blocking_dim": KDim,
            "blocking_size": 10,
            "gpu_memory_pool": False,
            "optimization_hooks": {
                gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowPost: mock_top_level_dataflow_hook2,
            },
        }

    sdfg: dace.SDFG | None = None
    mock_auto_optimize = mock.MagicMock()
    gt_auto_optimize = gtx_transformations.gt_auto_optimize
    mock_gpu_transformation = mock.MagicMock()
    gt_gpu_transformation = gtx_transformations.gt_gpu_transformation

    def mocked_auto_optimize(*args, **kwargs) -> dace.SDFG:
        nonlocal sdfg
        sdfg = args[0]
        mock_auto_optimize.__call__(*args, **kwargs)
        return gt_auto_optimize(*args, **kwargs)

    def mocked_gpu_transformation(*args, **kwargs) -> dace.SDFG:
        mock_gpu_transformation.__call__(*args, **kwargs)
        return gt_gpu_transformation(*args, **kwargs)

    monkeypatch.setattr(gtx_transformations, "gt_auto_optimize", mocked_auto_optimize)
    monkeypatch.setattr(gtx_transformations, "gt_gpu_transformation", mocked_gpu_transformation)

    custom_backend = dace_wf_backend.make_dace_backend(
        gpu=on_gpu,
        auto_optimize=auto_optimize,
        async_sdfg_call=True,
        optimization_args=optimization_args,
        unstructured_horizontal_has_unit_stride=on_gpu,
        use_metrics=True,
    )
    # The monkeypatched transformation functions exist only in this process, so
    # compilation must not be offloaded to a worker.
    with mock.patch.object(runners, "get_default_runner", return_value=runners.SerialRunner()):
        testee.with_backend(custom_backend).compile(offset_provider={})
        gtx.wait_for_compilation()

    # check call to `gt_gpu_transformation()`
    if on_gpu:
        mock_gpu_transformation.assert_called_once()
    else:
        mock_gpu_transformation.assert_not_called()

    # check call to `gt_auto_optimize()`
    if auto_optimize:
        if on_gpu:
            mock_auto_optimize.assert_called_once_with(
                sdfg,
                gpu=on_gpu,
                constant_symbols={
                    "__a_IDim_stride": 1,
                    "__b_IDim_stride": 1,
                    "__out_IDim_stride": 1,
                },
                make_persistent=optimization_args["make_persistent"],
                gpu_block_size=optimization_args["gpu_block_size"],
                gpu_block_size_2d=optimization_args["gpu_block_size_2d"],
                gpu_memory_pool=optimization_args["gpu_memory_pool"],
                optimization_hooks=optimization_args["optimization_hooks"],
                unit_strides_kind=gtx.common.DimensionKind.HORIZONTAL,
            )
            mock_top_level_dataflow_hook1.assert_called_once_with(sdfg)
            mock_top_level_dataflow_hook2.assert_not_called()
        else:
            mock_auto_optimize.assert_called_once_with(
                sdfg,
                gpu=on_gpu,
                constant_symbols={},
                make_persistent=optimization_args["make_persistent"],
                blocking_dim=optimization_args["blocking_dim"],
                blocking_size=optimization_args["blocking_size"],
                gpu_memory_pool=optimization_args["gpu_memory_pool"],
                optimization_hooks=optimization_args["optimization_hooks"],
                unit_strides_kind=None,
            )
            mock_top_level_dataflow_hook1.assert_not_called()
            mock_top_level_dataflow_hook2.assert_called_once_with(sdfg)
    else:
        mock_auto_optimize.assert_not_called()
        mock_top_level_dataflow_hook1.assert_not_called()
        mock_top_level_dataflow_hook2.assert_not_called()


def test_translate_produces_sdfg_source():
    """`Toolchain.translate` is the sanctioned partial run: frontend + translation only."""

    @gtx.field_operator
    def testee_op(a: cases.IField) -> cases.IField:
        return a

    @gtx.program
    def testee(a: cases.IField, out: cases.IField):
        testee_op(a, out=out)

    int_field = ts.FieldType(dims=[cases.IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))
    compile_time_args = arguments.CompileTimeArgs(
        args=(int_field, int_field),
        kwargs={},
        offset_provider={},
        column_axis=None,
        argument_descriptor_contexts={},
    )

    source = dace_wf_backend.run_dace_cpu.translate(testee.definition_stage, compile_time_args)

    assert isinstance(source, artifacts.ProgramSource)
    assert source.code_spec.file_extension == "sdfg"
    assert isinstance(dace.SDFG.from_json(source.source_code), dace.SDFG)
