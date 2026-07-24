# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test the bindings stage of the dace backend workflow."""

import re

import numpy as np
import pytest
import unittest.mock as mock

dace = pytest.importorskip("dace")

from gt4py import next as gtx
from gt4py._core import definitions as core_defs
from gt4py.next.otf import runners
from gt4py.next.program_processors.runners.dace.workflow import (
    backend as dace_wf_backend,
)
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.transformations import (
    auto_optimize as gtx_auto_optimize,
)

from next_tests.integration_tests import cases
from next_tests.integration_tests import cases_utils
from next_tests.integration_tests.cases_utils import KDim


@pytest.fixture(
    params=[
        pytest.param(core_defs.DeviceType.CPU),
        pytest.param(core_defs.CUPY_DEVICE_TYPE, marks=pytest.mark.requires_gpu),
    ]
)
def device_type(request) -> gtx.DeviceType:
    if request.param == core_defs.CUPY_DEVICE_TYPE:
        pytest.importorskip("cupy")
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
            "transient_memory_mode": gtx_transformations.TransientMemoryMode.POOL,
            "gpu_block_size": (32, 8, 1),
            "gpu_block_size_2d": (20, 20),
            "optimization_hooks": {
                gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowPost: mock_top_level_dataflow_hook1,
            },
        }
    else:
        optimization_args = {
            "transient_memory_mode": gtx_transformations.TransientMemoryMode.PERSISTENT,
            "blocking_dim": KDim,
            "blocking_size": 10,
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
                transient_memory_mode=optimization_args["transient_memory_mode"],
                gpu_block_size=optimization_args["gpu_block_size"],
                gpu_block_size_2d=optimization_args["gpu_block_size_2d"],
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
                transient_memory_mode=optimization_args["transient_memory_mode"],
                blocking_dim=optimization_args["blocking_dim"],
                blocking_size=optimization_args["blocking_size"],
                optimization_hooks=optimization_args["optimization_hooks"],
                unit_strides_kind=None,
            )
            mock_top_level_dataflow_hook1.assert_not_called()
            mock_top_level_dataflow_hook2.assert_called_once_with(sdfg)
    else:
        mock_auto_optimize.assert_not_called()
        mock_top_level_dataflow_hook1.assert_not_called()
        mock_top_level_dataflow_hook2.assert_not_called()


def test_make_backend_accepts_external_allocator_with_external_mode():
    external_memory_allocator = lambda size, storage: bytearray(size)

    backend = dace_wf_backend.make_dace_backend(
        gpu=False,
        auto_optimize=True,
        async_sdfg_call=False,
        optimization_args={
            "transient_memory_mode": gtx_transformations.TransientMemoryMode.EXTERNAL,
        },
        external_memory_allocator=external_memory_allocator,
    )

    assert backend.executor.compilation.external_memory_allocator is external_memory_allocator


def test_make_backend_infers_external_mode_when_allocator_is_provided():
    external_memory_allocator = lambda size, storage: bytearray(size)

    backend = dace_wf_backend.make_dace_backend(
        gpu=False,
        auto_optimize=True,
        async_sdfg_call=False,
        external_memory_allocator=external_memory_allocator,
    )

    assert (
        backend.executor.translation.step.auto_optimize_args["transient_memory_mode"]
        == gtx_transformations.TransientMemoryMode.EXTERNAL
    )
    assert backend.executor.compilation.external_memory_allocator is external_memory_allocator


def test_make_backend_warns_external_allocator_without_external_mode():
    external_memory_allocator = lambda size, storage: bytearray(size)

    with pytest.warns(UserWarning, match="External memory allocator provided"):
        backend = dace_wf_backend.make_dace_backend(
            gpu=False,
            auto_optimize=True,
            async_sdfg_call=False,
            optimization_args={
                "transient_memory_mode": gtx_transformations.TransientMemoryMode.POOL,
            },
            external_memory_allocator=external_memory_allocator,
        )

    # Explicit mode stays as requested by the caller; backend only warns.
    assert (
        backend.executor.translation.step.auto_optimize_args["transient_memory_mode"]
        == gtx_transformations.TransientMemoryMode.POOL
    )
    assert backend.executor.compilation.external_memory_allocator is external_memory_allocator


def _parse_generated_code_from_sdfg(sdfg: dace.SDFG, gpu_api_prefix: str) -> str:
    # Helper function to ignore the GPU device initialization code in the generated
    # cuda code, which is not relevant to the test.
    malloc_re = re.compile(
        rf"DACE_GPU_CHECK\(\s*{gpu_api_prefix}Malloc\(\s*\(void \*\*\)\s*&dev_X\s*,\s*1\s*\)\s*\)\s*;"
    )
    free_re = re.compile(rf"DACE_GPU_CHECK\(\s*{gpu_api_prefix}Free\(\s*dev_X\s*\)\s*\)\s*;")

    generated_code = ""
    for code in sdfg.generate_code():
        if code.language == "cu":
            clean_code_iter = iter(code.clean_code.splitlines())
            for line in clean_code_iter:
                if malloc_re.match(line.strip()):
                    line = next(clean_code_iter)  # not relevant to this test
                    assert free_re.match(line.strip())
                else:
                    generated_code += line + "\n"
        elif not code.name.endswith("_main"):
            generated_code += code.clean_code + "\n"

    return generated_code


@pytest.mark.parametrize("transient_memory_mode", list(gtx_transformations.TransientMemoryMode))
def test_transient_memory_mode(device_type, transient_memory_mode, monkeypatch):
    on_gpu = device_type == core_defs.CUPY_DEVICE_TYPE
    gpu_api_prefix = "hip" if core_defs.CUPY_DEVICE_TYPE == core_defs.DeviceType.ROCM else "cuda"
    gpu_malloc_marker = f"{gpu_api_prefix}Malloc("
    gpu_malloc_async_marker = f"{gpu_api_prefix}MallocAsync("
    gpu_free_marker = f"{gpu_api_prefix}Free("
    gpu_free_async_marker = f"{gpu_api_prefix}FreeAsync("
    workspace_requests: list[tuple[int, core_defs.DeviceType]] = []

    def external_memory_allocator(required_nbytes: int, device: core_defs.DeviceType):
        workspace_requests.append((required_nbytes, device))
        if device == core_defs.CUPY_DEVICE_TYPE:
            import cupy as cp
            return cp.empty((required_nbytes,), dtype=cp.uint8)
        return np.empty((required_nbytes,), dtype=np.uint8)

    @gtx.field_operator
    def testee_op(a: cases.IField, b: cases.IField) -> cases.IField:
        tmp = a + b
        return tmp + 1

    @gtx.program
    def testee(a: cases.IField, b: cases.IField, out: cases.IField):
        testee_op(a, b, out=out)

    captured_sdfg: dace.SDFG | None = None
    gt_auto_optimize = gtx_transformations.gt_auto_optimize

    def mocked_auto_optimize(*args, **kwargs) -> dace.SDFG:
        nonlocal captured_sdfg
        result = gt_auto_optimize(*args, **kwargs)
        captured_sdfg = result
        return result

    def no_op_top_level_map_processing(*, sdfg: dace.SDFG, **kwargs) -> dace.SDFG:
        return sdfg  # we need to keep the intermediate transient array

    monkeypatch.setattr(gtx_transformations, "gt_auto_optimize", mocked_auto_optimize)
    monkeypatch.setattr(
        gtx_auto_optimize,
        "_gt_auto_process_top_level_maps",
        no_op_top_level_map_processing,
    )

    custom_backend = dace_wf_backend.make_dace_backend(
        gpu=on_gpu,
        auto_optimize=True,
        async_sdfg_call=False,
        optimization_args={
            "transient_memory_mode": transient_memory_mode,
        },
        external_memory_allocator=external_memory_allocator,
    )

    test_case = cases.Case.from_cartesian_grid_descriptor(
        cases_utils.simple_cartesian_grid(),
        backend=custom_backend,
        allocator=custom_backend,
    )
    a = cases.allocate(test_case, testee, "a", strategy=cases.UniqueInitializer())()
    b = cases.allocate(test_case, testee, "b", strategy=cases.UniqueInitializer())()
    out = cases.allocate(test_case, testee, "out")()

    program = (
        testee.with_grid_type(gtx.common.GridType.CARTESIAN)
        .with_backend(custom_backend)
        .compile(offset_provider={})
    )
    gtx.wait_for_compilation()
    program(a, b, out=out, offset_provider={})

    assert captured_sdfg is not None
    generated_code = _parse_generated_code_from_sdfg(captured_sdfg, gpu_api_prefix)

    match transient_memory_mode:
        case gtx_transformations.TransientMemoryMode.EXTERNAL:
            # External mode wires explicit workspace API calls in generated host code.
            assert "set_external_memory" in generated_code
            assert "__dace_get_external_memory_size_" in generated_code
            if on_gpu:
                # Workspace must come from the external allocator, not from runtime GPU alloc/free.
                assert not any(
                    marker in generated_code
                    for marker in (gpu_malloc_marker, gpu_malloc_async_marker)
                )
                assert not any(
                    marker in generated_code for marker in (gpu_free_marker, gpu_free_async_marker)
                )
                expected_device = core_defs.CUPY_DEVICE_TYPE
            else:
                # CPU external mode should route transient workspace setup via
                # external-memory API calls rather than host malloc/free calls.
                assert any(marker in generated_code for marker in ("new ", "malloc"))
                assert any(marker in generated_code for marker in ("delete ", "free"))
                expected_device = core_defs.DeviceType.CPU

            assert workspace_requests
            assert all(device == expected_device for _, device in workspace_requests)

        case gtx_transformations.TransientMemoryMode.POOL:
            assert "set_external_memory" not in generated_code
            assert "__dace_get_external_memory_size_" not in generated_code
            assert not workspace_requests
            if on_gpu:
                # Pool mode on GPU should rely on pooled/async allocation APIs.
                assert all(
                    marker in generated_code
                    for marker in (gpu_malloc_async_marker, gpu_free_async_marker)
                )
                assert not any(
                    marker in generated_code for marker in (gpu_malloc_marker, gpu_free_marker)
                )
            else:
                # On CPU, pool mode behaves as regular scoped lifetime.
                assert any(marker in generated_code for marker in ("new ", "malloc"))
                assert any(marker in generated_code for marker in ("delete ", "free"))

        case (
            gtx_transformations.TransientMemoryMode.PERSISTENT,
            gtx_transformations.TransientMemoryMode.SCOPED,
        ):
            # `PERSISTENT` and `SCOPED` mode use the same memory APIs, but in different contexts.
            assert "set_external_memory" not in generated_code
            assert "__dace_get_external_memory_size_" not in generated_code
            assert not workspace_requests
            if on_gpu:
                # Persistent and scoped mode on GPU should rely on sync allocation APIs.
                assert all(
                    marker in generated_code for marker in (gpu_malloc_marker, gpu_free_marker)
                )
                assert not any(
                    marker in generated_code
                    for marker in (gpu_malloc_async_marker, gpu_free_async_marker)
                )
            else:
                assert any(marker in generated_code for marker in ("new ", "malloc"))
                assert any(marker in generated_code for marker in ("delete ", "free"))

    assert np.allclose(out.asnumpy(), a.asnumpy() + b.asnumpy() + 1)
