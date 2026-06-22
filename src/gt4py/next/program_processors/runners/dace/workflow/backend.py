# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import Any, Final

import gt4py.next.custom_layout_allocators as next_allocators
from gt4py._core import definitions as core_defs
from gt4py.next import backend, common, config
from gt4py.next.otf import stages, workflow
from gt4py.next.program_processors.runners.dace.workflow.factory import make_dace_workflow


def make_dace_backend(
    gpu: bool,
    cached: bool = True,
    auto_optimize: bool = True,
    async_sdfg_call: bool = True,
    optimization_args: dict[str, Any] | None = None,
    unstructured_horizontal_has_unit_stride: bool = config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE,
    use_metrics: bool = True,
    use_zero_origin: bool = False,
    use_max_domain_range_on_unstructured_shift: bool | None = None,
) -> backend.Backend:
    """Customize the dace backend with the given configuration parameters.

    Args:
        gpu: Enable GPU transformations and code generation.
        cached: Cache the lowered SDFG as a JSON file and the compiled programs.
        auto_optimize: Enable the SDFG auto-optimize pipeline.
        async_sdfg_call: Make an asynchronous SDFG call on GPU to allow overlapping
            of GPU kernel execution with the Python driver code.
        optimization_args: A `dict` containing configuration parameters for
            the SDFG auto-optimize pipeline, see `gt_auto_optimize()`.
        unstructured_horizontal_has_unit_stride: When the memory layout has unit stride
            in the horizontal dimension, replace the field stride symbol with '1'.
        use_metrics: Add SDFG instrumentation to collect the metric for stencil
            compute time.
        use_zero_origin: Can be set to `True` when all fields passed as program
            arguments have zero-based origin. This setting will skip generation
            of range start-symbols `_range_0` since they can be assumed to be zero.

    Note that `gt_auto_optimize()` parameters that are derived from GT4Py configuration
    cannot be overriden, and therefore cannot appear here. Thus, this function will
    throw an exception if called with any argument included in `gt_optimization_args`.

    Returns:
        A dace backend with custom configuration for the target device.
    """

    # The `gt_optimization_args` set contains the parameters of `gt_auto_optimize()`
    # that are derived from the gt4py configuration, and therefore cannot be customized.
    gt_optimization_args: Final[set[str]] = {"gpu", "constant_symbols", "unit_strides_kind"}

    if optimization_args is None:
        optimization_args = {}
    elif optimization_args and not auto_optimize:
        warnings.warn("Optimizations args given, but auto-optimize is disabled.", stacklevel=2)
    elif intersect_args := gt_optimization_args.intersection(optimization_args.keys()):
        raise ValueError(
            f"The following optimization arguments cannot be overriden: {intersect_args}."
        )

    # Set `unit_strides_kind` based on the gt4py env configuration.
    optimization_args = optimization_args | {
        "unit_strides_kind": common.DimensionKind.HORIZONTAL
        if unstructured_horizontal_has_unit_stride
        else None
    }

    allocator: next_allocators.FieldBufferAllocatorProtocol
    device_type: core_defs.DeviceType
    if gpu:
        allocator = next_allocators.StandardGPUFieldBufferAllocator()
        device_type = core_defs.CUPY_DEVICE_TYPE or core_defs.DeviceType.CUDA
        name_device = "gpu"
    else:
        allocator = next_allocators.StandardCPUFieldBufferAllocator()
        device_type = core_defs.DeviceType.CPU
        name_device = "cpu"

    otf_workflow = make_dace_workflow(
        device_type=device_type,
        auto_optimize=auto_optimize,
        cached_translation=cached,
        async_sdfg_call=(async_sdfg_call if gpu else False),
        auto_optimize_args=optimization_args,
        unstructured_horizontal_has_unit_stride=unstructured_horizontal_has_unit_stride,
        use_metrics=use_metrics,
        disable_field_origin_on_program_arguments=use_zero_origin,
        use_max_domain_range_on_unstructured_shift=use_max_domain_range_on_unstructured_shift,
    )

    executor = (
        workflow.CachedStep.in_memory(
            otf_workflow, input_fingerprinter=stages.fast_compilable_program_fingerprinter
        )
        if cached
        else otf_workflow
    )

    name_cached = "_cached" if cached else ""
    name_postfix = "_opt" if auto_optimize else ""
    return backend.Backend(
        name=f"run_dace_{name_device}{name_cached}{name_postfix}",
        executor=executor,
        allocator=allocator,
        transforms=backend.DEFAULT_TRANSFORMS,
    )


run_dace_cpu = make_dace_backend(
    gpu=False,
    cached=False,
    auto_optimize=True,
    async_sdfg_call=False,
)
run_dace_cpu_noopt = make_dace_backend(
    gpu=False,
    cached=False,
    auto_optimize=False,
    async_sdfg_call=False,
)
run_dace_cpu_cached = make_dace_backend(
    gpu=False,
    cached=True,
    auto_optimize=True,
    async_sdfg_call=False,
)

run_dace_gpu = make_dace_backend(
    gpu=True,
    cached=False,
    auto_optimize=True,
    async_sdfg_call=True,
)
run_dace_gpu_noopt = make_dace_backend(
    gpu=True,
    cached=False,
    auto_optimize=False,
    async_sdfg_call=True,
)
run_dace_gpu_cached = make_dace_backend(
    gpu=True,
    cached=True,
    auto_optimize=True,
    async_sdfg_call=True,
)
