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

import factory

import gt4py.next.allocators as next_allocators
from gt4py._core import definitions as core_defs
from gt4py.next import backend, common, config
from gt4py.next.otf import stages, workflow
from gt4py.next.program_processors.runners.dace.workflow.factory import DaCeWorkflowFactory


class DaCeBackendFactory(factory.Factory):
    """
    Workflow factory for the GTIR-DaCe backend.

    Several parameters are inherithed from `backend.Backend`, see below the specific ones.

    Args:
        auto_optimize: Enables the SDFG transformation pipeline.
    """

    class Meta:
        model = backend.Backend

    class Params:
        name_device = "cpu"
        name_cached = ""
        name_postfix = ""
        gpu = factory.Trait(
            allocator=next_allocators.StandardGPUFieldBufferAllocator(),
            device_type=core_defs.CUPY_DEVICE_TYPE or core_defs.DeviceType.CUDA,
            name_device="gpu",
        )
        cached = factory.Trait(
            executor=factory.LazyAttribute(
                lambda o: workflow.CachedStep(o.otf_workflow, hash_function=o.hash_function)
            ),
            name_cached="_cached",
        )
        device_type = core_defs.DeviceType.CPU
        hash_function = stages.compilation_hash
        otf_workflow = factory.SubFactory(
            DaCeWorkflowFactory,
            device_type=factory.SelfAttribute("..device_type"),
            auto_optimize=factory.SelfAttribute("..auto_optimize"),
        )
        auto_optimize = factory.Trait(name_postfix="_opt")

    name = factory.LazyAttribute(
        lambda o: f"run_dace_{o.name_device}{o.name_cached}{o.name_postfix}"
    )

    executor = factory.LazyAttribute(lambda o: o.otf_workflow)
    allocator = next_allocators.StandardCPUFieldBufferAllocator()
    transforms = backend.DEFAULT_TRANSFORMS


def make_dace_backend(
    gpu: bool,
    cached: bool = True,
    auto_optimize: bool = True,
    async_sdfg_call: bool = True,
    optimization_args: dict[str, Any] | None = None,
    use_metrics: bool = True,
    use_zero_origin: bool = False,
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
        if config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE
        else None
    }

    if gpu and core_defs.CUPY_DEVICE_TYPE == core_defs.DeviceType.ROCM:
        # Backend supports gpu memory pool only on CUDA target.
        optimization_args["gpu_memory_pool"] = False

    return DaCeBackendFactory(  # type: ignore[return-value] # factory-boy typing not precise enough
        gpu=gpu,
        cached=cached,
        auto_optimize=auto_optimize,
        otf_workflow__cached_translation=cached,
        otf_workflow__bare_translation__async_sdfg_call=(async_sdfg_call if gpu else False),
        otf_workflow__bare_translation__auto_optimize_args=optimization_args,
        otf_workflow__bare_translation__use_metrics=use_metrics,
        otf_workflow__bare_translation__disable_field_origin_on_program_arguments=use_zero_origin,
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
