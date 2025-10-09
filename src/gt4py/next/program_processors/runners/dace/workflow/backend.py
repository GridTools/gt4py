# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import factory

import gt4py.next.allocators as next_allocators
from gt4py._core import definitions as core_defs
from gt4py.next import backend
from gt4py.next.otf import stages, workflow
from gt4py.next.program_processors.runners.dace.workflow.factory import DaCeWorkflowFactory


class DaCeBackendFactory(factory.Factory):
    """
    Workflow factory for the GTIR-DaCe backend.

    Several parameters are inherithed from `backend.Backend`, see below the specific ones.

    Args:
        auto_optimize: Enables the SDFG transformation pipeline.
        make_persistent: Enables optimization in SDFG lowering and bindings generation
            assuming that the layout of temporary and global fields does not change
            across multiple SDFG calls.
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
            make_persistent=factory.SelfAttribute("..make_persistent"),
        )
        auto_optimize = factory.Trait(name_postfix="_opt")
        make_persistent = False

    name = factory.LazyAttribute(
        lambda o: f"run_dace_{o.name_device}{o.name_cached}{o.name_postfix}"
    )

    executor = factory.LazyAttribute(lambda o: o.otf_workflow)
    allocator = next_allocators.StandardCPUFieldBufferAllocator()
    transforms = backend.DEFAULT_TRANSFORMS


def _make_dace_backend(
    gpu: bool,
    cached: bool,
    auto_optimize: bool,
    async_sdfg_call: bool,
    use_memory_pool: bool,
) -> backend.Backend:
    """Helper function to create a dace cached backend with custom config for SDFG
    lowering and auto-optimize.

    Note that `async_sdfg_call=True` on GPU device relies on the dace configuration
    with 1 cuda stream, so that all kernels and memory operations are executed
    sequentially on the default stream.

    Args:
        gpu: Enable GPU transformations and code generation.
        cached: Cache the lowered SDFG as a JSON file and the compiled programs.
        auto_optimize: Enable the SDFG auto-optimize pipeline.
        async_sdfg_call: Enable asynchronous SDFG execution, only applicable
            when `gpu=True` because it relies on the gpu kernel queue.
        use_memory_pool: Enable memory pool for allocation of temporary fields.

    Returns:
        A custom dace backend object.
    """
    return DaCeBackendFactory(  # type: ignore[return-value] # factory-boy typing not precise enough
        gpu=gpu,
        cached=cached,
        auto_optimize=auto_optimize,
        otf_workflow__bare_translation__blocking_dim=None,
        otf_workflow__bare_translation__async_sdfg_call=(async_sdfg_call if gpu else False),
        otf_workflow__bare_translation__disable_field_origin_on_program_arguments=False,
        otf_workflow__bare_translation__make_persistent=False,
        otf_workflow__bare_translation__optimization_hooks=None,
        otf_workflow__bare_translation__use_memory_pool=use_memory_pool,
        otf_workflow__bare_translation__use_metrics=True,
        otf_workflow__bindings__make_persistent=False,
    )


run_dace_cpu = _make_dace_backend(
    gpu=False,
    cached=False,
    auto_optimize=True,
    async_sdfg_call=False,
    use_memory_pool=False,
)
run_dace_cpu_noopt = _make_dace_backend(
    gpu=False,
    cached=False,
    auto_optimize=False,
    async_sdfg_call=False,
    use_memory_pool=False,
)
run_dace_cpu_cached = _make_dace_backend(
    gpu=False,
    cached=True,
    auto_optimize=True,
    async_sdfg_call=False,
    use_memory_pool=False,
)

run_dace_gpu = _make_dace_backend(
    gpu=True,
    cached=False,
    auto_optimize=True,
    async_sdfg_call=True,
    use_memory_pool=(core_defs.CUPY_DEVICE_TYPE == core_defs.DeviceType.CUDA),
)
run_dace_gpu_noopt = _make_dace_backend(
    gpu=True,
    cached=False,
    auto_optimize=False,
    async_sdfg_call=True,
    use_memory_pool=(core_defs.CUPY_DEVICE_TYPE == core_defs.DeviceType.CUDA),
)
run_dace_gpu_cached = _make_dace_backend(
    gpu=True,
    cached=True,
    auto_optimize=True,
    async_sdfg_call=True,
    use_memory_pool=(core_defs.CUPY_DEVICE_TYPE == core_defs.DeviceType.CUDA),
)
