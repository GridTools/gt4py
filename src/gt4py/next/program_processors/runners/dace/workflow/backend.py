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
from gt4py.next import backend, common
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


def make_dace_backend(
    auto_optimize: bool,
    cached: bool,
    gpu: bool,
    async_sdfg_call: bool,
    make_persistent: bool,
    blocking_dim: common.Dimension | None,
    blocking_size: int = 10,
    disable_field_origin: bool = False,
) -> backend.Backend:
    """Helper function to create a dace cached backend with custom config for SDFG
    lowering and auto-optimize.

    Args:
        auto_optimize: Enable SDFG auto-optimize pipeline.
        cached: Cache the lowered SDFG as a JSON file and the compiled programs.
        gpu: Enable GPU transformations and code generation.
        async_sdfg_call: Enable asynchronous SDFG execution, only applicable
            when `gpu=True` because it relies on the gpu kernel queue.
        make_persistent: Allocate persistent arrays with constant layout.
        blocking_dim: When not 'None', apply 'LoopBlocking' SDFG transformation
            on this dimension.
        blocking_size: Block size to use in 'LoopBlocking' SDFG transformation,
            when enabled.
        disable_field_origin: Can be set to `True` when all fields passed as program
            arguments have zero-based origin.

    Returns:
        A custom dace backend object.
    """
    return DaCeBackendFactory(
        gpu=gpu,
        auto_optimize=auto_optimize,
        cached=cached,
        otf_workflow__cached_translation=cached,
        otf_workflow__bare_translation__async_sdfg_call=(gpu if async_sdfg_call else False),
        otf_workflow__bare_translation__make_persistent=make_persistent,
        otf_workflow__bare_translation__blocking_dim=blocking_dim,
        otf_workflow__bare_translation__blocking_size=blocking_size,
        otf_workflow__bare_translation__disable_field_origin_on_program_arguments=disable_field_origin,
        otf_workflow__bindings__make_persistent=make_persistent,
    )


run_dace_cpu = make_dace_backend(
    auto_optimize=True,
    cached=False,
    gpu=False,
    async_sdfg_call=False,
    make_persistent=False,
    blocking_dim=None,
)
run_dace_cpu_noopt = make_dace_backend(
    auto_optimize=False,
    cached=False,
    gpu=False,
    async_sdfg_call=False,
    make_persistent=False,
    blocking_dim=None,
)
run_dace_cpu_cached = make_dace_backend(
    auto_optimize=True,
    cached=True,
    gpu=False,
    async_sdfg_call=False,
    make_persistent=False,
    blocking_dim=None,
)

run_dace_gpu = make_dace_backend(
    auto_optimize=True,
    cached=False,
    gpu=True,
    async_sdfg_call=False,
    make_persistent=False,
    blocking_dim=None,
)
run_dace_gpu_noopt = make_dace_backend(
    auto_optimize=False,
    cached=False,
    gpu=True,
    async_sdfg_call=False,
    make_persistent=False,
    blocking_dim=None,
)
run_dace_gpu_cached = make_dace_backend(
    auto_optimize=True,
    cached=True,
    gpu=True,
    async_sdfg_call=False,
    make_persistent=False,
    blocking_dim=None,
)
