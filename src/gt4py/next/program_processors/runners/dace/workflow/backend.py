# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gt4py.next.custom_layout_allocators as next_allocators
from gt4py._core import definitions as core_defs
from gt4py.next import backend
from gt4py.next.otf import definitions, stages, workflow
from gt4py.next.program_processors.runners.dace.workflow.factory import make_dace_workflow
from gt4py.next.program_processors.runners.dace.workflow.translation import make_dace_translator


def make_dace_backend(
    gpu: bool,
    *,
    cached: bool = True,
    auto_optimize: bool = True,
    name_postfix: str = "",
    translation: definitions.TranslationStep | None = None,
    executor: workflow.Workflow[definitions.CompilableProgramDef, stages.ExecutableProgram]
    | None = None,
) -> backend.Backend:
    """Build the GTIR-DaCe backend for the given device and configuration.

    Cross-cutting configuration is passed as keyword arguments. To customize the
    translator-local options (async SDFG call, metrics, zero-origin, ...), inject a
    pre-built translator via ``translation=`` (see `make_dace_translator`); its
    ``device_type`` and ``auto_optimize`` are set to match ``gpu``/``auto_optimize``.
    Pass ``executor=`` to replace the whole executor workflow.

    Args:
        gpu: Enable GPU transformations and code generation.
        cached: Cache the lowered SDFG as a JSON file and the compiled programs.
        auto_optimize: Enable the SDFG auto-optimize pipeline.

    Returns:
        A dace backend for the target device.
    """
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

    if executor is None:
        otf_workflow = make_dace_workflow(
            device_type=device_type,
            auto_optimize=auto_optimize,
            cached_translation=cached,
            translation=translation,
        )
        executor = (
            workflow.CachedStep.in_memory(
                otf_workflow, input_fingerprinter=stages.fast_compilable_program_fingerprinter
            )
            if cached
            else otf_workflow
        )

    name_cached = "_cached" if cached else ""
    name_opt = "_opt" if auto_optimize else ""
    return backend.Backend(
        name=f"run_dace_{name_device}{name_cached}{name_opt}{name_postfix}",
        executor=executor,
        allocator=allocator,
        transforms=backend.DEFAULT_TRANSFORMS,
    )


run_dace_cpu = make_dace_backend(gpu=False, cached=False, auto_optimize=True)
run_dace_cpu_noopt = make_dace_backend(gpu=False, cached=False, auto_optimize=False)
run_dace_cpu_cached = make_dace_backend(gpu=False, cached=True, auto_optimize=True)

run_dace_gpu = make_dace_backend(
    gpu=True,
    cached=False,
    auto_optimize=True,
    translation=make_dace_translator(async_sdfg_call=True),
)
run_dace_gpu_noopt = make_dace_backend(
    gpu=True,
    cached=False,
    auto_optimize=False,
    translation=make_dace_translator(async_sdfg_call=True),
)
run_dace_gpu_cached = make_dace_backend(
    gpu=True,
    cached=True,
    auto_optimize=True,
    translation=make_dace_translator(async_sdfg_call=True),
)
