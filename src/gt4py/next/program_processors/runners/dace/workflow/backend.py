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


run_dace_cpu = DaCeBackendFactory(auto_optimize=True)
run_dace_cpu_noopt = DaCeBackendFactory(auto_optimize=False)
run_dace_cpu_cached = DaCeBackendFactory(
    auto_optimize=True, cached=True, otf_workflow__cached_translation=True
)

run_dace_gpu = DaCeBackendFactory(gpu=True, auto_optimize=True)
run_dace_gpu_noopt = DaCeBackendFactory(gpu=True, auto_optimize=False)
run_dace_gpu_cached = DaCeBackendFactory(
    gpu=True, auto_optimize=True, cached=True, otf_workflow__cached_translation=True
)
