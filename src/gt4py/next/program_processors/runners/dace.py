# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import factory

import gt4py._core.definitions as core_defs
import gt4py.next.allocators as next_allocators
from gt4py.next import backend
from gt4py.next.otf import workflow
from gt4py.next.program_processors.runners.dace_fieldview import workflow as dace_fieldview_workflow
from gt4py.next.program_processors.runners.gtfn import GTFNBackendFactory


class DaCeFieldviewBackendFactory(GTFNBackendFactory):
    class Meta:
        model = backend.Backend

    class Params:
        name_device = "cpu"
        name_cached = ""
        name_postfix = ""
        gpu = factory.Trait(
            allocator=next_allocators.StandardGPUFieldBufferAllocator(),
            device_type=next_allocators.CUPY_DEVICE or core_defs.DeviceType.CUDA,
            name_device="gpu",
        )
        cached = factory.Trait(
            executor=factory.LazyAttribute(
                lambda o: workflow.CachedStep(o.otf_workflow, hash_function=o.hash_function)
            ),
            name_cached="_cached",
        )
        device_type = core_defs.DeviceType.CPU
        otf_workflow = factory.SubFactory(
            dace_fieldview_workflow.DaCeWorkflowFactory,
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


run_dace_cpu = DaCeFieldviewBackendFactory(cached=True, auto_optimize=True)
run_dace_cpu_noopt = DaCeFieldviewBackendFactory(cached=True, auto_optimize=False)

run_dace_gpu = DaCeFieldviewBackendFactory(gpu=True, cached=True, auto_optimize=True)
run_dace_gpu_noopt = DaCeFieldviewBackendFactory(gpu=True, cached=True, auto_optimize=False)
