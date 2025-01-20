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
from gt4py.eve.utils import content_hash
from gt4py.next import backend
from gt4py.next.otf import stages, workflow
from gt4py.next.program_processors.runners.dace.workflow.factory import DaCeWorkflowFactory


def _compilation_hash(otf_closure: stages.CompilableProgram) -> int:
    """Given closure compute a hash uniquely determining if we need to recompile.

    This function was copied from 'next.program_processors.runners.gtfn'
    """
    offset_provider = otf_closure.args.offset_provider
    return hash(
        (
            otf_closure.data,
            # As the frontend types contain lists they are not hashable. As a workaround we just
            # use content_hash here.
            content_hash(tuple(arg for arg in otf_closure.args.args)),
            # Directly using the `id` of the offset provider is not possible as the decorator adds
            # the implicitly defined ones (i.e. to allow the `TDim + 1` syntax) resulting in a
            # different `id` every time. Instead use the `id` of each individual offset provider.
            tuple((k, id(v)) for (k, v) in offset_provider.items()) if offset_provider else None,
            otf_closure.args.column_axis,
        )
    )


class DaCeFieldviewBackendFactory(factory.Factory):
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
        hash_function = _compilation_hash
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


run_dace_cpu = DaCeFieldviewBackendFactory(cached=True, auto_optimize=True)
run_dace_cpu_noopt = DaCeFieldviewBackendFactory(cached=True, auto_optimize=False)

run_dace_gpu = DaCeFieldviewBackendFactory(gpu=True, cached=True, auto_optimize=True)
run_dace_gpu_noopt = DaCeFieldviewBackendFactory(gpu=True, cached=True, auto_optimize=False)
