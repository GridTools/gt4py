# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import factory

import gt4py._core.definitions as core_defs
import gt4py.next.custom_layout_allocators as next_allocators
from gt4py._core import filecache
from gt4py.next import backend, config
from gt4py.next.otf import recipes, stages, workflow
from gt4py.next.otf.binding import nanobind
from gt4py.next.otf.compilation import compiler
from gt4py.next.otf.compilation.build_systems import compiledb
from gt4py.next.program_processors.codegens.gtfn import gtfn_module
from gt4py.next.program_processors.runners import gtfn_decoration


class GTFNBuildWorkflowFactory(factory.Factory):
    class Meta:
        model = recipes.OTFBuildWorkflow

    class Params:
        device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
        cmake_build_type: config.CMakeBuildType = factory.LazyFunction(  # type: ignore[assignment] # factory-boy typing not precise enough
            lambda: config.CMAKE_BUILD_TYPE
        )
        builder_factory: compiler.BuildSystemProjectGenerator = factory.LazyAttribute(  # type: ignore[assignment] # factory-boy typing not precise enough
            lambda o: compiledb.CompiledbFactory(cmake_build_type=o.cmake_build_type)
        )

        cached_translation = factory.Trait(
            translation=factory.LazyAttribute(
                lambda o: workflow.CachedStep(
                    o.bare_translation,
                    hash_function=stages.fingerprint_compilable_program,
                    cache=filecache.FileCache(str(config.BUILD_CACHE_DIR / "gtfn_cache")),
                )
            ),
        )

        bare_translation = factory.SubFactory(
            gtfn_module.GTFNTranslationStepFactory,
            device_type=factory.SelfAttribute("..device_type"),
        )

    translation = factory.LazyAttribute(lambda o: o.bare_translation)

    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableProject] = (
        nanobind.bind_source
    )
    compilation = factory.SubFactory(
        compiler.CompilerFactory,
        cache_lifetime=factory.LazyFunction(lambda: config.BUILD_CACHE_LIFETIME),
        builder_factory=factory.SelfAttribute("..builder_factory"),
        device_type=factory.SelfAttribute("..device_type"),
        decorator=gtfn_decoration.convert_args,
    )


class GTFNBackendFactory(factory.Factory):
    class Meta:
        model = backend.Backend

    class Params:
        name_device = "cpu"
        name_cached = ""
        name_temps = ""
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
            GTFNBuildWorkflowFactory, device_type=factory.SelfAttribute("..device_type")
        )

    name = factory.LazyAttribute(
        lambda o: f"run_gtfn_{o.name_device}{o.name_temps}{o.name_cached}{o.name_postfix}"
    )

    executor = factory.LazyAttribute(lambda o: o.otf_workflow)
    allocator = next_allocators.StandardCPUFieldBufferAllocator()
    transforms = backend.DEFAULT_TRANSFORMS


run_gtfn = GTFNBackendFactory()

run_gtfn_imperative = GTFNBackendFactory(
    name_postfix="_imperative", otf_workflow__translation__use_imperative_backend=True
)

run_gtfn_cached = GTFNBackendFactory(cached=True, otf_workflow__cached_translation=True)

run_gtfn_gpu = GTFNBackendFactory(gpu=True)

run_gtfn_gpu_cached = GTFNBackendFactory(
    gpu=True, cached=True, otf_workflow__cached_translation=True
)

run_gtfn_no_transforms = GTFNBackendFactory(
    otf_workflow__bare_translation__enable_itir_transforms=False
)
