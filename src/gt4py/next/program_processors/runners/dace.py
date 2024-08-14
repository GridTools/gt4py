# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools

import factory

import gt4py._core.definitions as core_defs
from gt4py.next import config
from gt4py.next.otf import recipes, stages
from gt4py.next.program_processors.runners.dace_iterator.workflow import (
    DaCeCompilationStepFactory,
    DaCeTranslationStepFactory,
    convert_args,
)
from gt4py.next.program_processors.runners.gtfn import GTFNBackendFactory


def _no_bindings(inp: stages.ProgramSource) -> stages.CompilableSource:
    return stages.CompilableSource(program_source=inp, binding_source=None)


class DaCeWorkflowFactory(factory.Factory):
    class Meta:
        model = recipes.OTFCompileWorkflow

    class Params:
        device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
        cmake_build_type: config.CMakeBuildType = factory.LazyFunction(
            lambda: config.CMAKE_BUILD_TYPE
        )
        use_field_canonical_representation: bool = False

    translation = factory.SubFactory(
        DaCeTranslationStepFactory,
        device_type=factory.SelfAttribute("..device_type"),
        use_field_canonical_representation=factory.SelfAttribute(
            "..use_field_canonical_representation"
        ),
    )
    bindings = _no_bindings
    compilation = factory.SubFactory(
        DaCeCompilationStepFactory,
        cache_lifetime=factory.LazyFunction(lambda: config.BUILD_CACHE_LIFETIME),
        cmake_build_type=factory.SelfAttribute("..cmake_build_type"),
    )
    decoration = factory.LazyAttribute(
        lambda o: functools.partial(
            convert_args,
            device=o.device_type,
            use_field_canonical_representation=o.use_field_canonical_representation,
        )
    )


class DaCeBackendFactory(GTFNBackendFactory):
    class Params:
        otf_workflow = factory.SubFactory(
            DaCeWorkflowFactory,
            device_type=factory.SelfAttribute("..device_type"),
            use_field_canonical_representation=factory.SelfAttribute(
                "..use_field_canonical_representation"
            ),
        )
        name = factory.LazyAttribute(
            lambda o: f"run_dace_{o.name_device}{o.name_temps}{o.name_cached}{o.name_postfix}"
        )
        auto_optimize = factory.Trait(
            otf_workflow__translation__auto_optimize=True, name_temps="_opt"
        )
        use_field_canonical_representation: bool = False


run_dace_cpu = DaCeBackendFactory(cached=True, auto_optimize=True)
run_dace_cpu_with_temporaries = DaCeBackendFactory(
    cached=True, auto_optimize=True, use_temporaries=True
)
run_dace_cpu_noopt = DaCeBackendFactory(cached=True, auto_optimize=False)

run_dace_gpu = DaCeBackendFactory(gpu=True, cached=True, auto_optimize=True)
run_dace_gpu_with_temporaries = DaCeBackendFactory(
    gpu=True, cached=True, auto_optimize=True, use_temporaries=True
)
run_dace_gpu_noopt = DaCeBackendFactory(gpu=True, cached=True, auto_optimize=False)
