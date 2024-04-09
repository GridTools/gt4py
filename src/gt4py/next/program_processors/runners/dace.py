# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

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

run_dace_gpu = DaCeBackendFactory(gpu=True, cached=True, auto_optimize=True)
run_dace_gpu_with_temporaries = DaCeBackendFactory(
    gpu=True, cached=True, auto_optimize=True, use_temporaries=True
)
