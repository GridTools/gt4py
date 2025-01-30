# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools

import factory

from gt4py._core import definitions as core_defs
from gt4py.next import config
from gt4py.next.otf import recipes, stages
from gt4py.next.program_processors.runners.dace.workflow import decoration as decoration_step
from gt4py.next.program_processors.runners.dace.workflow.compilation import (
    DaCeCompilationStepFactory,
)
from gt4py.next.program_processors.runners.dace.workflow.translation import (
    DaCeTranslationStepFactory,
)


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
        auto_optimize: bool = False

    translation = factory.SubFactory(
        DaCeTranslationStepFactory,
        device_type=factory.SelfAttribute("..device_type"),
        auto_optimize=factory.SelfAttribute("..auto_optimize"),
    )
    bindings = _no_bindings
    compilation = factory.SubFactory(
        DaCeCompilationStepFactory,
        cache_lifetime=factory.LazyFunction(lambda: config.BUILD_CACHE_LIFETIME),
        cmake_build_type=factory.SelfAttribute("..cmake_build_type"),
    )
    decoration = factory.LazyAttribute(
        lambda o: functools.partial(
            decoration_step.convert_args,
            device=o.device_type,
        )
    )
