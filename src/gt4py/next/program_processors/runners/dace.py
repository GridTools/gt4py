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
from typing import cast

import dace
import factory
from dace.codegen.compiled_sdfg import CompiledSDFG

import gt4py._core.definitions as core_defs
from gt4py.next import common, config
from gt4py.next.otf import recipes, stages
from gt4py.next.program_processors.runners.dace_iterator import (
    get_sdfg_args,
    workflow as dace_workflow,
)
from gt4py.next.program_processors.runners.gtfn import GTFNBackendFactory


class DaCeTranslationStepFactory(factory.Factory):
    class Meta:
        model = dace_workflow.DaCeTranslator


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = dace_workflow.DaCeCompiler


def _convert_args(
    inp: stages.CompiledProgram, device: core_defs.DeviceType = core_defs.DeviceType.CPU
) -> stages.CompiledProgram:
    sdfg_program = cast(CompiledSDFG, inp)
    on_gpu = True if device == core_defs.DeviceType.CUDA else False
    sdfg = sdfg_program.sdfg

    def decorated_program(
        *args, offset_provider: dict[str, common.Connectivity | common.Dimension]
    ):
        sdfg_args = get_sdfg_args(
            sdfg, *args, check_args=False, offset_provider=offset_provider, on_gpu=on_gpu
        )

        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "allow_view_arguments", value=True)
            return inp(**sdfg_args)

    return decorated_program


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

    translation = factory.SubFactory(
        DaCeTranslationStepFactory, device_type=factory.SelfAttribute("..device_type")
    )
    bindings = _no_bindings
    compilation = factory.SubFactory(
        DaCeCompilationStepFactory,
        cache_lifetime=factory.LazyFunction(lambda: config.BUILD_CACHE_LIFETIME),
        cmake_build_type=factory.SelfAttribute("..cmake_build_type"),
    )
    decoration = factory.LazyAttribute(
        lambda o: functools.partial(_convert_args, device=o.device_type)
    )


class DaCeBackendFactory(GTFNBackendFactory):
    class Params:
        otf_workflow = factory.SubFactory(
            DaCeWorkflowFactory, device_type=factory.SelfAttribute("..device_type")
        )
        name = factory.LazyAttribute(
            lambda o: f"run_dace_{o.name_device}{o.name_temps}{o.name_cached}{o.name_postfix}"
        )


run_dace_cpu = DaCeBackendFactory(cached=True)

run_dace_gpu = DaCeBackendFactory(gpu=True, cached=True)
