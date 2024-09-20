# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
from typing import Optional

import dace
import factory

from gt4py._core import definitions as core_defs
from gt4py.next import common, config
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import languages, recipes, stages, step_types, workflow
from gt4py.next.otf.binding import interface
from gt4py.next.otf.languages import LanguageSettings
from gt4py.next.program_processors.runners.dace_common import workflow as dace_workflow
from gt4py.next.program_processors.runners.dace_fieldview import gtir_to_sdfg
from gt4py.next.type_system import type_translation as tt


@dataclasses.dataclass(frozen=True)
class DaCeTranslator(
    workflow.ChainableWorkflowMixin[
        stages.ProgramCall, stages.ProgramSource[languages.SDFG, languages.LanguageSettings]
    ],
    step_types.TranslationStep[languages.SDFG, languages.LanguageSettings],
):
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU

    def _language_settings(self) -> languages.LanguageSettings:
        return languages.LanguageSettings(
            formatter_key="", formatter_style="", file_extension="sdfg"
        )

    def generate_sdfg(
        self,
        ir: itir.Program,
        offset_provider: dict[str, common.Dimension | common.Connectivity],
        column_axis: Optional[common.Dimension],
    ) -> dace.SDFG:
        # TODO(edopao): Call IR transformations and domain inference, finally lower IR to SDFG
        raise NotImplementedError

        return gtir_to_sdfg.build_sdfg_from_gtir(program=ir, offset_provider=offset_provider)

    def __call__(
        self, inp: stages.ProgramCall
    ) -> stages.ProgramSource[languages.SDFG, LanguageSettings]:
        """Generate DaCe SDFG file from the GTIR definition."""
        program = inp.program
        assert isinstance(program, itir.Program)

        sdfg = self.generate_sdfg(
            program,
            inp.kwargs["offset_provider"],
            inp.kwargs.get("column_axis", None),
        )

        param_types = tuple(
            interface.Parameter(param, tt.from_value(arg))
            for param, arg in zip(sdfg.arg_names, inp.args)
        )

        module: stages.ProgramSource[languages.SDFG, languages.LanguageSettings] = (
            stages.ProgramSource(
                entry_point=interface.Function(program.id, param_types),
                source_code=sdfg.to_json(),
                library_deps=tuple(),
                language=languages.SDFG,
                language_settings=self._language_settings(),
            )
        )
        return module


class DaCeTranslationStepFactory(factory.Factory):
    class Meta:
        model = DaCeTranslator


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
        DaCeTranslationStepFactory,
        device_type=factory.SelfAttribute("..device_type"),
    )
    bindings = _no_bindings
    compilation = factory.SubFactory(
        dace_workflow.DaCeCompilationStepFactory,
        cache_lifetime=factory.LazyFunction(lambda: config.BUILD_CACHE_LIFETIME),
        cmake_build_type=factory.SelfAttribute("..cmake_build_type"),
    )
    decoration = factory.LazyAttribute(
        lambda o: functools.partial(
            dace_workflow.convert_args,
            device=o.device_type,
        )
    )
