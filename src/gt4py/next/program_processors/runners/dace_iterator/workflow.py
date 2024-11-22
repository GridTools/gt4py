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
from typing import Callable, Optional, Sequence

import dace
import factory

from gt4py._core import definitions as core_defs
from gt4py.next import common, config
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms import program_to_fencil
from gt4py.next.otf import languages, recipes, stages, step_types, workflow
from gt4py.next.otf.binding import interface
from gt4py.next.otf.languages import LanguageSettings
from gt4py.next.program_processors.runners.dace_common import workflow as dace_workflow
from gt4py.next.type_system import type_specifications as ts

from . import build_sdfg_from_itir


@dataclasses.dataclass(frozen=True)
class DaCeTranslator(
    workflow.ChainableWorkflowMixin[
        stages.CompilableProgram, stages.ProgramSource[languages.SDFG, languages.LanguageSettings]
    ],
    step_types.TranslationStep[languages.SDFG, languages.LanguageSettings],
):
    auto_optimize: bool = False
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
    symbolic_domain_sizes: Optional[dict[str, str]] = None
    temporary_extraction_heuristics: Optional[
        Callable[[itir.StencilClosure], Callable[[itir.Expr], bool]]
    ] = None
    use_field_canonical_representation: bool = False

    def _language_settings(self) -> languages.LanguageSettings:
        return languages.LanguageSettings(
            formatter_key="", formatter_style="", file_extension="sdfg"
        )

    def generate_sdfg(
        self,
        program: itir.FencilDefinition,
        arg_types: Sequence[ts.TypeSpec],
        offset_provider: dict[str, common.Dimension | common.Connectivity],
        column_axis: Optional[common.Dimension],
    ) -> dace.SDFG:
        on_gpu = (
            True
            if self.device_type in [core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM]
            else False
        )

        return build_sdfg_from_itir(
            program,
            arg_types,
            offset_provider=offset_provider,
            auto_optimize=self.auto_optimize,
            on_gpu=on_gpu,
            column_axis=column_axis,
            symbolic_domain_sizes=self.symbolic_domain_sizes,
            temporary_extraction_heuristics=self.temporary_extraction_heuristics,
            load_sdfg_from_file=False,
            save_sdfg=False,
            use_field_canonical_representation=self.use_field_canonical_representation,
        )

    def __call__(
        self, inp: stages.CompilableProgram
    ) -> stages.ProgramSource[languages.SDFG, LanguageSettings]:
        """Generate DaCe SDFG file from the ITIR definition."""
        program: itir.FencilDefinition | itir.Program = inp.data

        if isinstance(program, itir.Program):
            program = program_to_fencil.program_to_fencil(program)

        sdfg = self.generate_sdfg(
            program,
            inp.args.args,
            inp.args.offset_provider,
            inp.args.column_axis,
        )

        param_types = tuple(
            interface.Parameter(param, arg) for param, arg in zip(sdfg.arg_names, inp.args.args)
        )

        module: stages.ProgramSource[languages.SDFG, languages.LanguageSettings] = (
            stages.ProgramSource(
                entry_point=interface.Function(program.id, param_types),
                source_code=sdfg.to_json(),
                library_deps=tuple(),
                language=languages.SDFG,
                language_settings=self._language_settings(),
                implicit_domain=inp.data.implicit_domain,
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
        dace_workflow.DaCeCompilationStepFactory,
        cache_lifetime=factory.LazyFunction(lambda: config.BUILD_CACHE_LIFETIME),
        cmake_build_type=factory.SelfAttribute("..cmake_build_type"),
    )
    decoration = factory.LazyAttribute(
        lambda o: functools.partial(
            dace_workflow.convert_args,
            device=o.device_type,
            use_field_canonical_representation=o.use_field_canonical_representation,
        )
    )
