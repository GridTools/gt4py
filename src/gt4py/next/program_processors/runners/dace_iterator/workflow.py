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

from __future__ import annotations

import dataclasses
import warnings
from typing import Callable, Optional, cast

import dace
import factory
from dace.codegen.compiled_sdfg import CompiledSDFG

from gt4py._core import definitions as core_defs
from gt4py.next import common, config
from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms import LiftMode
from gt4py.next.otf import languages, stages, step_types, workflow
from gt4py.next.otf.binding import interface
from gt4py.next.otf.compilation import cache
from gt4py.next.otf.languages import LanguageSettings
from gt4py.next.type_system import type_translation as tt

from . import build_sdfg_from_itir, get_sdfg_args


@dataclasses.dataclass(frozen=True)
class DaCeTranslator(
    workflow.ChainableWorkflowMixin[
        stages.ProgramCall, stages.ProgramSource[languages.SDFG, languages.LanguageSettings]
    ],
    step_types.TranslationStep[languages.SDFG, languages.LanguageSettings],
):
    auto_optimize: bool = False
    lift_mode: LiftMode = LiftMode.FORCE_INLINE
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
    temporary_extraction_heuristics: Optional[
        Callable[[itir.StencilClosure], Callable[[itir.Expr], bool]]
    ] = None
    use_field_canonical_representation: bool = False

    def _language_settings(self) -> languages.LanguageSettings:
        return languages.LanguageSettings(
            formatter_key="", formatter_style="", file_extension="sdfg"
        )

    def __call__(
        self, inp: stages.ProgramCall
    ) -> stages.ProgramSource[languages.SDFG, LanguageSettings]:
        """Generate DaCe SDFG file from the ITIR definition."""
        program: itir.FencilDefinition = inp.program
        on_gpu = True if self.device_type in [core_defs.DeviceType.CUDA, core_defs.DeciveType.ROCM] else False

        # ITIR parameters
        column_axis: Optional[Dimension] = inp.kwargs.get("column_axis", None)
        offset_provider: dict[str, common.Dimension | common.Connectivity] = inp.kwargs[
            "offset_provider"
        ]
        runtime_lift_mode: Optional[LiftMode] = inp.kwargs.get("lift_mode", None)

        # TODO(tehrengruber): Remove `lift_mode` from call interface. It has been implicitly added
        #  to the interface of all (or at least all of concern) backends, but instead should be
        #  configured in the backend itself (like it is here), until then we respect the argument
        #  here and warn the user if it differs from the one configured.
        lift_mode = runtime_lift_mode or self.lift_mode
        if runtime_lift_mode and runtime_lift_mode != self.lift_mode:
            warnings.warn(
                f"DaCe Backend was configured for LiftMode `{self.lift_mode!s}`, but "
                f"overriden to be {runtime_lift_mode!s} at runtime.",
                stacklevel=2,
            )

        sdfg = build_sdfg_from_itir(
            program,
            *inp.args,
            offset_provider=offset_provider,
            auto_optimize=self.auto_optimize,
            on_gpu=on_gpu,
            column_axis=column_axis,
            lift_mode=lift_mode,
            load_sdfg_from_file=False,
            save_sdfg=False,
            use_field_canonical_representation=self.use_field_canonical_representation,
        )

        arg_types = tuple(
            interface.Parameter(param, tt.from_value(arg))
            for param, arg in zip(sdfg.arg_names, inp.args)
        )

        module: stages.ProgramSource[languages.SDFG, languages.LanguageSettings] = (
            stages.ProgramSource(
                entry_point=interface.Function(program.id, arg_types),
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


@dataclasses.dataclass(frozen=True)
class DaCeCompiler(
    workflow.ChainableWorkflowMixin[
        stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
        stages.CompiledProgram,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
        stages.CompiledProgram,
    ],
    step_types.CompilationStep[languages.SDFG, languages.LanguageSettings, languages.Python],
):
    """Use the dace build system to compile a GT4Py program to a ``gt4py.next.otf.stages.CompiledProgram``."""

    cache_lifetime: config.BuildCacheLifetime
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
    cmake_build_type: config.CMakeBuildType = config.CMakeBuildType.DEBUG

    def __call__(
        self,
        inp: stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
    ) -> stages.CompiledProgram:
        sdfg = dace.SDFG.from_json(inp.program_source.source_code)

        src_dir = cache.get_cache_folder(inp, self.cache_lifetime)
        sdfg.build_folder = src_dir / ".dacecache"

        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "build_type", value=self.cmake_build_type.value)
            if self.device_type == core_defs.DeviceType.CPU:
                compiler_args = dace.config.Config.get("compiler", "cpu", "args")
                # disable finite-math-only in order to support isfinite/isinf/isnan builtins
                if "-ffast-math" in compiler_args:
                    compiler_args += " -fno-finite-math-only"
                if "-ffinite-math-only" in compiler_args:
                    compiler_args.replace("-ffinite-math-only", "")

                dace.config.Config.set("compiler", "cpu", "args", value=compiler_args)
            sdfg_program = sdfg.compile(validate=False)

        return sdfg_program


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler


def convert_args(
    inp: stages.CompiledProgram,
    device: core_defs.DeviceType = core_defs.DeviceType.CPU,
    use_field_canonical_representation: bool = False,
) -> stages.CompiledProgram:
    sdfg_program = cast(CompiledSDFG, inp)
    on_gpu = True if device == core_defs.DeviceType.CUDA else False
    sdfg = sdfg_program.sdfg

    def decorated_program(
        *args, offset_provider: dict[str, common.Connectivity | common.Dimension]
    ):
        sdfg_args = get_sdfg_args(
            sdfg,
            *args,
            check_args=False,
            offset_provider=offset_provider,
            on_gpu=on_gpu,
            use_field_canonical_representation=use_field_canonical_representation,
        )

        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "allow_view_arguments", value=True)
            return inp(**sdfg_args)

    return decorated_program
