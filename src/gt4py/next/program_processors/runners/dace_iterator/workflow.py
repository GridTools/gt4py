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

import ctypes
import dataclasses
import functools
import itertools
import operator
from typing import Callable, Optional

import dace
import factory
from dace.codegen.compiled_sdfg import _array_interface_ptr as get_array_interface_ptr

from gt4py._core import definitions as core_defs
from gt4py.next import common, config
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms import LiftMode
from gt4py.next.otf import languages, stages, step_types, workflow
from gt4py.next.otf.binding import interface
from gt4py.next.otf.compilation import cache
from gt4py.next.otf.languages import LanguageSettings
from gt4py.next.type_system import type_specifications as ts, type_translation as tt

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
        arg_types: list[ts.TypeSpec],
        offset_provider: dict[str, common.Dimension | common.Connectivity],
        column_axis: Optional[common.Dimension],
    ) -> dace.SDFG:
        on_gpu = True if self.device_type == core_defs.DeviceType.CUDA else False

        return build_sdfg_from_itir(
            program,
            arg_types,
            offset_provider=offset_provider,
            auto_optimize=self.auto_optimize,
            on_gpu=on_gpu,
            column_axis=column_axis,
            lift_mode=self.lift_mode,
            symbolic_domain_sizes=self.symbolic_domain_sizes,
            temporary_extraction_heuristics=self.temporary_extraction_heuristics,
            load_sdfg_from_file=False,
            save_sdfg=False,
            use_field_canonical_representation=self.use_field_canonical_representation,
        )

    def __call__(
        self, inp: stages.ProgramCall
    ) -> stages.ProgramSource[languages.SDFG, LanguageSettings]:
        """Generate DaCe SDFG file from the ITIR definition."""
        program: itir.FencilDefinition = inp.program
        arg_types = [tt.from_value(arg) for arg in inp.args]

        sdfg = self.generate_sdfg(
            program,
            arg_types,
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


class CompiledDaceProgram(stages.CompiledProgram):
    sdfg_program: dace.CompiledSDFG
    sdfg_scalar_args: dict[str, int]  # map arg to its position in program ABI

    def __init__(self, program, scalar_args):
        self.sdfg_program = program
        self.sdfg_scalar_args = scalar_args

    def __call__(self, *args, **kwargs) -> None:
        self.sdfg_program(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class DaCeCompiler(
    workflow.ChainableWorkflowMixin[
        stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
        CompiledDaceProgram,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
        CompiledDaceProgram,
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
    ) -> CompiledDaceProgram:
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

        # extract position of scalar arguments from program ABI
        array_symbols = functools.reduce(
            operator.or_, itertools.chain(x.free_symbols for x in sdfg.arrays.values())
        )
        array_symbols = {str(x) for x in array_symbols}
        sdfg_arglist = sdfg_program.sdfg.signature_arglist(with_types=False)
        sdfg_scalar_args = {
            param: pos
            for pos, param in enumerate(sdfg_arglist)
            if (param not in sdfg.arrays and param not in array_symbols)
        }
        return CompiledDaceProgram(sdfg_program, sdfg_scalar_args)


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler


def convert_args(
    inp: CompiledDaceProgram,
    device: core_defs.DeviceType = core_defs.DeviceType.CPU,
    use_field_canonical_representation: bool = False,
) -> stages.CompiledProgram:
    sdfg_program = inp.sdfg_program
    sdfg = sdfg_program.sdfg
    on_gpu = True if device == core_defs.DeviceType.CUDA else False

    def decorated_program(
        *args, offset_provider: dict[str, common.Connectivity | common.Dimension]
    ):
        if sdfg_program._lastargs:
            # The scalar arguments should be replaced with the actual value; for field arguments,
            # the data pointer should remain the same otherwise reconstruct the args list.
            use_fast_call = True
            scalar_args_table: dict[str, ctypes._SimpleCData] = {}
            for pos, (param, arg) in enumerate(zip(sdfg.arg_names, args), start=1):
                if isinstance(arg, common.Field):
                    storage_type = sdfg.arrays[param].storage
                    assert isinstance(sdfg_program._lastargs[0][pos], ctypes.c_void_p)
                    if sdfg_program._lastargs[0][pos].value != get_array_interface_ptr(
                        arg.ndarray, storage_type
                    ):
                        use_fast_call = False
                        break
                else:
                    scalar_args_table[param] = arg

            if use_fast_call:
                for param, pos in inp.sdfg_scalar_args.items():
                    assert not isinstance(sdfg_program._lastargs[0][pos], ctypes.c_void_p)
                    sdfg_program._lastargs[0][pos].value = scalar_args_table[param]
                return sdfg_program.fast_call(*sdfg_program._lastargs)

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
