# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Any

import dace
import factory

from gt4py._core import definitions as core_defs
from gt4py.next import config
from gt4py.next.otf import languages, stages, step_types, workflow
from gt4py.next.otf.compilation import cache


class CompiledDaceProgram(stages.ExtendedCompiledProgram):
    sdfg_program: dace.CompiledSDFG

    # Sorted list of SDFG arguments as they appear in program ABI and corresponding data type;
    # scalar arguments that are not used in the SDFG will not be present.
    sdfg_arglist: list[tuple[str, dace.dtypes.Data]]

    def __init__(self, program: dace.CompiledSDFG, implicit_domain: bool):
        self.sdfg_program = program
        self.implicit_domain = implicit_domain
        # `dace.CompiledSDFG.arglist()` returns an ordered dictionary that maps the argument
        # name to its data type, in the same order as arguments appear in the program ABI.
        # This is also the same order of arguments in `dace.CompiledSDFG._lastargs[0]`.
        self.sdfg_arglist = [
            (arg_name, arg_type) for arg_name, arg_type in program.sdfg.arglist().items()
        ]

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        result = self.sdfg_program(*args, **kwargs)
        assert result is None

    def fast_call(self) -> None:
        result = self.sdfg_program.fast_call(*self.sdfg_program._lastargs)
        assert result is None


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
        sdfg.build_folder = cache.get_cache_folder(inp, self.cache_lifetime)

        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "build_type", value=self.cmake_build_type.value)
            dace.config.Config.set("compiler", "use_cache", value=False)  # we use the gt4py cache
            # In some stencils, mostly in `apply_diffusion_to_w` the cuda codegen messes
            #  up with the cuda streams, i.e. it allocates N streams but uses N+1.
            #  This is a workaround until this issue if fixed in DaCe.
            dace.config.Config.set("compiler", "cuda", "max_concurrent_streams", value=1)

            if self.device_type == core_defs.DeviceType.CPU:
                compiler_args = dace.config.Config.get("compiler", "cpu", "args")
                # disable finite-math-only in order to support isfinite/isinf/isnan builtins
                if "-ffast-math" in compiler_args:
                    compiler_args += " -fno-finite-math-only"
                if "-ffinite-math-only" in compiler_args:
                    compiler_args.replace("-ffinite-math-only", "")

                dace.config.Config.set("compiler", "cpu", "args", value=compiler_args)
            sdfg_program = sdfg.compile(validate=False)

        return CompiledDaceProgram(sdfg_program, inp.program_source.implicit_domain)


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler
