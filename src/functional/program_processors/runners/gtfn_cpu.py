# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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


import dataclasses
from typing import Any, Callable, Final, Optional

import numpy as np

from functional.iterator import ir as itir
from functional.otf import languages, stages, workflow
from functional.otf.binding import pybind
from functional.otf.compilation import cache, compiler
from functional.otf.compilation.build_systems import compiledb
from functional.otf.source import cpp_gen
from functional.program_processors import processor_interface as fpi
from functional.program_processors.codegens.gtfn import gtfn_module


# TODO(ricoh): Add support for the whole range of arguments that can be passed to a fencil.
def convert_arg(arg: Any) -> Any:
    if hasattr(arg, "__array__"):
        return np.asarray(arg)
    else:
        return arg


@dataclasses.dataclass(frozen=True)
class GTFNExecutor(fpi.ProgramExecutor):
    language_settings: languages.LanguageWithHeaderFilesSettings = cpp_gen.CPP_DEFAULT
    builder_factory: compiler.BuildSystemProjectGenerator = compiledb.CompiledbFactory()

    name: Optional[str] = None

    def __call__(self, program: itir.FencilDefinition, *args: Any, **kwargs: Any) -> None:
        """
        Execute the iterator IR program with the provided arguments.

        The program is compiled to machine code with C++ as an intermediate step,
        so the first execution is expected to have a significant overhead, while subsequent
        calls are very fast. Only scalar and buffer arguments are supported currently.

        See ``FencilExecutorFunction`` for details.
        """

        def convert_args(inp: Callable) -> Callable:
            def decorated_program(*args):
                return inp(*[convert_arg(arg) for arg in args])

            return decorated_program

        @workflow.make_step
        def itir_to_src(inp: stages.ProgramCall) -> stages.ProgramSource:
            return gtfn_module.GTFNSourceGenerator(self.language_settings)(
                inp.program, *inp.args, **inp.kwargs
            )

        def src_to_otf(inp: stages.ProgramSource) -> stages.CompilableSource:
            return stages.CompilableSource(
                program_source=inp, binding_source=pybind.create_bindings(inp)
            )

        otf_workflow: Final[workflow.Workflow[stages.ProgramCall, Any, stages.CompiledProgram]] = (
            itir_to_src.chain(pybind.program_source_to_compilable_source)
            .chain(
                compiler.Compiler(
                    cache_strategy=cache.Strategy.SESSION, builder_factory=self.builder_factory
                )
            )
            .chain(convert_args)
        )

        otf_closure = stages.ProgramCall(program, args, kwargs)

        compiled_runner = otf_workflow(otf_closure)

        compiled_runner(*args)

    @property
    def __name__(self) -> str:
        return self.name or repr(self)


run_gtfn: Final[fpi.ProgramProcessorProtocol[None, fpi.ProgramExecutor]] = GTFNExecutor(
    name="run_gtfn"
)
