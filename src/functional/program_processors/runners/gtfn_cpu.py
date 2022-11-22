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
from functional.otf.binding import cpp_interface, pybind
from functional.otf.compilation import cache, compiler
from functional.otf.compilation.build_systems import compiledb
from functional.program_processors import processor_interface as ppi
from functional.program_processors.codegens.gtfn import gtfn_module


# TODO(ricoh): Add support for the whole range of arguments that can be passed to a fencil.
def convert_arg(arg: Any) -> Any:
    if hasattr(arg, "__array__"):
        return np.asarray(arg)
    else:
        return arg


@dataclasses.dataclass(frozen=True)
class GTFNExecutor(ppi.ProgramExecutor):
    language_settings: languages.LanguageWithHeaderFilesSettings = cpp_interface.CPP_DEFAULT
    builder_factory: compiler.BuildSystemProjectGenerator = compiledb.CompiledbFactory()

    name: Optional[str] = None

    def __call__(self, program: itir.FencilDefinition, *args: Any, **kwargs: Any) -> None:
        """
        Execute the iterator IR program with the provided arguments.

        The program is compiled to machine code with C++ as an intermediate step,
        so the first execution is expected to have a significant overhead, while subsequent
        calls are very fast. Only scalar and buffer arguments are supported currently.

        See ``ProgramExecutorFunction`` for details.
        """

        def convert_args(inp: Callable) -> Callable:
            def decorated_program(*args):
                return inp(*[convert_arg(arg) for arg in args])

            return decorated_program

        otf_workflow: Final[workflow.Workflow[stages.ProgramCall, stages.CompiledProgram]] = (
            gtfn_module.GTFNTranslationStep(self.language_settings)
            .chain(pybind.bind_source)
            .chain(
                compiler.Compiler(
                    cache_strategy=cache.Strategy.PERSISTENT, builder_factory=self.builder_factory
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


run_gtfn: Final[ppi.ProgramProcessor[None, ppi.ProgramExecutor]] = GTFNExecutor(name="run_gtfn")
