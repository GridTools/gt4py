# GT4Py - GridTools Framework
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
import numpy.typing as npt

from gt4py.eve.utils import content_hash
from gt4py.next import common
from gt4py.next.type_system.type_translation import from_value
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import languages, stages, workflow
from gt4py.next.otf.binding import cpp_interface, pybind
from gt4py.next.otf.compilation import cache, compiler
from gt4py.next.otf.compilation.build_systems import compiledb, cmake
from gt4py.next.program_processors import processor_interface as ppi
from gt4py.next.program_processors.codegens.gtfn import gtfn_module


# TODO(ricoh): Add support for the whole range of arguments that can be passed to a fencil.
def convert_arg(arg: Any) -> Any:
    if hasattr(arg, "__array__"):
        return np.asarray(arg)
    else:
        return arg


def extract_connectivity_args(
    offset_provider: dict[str, common.Connectivity | common.Dimension]
) -> list[npt.NDArray]:
    # note: the order here needs to agree with the order of the generated bindings
    args: list[npt.NDArray] = []
    for name, conn in offset_provider.items():
        if isinstance(conn, common.Connectivity):
            if not isinstance(conn, common.NeighborTable):
                raise NotImplementedError(
                    "Only `NeighborTable` connectivities implemented at this point."
                )
            args.append(conn.table)
        elif isinstance(conn, common.Dimension):
            pass
        else:
            raise AssertionError(
                f"Expected offset provider `{name}` to be a `Connectivity` or `Dimension`, "
                f"but got {type(conn).__name__}."
            )
    return args


@dataclasses.dataclass(frozen=True)
class GTFNExecutor(ppi.ProgramExecutor):
    language_settings: languages.LanguageWithHeaderFilesSettings = cpp_interface.CPP_DEFAULT
    builder_factory: compiler.BuildSystemProjectGenerator = cmake.CMakeFactory()

    name: Optional[str] = None
    enable_itir_transforms: bool = True  # TODO replace by more general mechanism, see https://github.com/GridTools/gt4py/issues/1135
    use_imperative_backend: bool = False

    _cache: dict[int, Callable] = dataclasses.field(repr=False, init=False, default_factory=dict)

    def __call__(self, program: itir.FencilDefinition, *args: Any, **kwargs: Any) -> None:
        """
        Execute the iterator IR program with the provided arguments.

        The program is compiled to machine code with C++ as an intermediate step,
        so the first execution is expected to have a significant overhead, while subsequent
        calls are very fast. Only scalar and buffer arguments are supported currently.

        See ``ProgramExecutorFunction`` for details.
        """
        cache_key = hash(
            (
                program,
                # TODO(tehrengruber): as the resulting frontend types contain lists they are
                #  not hashable. As a workaround we just use content_hash here.
                content_hash(tuple(from_value(arg) for arg in args)),
                id(kwargs["offset_provider"]),
                kwargs["column_axis"],
            )
        )

        def convert_args(inp: Callable) -> Callable:
            def decorated_program(*args):
                return inp(
                    *[convert_arg(arg) for arg in args],
                    *extract_connectivity_args(kwargs["offset_provider"]),
                )

            return decorated_program

        if cache_key not in self._cache:
            otf_workflow: Final[workflow.Workflow[stages.ProgramCall, stages.CompiledProgram]] = (
                gtfn_module.GTFNTranslationStep(
                    self.language_settings, self.enable_itir_transforms, self.use_imperative_backend
                )
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
        else:
            compiled_runner = self._cache[cache_key]

        compiled_runner(*args)

    @property
    def __name__(self) -> str:
        return self.name or repr(self)


run_gtfn: Final[ppi.ProgramProcessor[None, ppi.ProgramExecutor]] = GTFNExecutor(
    name="run_gtfn", use_imperative_backend=False
)
run_gtfn_imperative: Final[ppi.ProgramProcessor[None, ppi.ProgramExecutor]] = GTFNExecutor(
    name="run_gtfn_imperative", use_imperative_backend=True
)
