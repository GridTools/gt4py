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
from typing import Any, Final, TypeVar

import numpy as np

from functional.common import Connectivity, Dimension, DimensionKind
from functional.iterator import ir as itir
from functional.otf import languages, stages, step_types, workflow
from functional.otf.binding import cpp_interface, interface
from functional.program_processors.codegens.gtfn import gtfn_backend


T = TypeVar("T")


def get_param_description(
    name: str, obj: Any
) -> interface.ScalarParameter | interface.BufferParameter:
    view: np.ndarray = np.asarray(obj)
    if view.ndim > 0:
        return interface.BufferParameter(
            name,
            tuple(
                dim.value if dim.kind != DimensionKind.LOCAL else dim.value + "Dim"
                for dim in obj.axes
            ),
            view.dtype,
        )
    else:
        return interface.ScalarParameter(name, view.dtype)


@dataclasses.dataclass(frozen=True)
class GTFNTranslationStep(
    step_types.TranslationStep[languages.Cpp, languages.LanguageWithHeaderFilesSettings],
):
    language_settings: languages.LanguageWithHeaderFilesSettings = cpp_interface.CPP_DEFAULT

    def __call__(
        self,
        inp: stages.ProgramCall,
    ) -> stages.ProgramSource[languages.Cpp, languages.LanguageWithHeaderFilesSettings]:
        """Generate GTFN C++ code from the ITIR definition."""
        program = inp.program
        parameters = [
            get_param_description(program_param.id, obj)
            for obj, program_param in zip(inp.args, program.params)
        ]
        for name, connectivity in inp.kwargs["offset_provider"].items():
            if isinstance(connectivity, Connectivity):
                parameters.append(
                    interface.ConnectivityParameter(
                        "__conn_" + name.lower(), connectivity.origin_axis.value, name
                    )
                )
            elif isinstance(connectivity, Dimension):
                pass
            else:
                raise ValueError(
                    f"Expected offset provider `{name}` to be a "
                    f"`Connectivity` or `Dimension`, but got "
                    f"{type(connectivity).__name__}"
                )
        function = interface.Function(program.id, tuple(parameters))

        connectivity_args = []
        for name, connectivity in inp.kwargs["offset_provider"].items():
            if isinstance(connectivity, Connectivity):
                nbtbl = (
                    f"as_neighbor_table<generated::{connectivity.origin_axis.value}_t, "
                    f"generated::{name}_t, {connectivity.max_neighbors}>(__conn_{name.lower()})"
                )
                connectivity_args.append(
                    f"gridtools::hymap::keys<generated::{name}_t>::make_values({nbtbl})"
                )  # TODO std::forward, type and max_neighbors)
            elif isinstance(connectivity, Dimension):
                pass
            else:
                raise ValueError(
                    f"Expected offset provider `{name}` to be a "
                    f"`Connectivity` or `Dimension`, but got "
                    f"{type(connectivity).__name__}"
                )
        rendered_connectivity_args = ", ".join(connectivity_args)

        import eve.trees
        import eve.utils

        scalar_parameters = (
            eve.trees.pre_walk_values(
                eve.utils.XIterable(program.closures).getattr("inputs").to_list()
            )
            .if_isinstance(itir.SymRef)
            .getattr("id")
            .map(str)
            .to_list()
        )

        parameter_args = ["gridtools::fn::backend::naive{}"]
        for p in parameters:
            if isinstance(p, (interface.ScalarParameter, interface.BufferParameter)):
                if isinstance(p, interface.ScalarParameter) and p.name in scalar_parameters:
                    parameter_args.append(f"gridtools::stencil::global_parameter({p.name})")
                else:
                    parameter_args.append(p.name)

        rendered_parameter_args = ", ".join(parameter_args)
        decl_body = f"return generated::{function.name}({rendered_connectivity_args})({rendered_parameter_args});"
        decl_src = cpp_interface.render_function_declaration(function, body=decl_body)
        stencil_src = gtfn_backend.generate(program, **inp.kwargs)
        source_code = interface.format_source(
            self.language_settings,
            f"""
                    #include <gridtools/fn/backend/naive.hpp>
                    #include <gridtools/stencil/global_parameter.hpp>
                    {stencil_src}
                    using gridtools::fn::sid_neighbor_table::as_neighbor_table;
                    {decl_src}
                    """.strip(),
        )

        module = stages.ProgramSource(
            entry_point=function,
            library_deps=(interface.LibraryDependency("gridtools", "master"),),
            source_code=source_code,
            language=languages.Cpp,
            language_settings=self.language_settings,
        )
        return module

    def chain(
        self,
        step: workflow.Workflow[
            stages.ProgramSource[languages.Cpp, languages.LanguageWithHeaderFilesSettings], T
        ],
    ) -> workflow.CombinedStep[
        stages.ProgramCall,
        stages.ProgramSource[languages.Cpp, languages.LanguageWithHeaderFilesSettings],
        T,
    ]:
        return workflow.CombinedStep(first=self, second=step)


translate_program: Final[
    step_types.TranslationStep[languages.Cpp, languages.LanguageWithHeaderFilesSettings]
] = GTFNTranslationStep()
