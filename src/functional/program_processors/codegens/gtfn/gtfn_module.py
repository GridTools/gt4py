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

import eve.trees
import eve.utils
from functional.common import Connectivity, Dimension
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
        return interface.BufferParameter(name, tuple(dim.value for dim in obj.axes), view.dtype)
    else:
        return interface.ScalarParameter(name, view.dtype)


@dataclasses.dataclass(frozen=True)
class GTFNTranslationStep(
    step_types.TranslationStep[languages.Cpp, languages.LanguageWithHeaderFilesSettings],
):
    language_settings: languages.LanguageWithHeaderFilesSettings = cpp_interface.CPP_DEFAULT

    def _process_arguments(
        self,
        program: itir.FencilDefinition,
        args: tuple[Any, ...],
        offset_provider: dict[str, Connectivity],
    ):
        """Given an ITIR program and arguments generate the interface parameters and arguments."""
        parameters: list[
            interface.ScalarParameter | interface.BufferParameter | interface.ConnectivityParameter
        ] = []
        parameter_args: list[str] = ["gridtools::fn::backend::naive{}"]
        connectivity_args: list[str] = []

        # handle parameters and arguments of the program
        # TODO(tehrengruber): The backend expects all arguments to a stencil closure to be a SID
        #  so transform all scalar arguments that are used in a closure into one before we pass
        #  them to the generated source. This is not a very clean solution and will fail when
        #  the respective parameter is used elsewhere, e.g. in a domain construction, as it is
        #  expected to be scalar there (instead of a SID). We could solve this by:
        #   1.) Extending the backend to support scalar arguments in a closure (as in embedded
        #       backend).
        #   2.) Use SIDs for all arguments and deref when a scalar is required.
        closure_scalar_parameters = (
            eve.trees.pre_walk_values(
                eve.utils.XIterable(program.closures).getattr("inputs").to_list()
            )
            .if_isinstance(itir.SymRef)
            .getattr("id")
            .map(str)
            .to_list()
        )
        for obj, program_param in zip(args, program.params):
            # parameter
            parameter = get_param_description(program_param.id, obj)
            parameters.append(parameter)

            # argument expression
            if (
                isinstance(parameter, interface.ScalarParameter)
                and parameter.name in closure_scalar_parameters
            ):
                # convert into sid
                parameter_args.append(f"gridtools::stencil::global_parameter({parameter.name})")
            else:
                # pass as is
                parameter_args.append(parameter.name)

        # handle connectivity parameters and arguments
        for name, connectivity in offset_provider.items():
            if isinstance(connectivity, Connectivity):
                if connectivity.index_type not in [np.int32, np.int64]:
                    raise ValueError(
                        "Neighbor table indices must be of type `np.int32` or `np.int64`."
                    )

                # parameter
                parameters.append(
                    interface.ConnectivityParameter(
                        "__conn_" + name.lower(),
                        connectivity.origin_axis.value,
                        name,
                        connectivity.index_type,  # type: ignore[arg-type]
                    )
                )

                # connectivity argument expression
                nbtbl = (
                    f"gridtools::fn::sid_neighbor_table::as_neighbor_table<"
                    f"generated::{connectivity.origin_axis.value}_t, "
                    f"generated::{name}_t, {connectivity.max_neighbors}"
                    f">(__conn_{name.lower()})"
                )
                connectivity_args.append(
                    f"gridtools::hymap::keys<generated::{name}_t>::make_values({nbtbl})"
                )  # TODO(havogt): std::forward, type and max_neighbors
            elif isinstance(connectivity, Dimension):
                pass
            else:
                raise ValueError(
                    f"Expected offset provider `{name}` to be a `Connectivity` or `Dimension`, "
                    f"but got {type(connectivity).__name__}."
                )

        return parameters, parameter_args, connectivity_args

    def __call__(
        self,
        inp: stages.ProgramCall,
    ) -> stages.ProgramSource[languages.Cpp, languages.LanguageWithHeaderFilesSettings]:
        """Generate GTFN C++ code from the ITIR definition."""
        program = inp.program
        parameters, parameter_args, connectivity_args = self._process_arguments(
            program, inp.args, inp.kwargs["offset_provider"]
        )

        function = interface.Function(program.id, tuple(parameters))
        decl_body = (
            f"return generated::{function.name}("
            f"{', '.join(connectivity_args)})({', '.join(parameter_args)});"
        )
        decl_src = cpp_interface.render_function_declaration(function, body=decl_body)
        stencil_src = gtfn_backend.generate(program, **inp.kwargs)
        source_code = interface.format_source(
            self.language_settings,
            f"""
                    #include <gridtools/fn/backend/naive.hpp>
                    #include <gridtools/stencil/global_parameter.hpp>
                    {stencil_src}
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
