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
from typing import Any, Final, TypeVar

import numpy as np

from gt4py.eve import trees, utils
from gt4py.next.common import Connectivity, Dimension, DimensionKind
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import languages, stages, step_types, workflow
from gt4py.next.otf.binding import cpp_interface, interface
from gt4py.next.program_processors.codegens.gtfn import gtfn_backend


T = TypeVar("T")

GENERATED_CONNECTIVITY_PARAM_PREFIX = "gt_conn_"


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
    enable_itir_transforms: bool = True  # TODO replace by more general mechanism, see https://github.com/GridTools/gt4py/issues/1135
    use_imperative_backend: bool = False

    def _process_regular_arguments(
        self,
        program: itir.FencilDefinition,
        args: tuple[Any, ...],
    ):
        parameters: list[interface.ScalarParameter | interface.BufferParameter] = []
        arg_exprs: list[str] = []

        # TODO(tehrengruber): The backend expects all arguments to a stencil closure to be a SID
        #  so transform all scalar arguments that are used in a closure into one before we pass
        #  them to the generated source. This is not a very clean solution and will fail when
        #  the respective parameter is used elsewhere, e.g. in a domain construction, as it is
        #  expected to be scalar there (instead of a SID). We could solve this by:
        #   1.) Extending the backend to support scalar arguments in a closure (as in embedded
        #       backend).
        #   2.) Use SIDs for all arguments and deref when a scalar is required.
        closure_scalar_parameters = (
            trees.pre_walk_values(utils.XIterable(program.closures).getattr("inputs").to_list())
            .if_isinstance(itir.SymRef)
            .getattr("id")
            .map(str)
            .to_list()
        )
        for obj, program_param in zip(args, program.params):
            # parameter
            parameter = get_param_description(program_param.id, obj)
            parameters.append(parameter)

            # argument conversion expression
            if (
                isinstance(parameter, interface.ScalarParameter)
                and parameter.name in closure_scalar_parameters
            ):
                # convert into sid
                arg_exprs.append(f"gridtools::stencil::global_parameter({parameter.name})")
            else:
                # pass as is
                arg_exprs.append(parameter.name)
        return parameters, arg_exprs

    def _process_connectivity_args(
        self,
        offset_provider: dict[str, Connectivity | Dimension],
    ):
        parameters: list[interface.ConnectivityParameter] = []
        arg_exprs: list[str] = []

        for name, connectivity in offset_provider.items():
            if isinstance(connectivity, Connectivity):
                if connectivity.index_type not in [np.int32, np.int64]:
                    raise ValueError(
                        "Neighbor table indices must be of type `np.int32` or `np.int64`."
                    )

                # parameter
                parameters.append(
                    interface.ConnectivityParameter(
                        GENERATED_CONNECTIVITY_PARAM_PREFIX + name.lower(),
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
                    f">({GENERATED_CONNECTIVITY_PARAM_PREFIX}{name.lower()})"
                )
                arg_exprs.append(
                    f"gridtools::hymap::keys<generated::{name}_t>::make_values({nbtbl})"
                )
            elif isinstance(connectivity, Dimension):
                pass
            else:
                raise AssertionError(
                    f"Expected offset provider `{name}` to be a `Connectivity` or `Dimension`, "
                    f"but got {type(connectivity).__name__}."
                )

        return parameters, arg_exprs

    def __call__(
        self,
        inp: stages.ProgramCall,
    ) -> stages.ProgramSource[languages.Cpp, languages.LanguageWithHeaderFilesSettings]:
        """Generate GTFN C++ code from the ITIR definition."""
        program: itir.FencilDefinition = inp.program

        # handle regular parameters and arguments of the program (i.e. what the user defined in
        #  the program)
        regular_parameters, regular_args_expr = self._process_regular_arguments(program, inp.args)

        # handle connectivity parameters and arguments (i.e. what the user provided in the offset
        #  provider)
        connectivity_parameters, connectivity_args_expr = self._process_connectivity_args(
            inp.kwargs["offset_provider"]
        )

        # combine into a format that is aligned with what the backend expects
        parameters: list[
            interface.ScalarParameter | interface.BufferParameter | interface.ConnectivityParameter
        ] = [*regular_parameters, *connectivity_parameters]
        args_expr: list[str] = ["gridtools::fn::backend::naive{}", *regular_args_expr]

        function = interface.Function(program.id, tuple(parameters))
        decl_body = (
            f"return generated::{function.name}("
            f"{', '.join(connectivity_args_expr)})({', '.join(args_expr)});"
        )
        decl_src = cpp_interface.render_function_declaration(function, body=decl_body)
        stencil_src = gtfn_backend.generate(
            program,
            enable_itir_transforms=self.enable_itir_transforms,
            imperative=self.use_imperative_backend,
            **inp.kwargs,
        )
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
