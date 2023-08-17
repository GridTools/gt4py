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
import enum
from typing import Any, Final, TypeVar, Optional

import numpy as np

from gt4py.eve import trees, utils
from gt4py.next import common
from gt4py.next.common import Connectivity, Dimension
from gt4py.next.ffront import fbuiltins
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import languages, stages, step_types, workflow
from gt4py.next.otf.binding import cpp_interface, interface
from gt4py.next.program_processors.codegens.gtfn import gtfn_backend
from gt4py.next.type_system import type_specifications as ts, type_translation


T = TypeVar("T")

GENERATED_CONNECTIVITY_PARAM_PREFIX = "gt_conn_"


def get_param_description(name: str, obj: Any) -> interface.Parameter:
    return interface.Parameter(name, type_translation.from_value(obj))


class HardwareAccelerator(enum.IntEnum):
    CPU = enum.auto()
    GPU = enum.auto()


@dataclasses.dataclass(frozen=True)
class GTFNTranslationStep(
    workflow.ChainableWorkflowMixin[
        stages.ProgramCall,
        stages.ProgramSource[languages.Cpp, languages.LanguageWithHeaderFilesSettings],
    ],
    step_types.TranslationStep[languages.Cpp, languages.LanguageWithHeaderFilesSettings],
):
    language_settings: Optional[languages.LanguageWithHeaderFilesSettings] = None
    enable_itir_transforms: bool = True  # TODO replace by more general mechanism, see https://github.com/GridTools/gt4py/issues/1135
    use_imperative_backend: bool = False
    hardware_accelerator: HardwareAccelerator = HardwareAccelerator.CPU

    @staticmethod
    def _default_language_settings(hardware_accelerator: HardwareAccelerator) -> languages.LanguageSettings:
        if hardware_accelerator == HardwareAccelerator.GPU:
            return languages.LanguageWithHeaderFilesSettings(
                formatter_key=cpp_interface.CPP_DEFAULT.formatter_key,
                formatter_style=cpp_interface.CPP_DEFAULT.formatter_style,
                file_extension="cu",
                header_extension="cuh",
            )
        return cpp_interface.CPP_DEFAULT

    def _process_regular_arguments(
        self,
        program: itir.FencilDefinition,
        args: tuple[Any, ...],
        offset_provider: dict[str, Connectivity | Dimension],
    ) -> tuple[list[interface.Parameter], list[str]]:
        parameters: list[interface.Parameter] = []
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

            arg = f"std::forward<decltype({parameter.name})>({parameter.name})"

            # argument conversion expression
            if (
                isinstance(parameter.type_, ts.ScalarType)
                and parameter.name in closure_scalar_parameters
            ):
                # convert into sid
                arg = f"gridtools::stencil::global_parameter({arg})"
            elif isinstance(parameter.type_, ts.FieldType):
                for dim in parameter.type_.dims:
                    if (
                        isinstance(
                            dim, fbuiltins.FieldOffset
                        )  # TODO(havogt): remove support for FieldOffset as Dimension
                        or dim.kind == common.DimensionKind.LOCAL
                    ):
                        # translate sparse dimensions to tuple dtype
                        dim_name = dim.value
                        connectivity = offset_provider[dim_name]
                        assert isinstance(connectivity, Connectivity)
                        size = connectivity.max_neighbors
                        arg = f"gridtools::sid::dimension_to_tuple_like<generated::{dim_name}_t, {size}>({arg})"
            arg_exprs.append(arg)
        return parameters, arg_exprs

    def _process_connectivity_args(
        self,
        offset_provider: dict[str, Connectivity | Dimension],
    ) -> tuple[list[interface.Parameter], list[str]]:
        parameters: list[interface.Parameter] = []
        arg_exprs: list[str] = []

        for name, connectivity in offset_provider.items():
            if isinstance(connectivity, Connectivity):
                if connectivity.index_type not in [np.int32, np.int64]:
                    raise ValueError(
                        "Neighbor table indices must be of type `np.int32` or `np.int64`."
                    )

                # parameter
                parameters.append(
                    interface.Parameter(
                        name=GENERATED_CONNECTIVITY_PARAM_PREFIX + name.lower(),
                        type_=ts.FieldType(
                            dims=[connectivity.origin_axis, Dimension(name)],
                            dtype=ts.ScalarType(
                                type_translation.get_scalar_kind(connectivity.index_type)
                            ),
                        ),
                    )
                )

                # connectivity argument expression
                nbtbl = (
                    f"gridtools::fn::sid_neighbor_table::as_neighbor_table<"
                    f"generated::{connectivity.origin_axis.value}_t, "
                    f"generated::{name}_t, {connectivity.max_neighbors}"
                    f">(std::forward<decltype({GENERATED_CONNECTIVITY_PARAM_PREFIX}{name.lower()})>({GENERATED_CONNECTIVITY_PARAM_PREFIX}{name.lower()}))"
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
        regular_parameters, regular_args_expr = self._process_regular_arguments(
            program, inp.args, inp.kwargs["offset_provider"]
        )

        # handle connectivity parameters and arguments (i.e. what the user provided in the offset
        #  provider)
        connectivity_parameters, connectivity_args_expr = self._process_connectivity_args(
            inp.kwargs["offset_provider"]
        )

        # combine into a format that is aligned with what the backend expects
        parameters: list[interface.Parameter] = regular_parameters + connectivity_parameters
        backend_arg = self._get_backend_type()
        args_expr: list[str] = [backend_arg, *regular_args_expr]

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
        language_settings = self.language_settings if self.language_settings is not None else GTFNTranslationStep._default_language_settings(self.hardware_accelerator)
        backend_header = "gridtools/fn/backend/gpu.hpp" if self.hardware_accelerator == HardwareAccelerator.GPU else "gridtools/fn/backend/naive.hpp"
        source_code = interface.format_source(
            language_settings,
            f"""
                    #include <{backend_header}>
                    #include <gridtools/stencil/global_parameter.hpp>
                    #include <gridtools/sid/dimension_to_tuple_like.hpp>
                    {stencil_src}
                    {decl_src}
                    """.strip(),
        )

        language = languages.Cuda if self.hardware_accelerator == HardwareAccelerator.GPU else languages.Cpp
        library_name = "gridtools_gpu" if self.hardware_accelerator == HardwareAccelerator.GPU else "gridtools_cpu"
        module = stages.ProgramSource(
            entry_point=function,
            library_deps=(interface.LibraryDependency(library_name, "master"),),
            source_code=source_code,
            language=language,
            language_settings=language_settings,
        )
        return module

    def _get_backend_type(self):
        if self.hardware_accelerator == HardwareAccelerator.GPU:

            return "gridtools::fn::backend::gpu<generated::block_sizes_t>{}"
        return "gridtools::fn::backend::naive{}"


translate_program: Final[
    step_types.TranslationStep[languages.Cpp, languages.LanguageWithHeaderFilesSettings]
] = GTFNTranslationStep()
