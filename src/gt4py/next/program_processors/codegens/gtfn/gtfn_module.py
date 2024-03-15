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
import functools
import warnings
from typing import Any, Callable, Final, Optional

import factory
import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.eve import codegen, trees, utils
from gt4py.next import common
from gt4py.next.common import Connectivity, Dimension
from gt4py.next.ffront import fbuiltins
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms import LiftMode, pass_manager
from gt4py.next.otf import languages, stages, step_types, workflow
from gt4py.next.otf.binding import cpp_interface, interface
from gt4py.next.program_processors.codegens.gtfn.codegen import GTFNCodegen, GTFNIMCodegen
from gt4py.next.program_processors.codegens.gtfn.gtfn_ir_to_gtfn_im_ir import GTFN_IM_lowering
from gt4py.next.program_processors.codegens.gtfn.itir_to_gtfn_ir import GTFN_lowering
from gt4py.next.type_system import type_specifications as ts, type_translation


GENERATED_CONNECTIVITY_PARAM_PREFIX = "gt_conn_"


def get_param_description(name: str, obj: Any) -> interface.Parameter:
    return interface.Parameter(name, type_translation.from_value(obj))


@dataclasses.dataclass(frozen=True)
class GTFNTranslationStep(
    workflow.ChainableWorkflowMixin[
        stages.ProgramCall,
        stages.ProgramSource[languages.NanobindSrcL, languages.LanguageWithHeaderFilesSettings],
    ],
    step_types.TranslationStep[languages.NanobindSrcL, languages.LanguageWithHeaderFilesSettings],
):
    language_settings: Optional[languages.LanguageWithHeaderFilesSettings] = None
    # TODO replace by more general mechanism, see https://github.com/GridTools/gt4py/issues/1135
    enable_itir_transforms: bool = True
    use_imperative_backend: bool = False
    lift_mode: Optional[LiftMode] = None
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
    symbolic_domain_sizes: Optional[dict[str, str]] = None
    temporary_extraction_heuristics: Optional[
        Callable[[itir.StencilClosure], Callable[[itir.Expr], bool]]
    ] = None

    def _default_language_settings(self) -> languages.LanguageWithHeaderFilesSettings:
        match self.device_type:
            case core_defs.DeviceType.CUDA:
                return languages.LanguageWithHeaderFilesSettings(
                    formatter_key=cpp_interface.CPP_DEFAULT.formatter_key,
                    formatter_style=cpp_interface.CPP_DEFAULT.formatter_style,
                    file_extension="cu",
                    header_extension="cuh",
                )
            case core_defs.DeviceType.CPU:
                return cpp_interface.CPP_DEFAULT
            case _:
                raise self._not_implemented_for_device_type()

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
                        or dim.kind is common.DimensionKind.LOCAL
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
                        "Neighbor table indices must be of type 'np.int32' or 'np.int64'."
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
                    f"Expected offset provider '{name}' to be a 'Connectivity' or 'Dimension', "
                    f"got '{type(connectivity).__name__}'."
                )

        return parameters, arg_exprs

    def _preprocess_program(
        self,
        program: itir.FencilDefinition,
        offset_provider: dict[str, Connectivity | Dimension],
        runtime_lift_mode: Optional[LiftMode],
    ) -> itir.FencilDefinition:
        # TODO(tehrengruber): Remove `lift_mode` from call interface. It has been implicitly added
        #  to the interface of all (or at least all of concern) backends, but instead should be
        #  configured in the backend itself (like it is here), until then we respect the argument
        #  here and warn the user if it differs from the one configured.
        lift_mode = runtime_lift_mode or self.lift_mode
        if runtime_lift_mode and runtime_lift_mode != self.lift_mode:
            warnings.warn(
                f"GTFN Backend was configured for LiftMode `{self.lift_mode!s}`, but "
                f"overriden to be {runtime_lift_mode!s} at runtime.",
                stacklevel=2,
            )

        if not self.enable_itir_transforms:
            return program

        apply_common_transforms = functools.partial(
            pass_manager.apply_common_transforms,
            lift_mode=lift_mode,
            offset_provider=offset_provider,
            # sid::composite (via hymap) supports assigning from tuple with more elements to tuple with fewer elements
            unconditionally_collapse_tuples=True,
            symbolic_domain_sizes=self.symbolic_domain_sizes,
            temporary_extraction_heuristics=self.temporary_extraction_heuristics,
        )

        new_program = apply_common_transforms(
            program, unroll_reduce=not self.use_imperative_backend
        )

        if self.use_imperative_backend and any(
            node.id == "neighbors"
            for node in new_program.pre_walk_values().if_isinstance(itir.SymRef)
        ):
            # if we don't unroll, there may be lifts left in the itir which can't be lowered to
            # gtfn. In this case, just retry with unrolled reductions.
            new_program = apply_common_transforms(program, unroll_reduce=True)

        return new_program

    def generate_stencil_source(
        self,
        program: itir.FencilDefinition,
        offset_provider: dict[str, Connectivity | Dimension],
        column_axis: Optional[common.Dimension],
        runtime_lift_mode: Optional[LiftMode] = None,
    ) -> str:
        new_program = self._preprocess_program(program, offset_provider, runtime_lift_mode)
        gtfn_ir = GTFN_lowering.apply(
            new_program,
            offset_provider=offset_provider,
            column_axis=column_axis,
        )

        if self.use_imperative_backend:
            gtfn_im_ir = GTFN_IM_lowering().visit(node=gtfn_ir)
            generated_code = GTFNIMCodegen.apply(gtfn_im_ir)
        else:
            generated_code = GTFNCodegen.apply(gtfn_ir)
        return codegen.format_source("cpp", generated_code, style="LLVM")

    def __call__(
        self,
        inp: stages.ProgramCall,
    ) -> stages.ProgramSource[languages.NanobindSrcL, languages.LanguageWithHeaderFilesSettings]:
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
        backend_arg = self._backend_type()
        args_expr: list[str] = [backend_arg, *regular_args_expr]

        function = interface.Function(program.id, tuple(parameters))
        decl_body = (
            f"return generated::{function.name}("
            f"{', '.join(connectivity_args_expr)})({', '.join(args_expr)});"
        )
        decl_src = cpp_interface.render_function_declaration(function, body=decl_body)
        stencil_src = self.generate_stencil_source(
            program,
            inp.kwargs["offset_provider"],
            inp.kwargs.get("column_axis", None),
            inp.kwargs.get("lift_mode", None),
        )
        source_code = interface.format_source(
            self._language_settings(),
            f"""
                    #include <{self._backend_header()}>
                    #include <gridtools/stencil/global_parameter.hpp>
                    #include <gridtools/sid/dimension_to_tuple_like.hpp>
                    {stencil_src}
                    {decl_src}
                    """.strip(),
        )

        module: stages.ProgramSource[
            languages.NanobindSrcL, languages.LanguageWithHeaderFilesSettings
        ] = stages.ProgramSource(
            entry_point=function,
            library_deps=(interface.LibraryDependency(self._library_name(), "master"),),
            source_code=source_code,
            language=self._language(),
            language_settings=self._language_settings(),
        )
        return module

    def _backend_header(self) -> str:
        match self.device_type:
            case core_defs.DeviceType.CUDA:
                return "gridtools/fn/backend/gpu.hpp"
            case core_defs.DeviceType.CPU:
                return "gridtools/fn/backend/naive.hpp"
            case _:
                raise self._not_implemented_for_device_type()

    def _backend_type(self) -> str:
        match self.device_type:
            case core_defs.DeviceType.CUDA:
                return "gridtools::fn::backend::gpu<generated::block_sizes_t>{}"
            case core_defs.DeviceType.CPU:
                return "gridtools::fn::backend::naive{}"
            case _:
                raise self._not_implemented_for_device_type()

    def _language(self) -> type[languages.NanobindSrcL]:
        match self.device_type:
            case core_defs.DeviceType.CUDA:
                return languages.Cuda
            case core_defs.DeviceType.CPU:
                return languages.Cpp
            case _:
                raise self._not_implemented_for_device_type()

    def _language_settings(self) -> languages.LanguageWithHeaderFilesSettings:
        return (
            self.language_settings
            if self.language_settings is not None
            else self._default_language_settings()
        )

    def _library_name(self) -> str:
        match self.device_type:
            case core_defs.DeviceType.CUDA:
                return "gridtools_gpu"
            case core_defs.DeviceType.CPU:
                return "gridtools_cpu"
            case _:
                raise self._not_implemented_for_device_type()

    def _not_implemented_for_device_type(self) -> NotImplementedError:
        return NotImplementedError(
            f"{self.__class__.__name__} is not implemented for "
            f"device type {self.device_type.name}"
        )


class GTFNTranslationStepFactory(factory.Factory):
    class Meta:
        model = GTFNTranslationStep


translate_program_cpu: Final[step_types.TranslationStep] = GTFNTranslationStep()

translate_program_gpu: Final[step_types.TranslationStep] = GTFNTranslationStep(
    device_type=core_defs.DeviceType.CUDA
)
