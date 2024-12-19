# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Final, Optional

import factory
import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.eve import codegen
from gt4py.next import common
from gt4py.next.ffront import fbuiltins
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms import pass_manager
from gt4py.next.otf import languages, stages, step_types, workflow
from gt4py.next.otf.binding import cpp_interface, interface
from gt4py.next.program_processors.codegens.gtfn.codegen import GTFNCodegen, GTFNIMCodegen
from gt4py.next.program_processors.codegens.gtfn.gtfn_ir_to_gtfn_im_ir import GTFN_IM_lowering
from gt4py.next.program_processors.codegens.gtfn.itir_to_gtfn_ir import GTFN_lowering
from gt4py.next.type_system import type_specifications as ts, type_translation


GENERATED_CONNECTIVITY_PARAM_PREFIX = "gt_conn_"


def get_param_description(name: str, type_: Any) -> interface.Parameter:
    return interface.Parameter(name, type_)


@dataclasses.dataclass(frozen=True)
class GTFNTranslationStep(
    workflow.ReplaceEnabledWorkflowMixin[
        stages.CompilableProgram,
        stages.ProgramSource[languages.NanobindSrcL, languages.LanguageWithHeaderFilesSettings],
    ],
    workflow.ChainableWorkflowMixin[
        stages.CompilableProgram,
        stages.ProgramSource[languages.NanobindSrcL, languages.LanguageWithHeaderFilesSettings],
    ],
):
    language_settings: Optional[languages.LanguageWithHeaderFilesSettings] = None
    # TODO replace by more general mechanism, see https://github.com/GridTools/gt4py/issues/1135
    enable_itir_transforms: bool = True
    use_imperative_backend: bool = False
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
    symbolic_domain_sizes: Optional[dict[str, str]] = None

    def _default_language_settings(self) -> languages.LanguageWithHeaderFilesSettings:
        match self.device_type:
            case core_defs.DeviceType.CUDA:
                return languages.LanguageWithHeaderFilesSettings(
                    formatter_key=cpp_interface.CPP_DEFAULT.formatter_key,
                    formatter_style=cpp_interface.CPP_DEFAULT.formatter_style,
                    file_extension="cu",
                    header_extension="cuh",
                )
            case core_defs.DeviceType.ROCM:
                return languages.LanguageWithHeaderFilesSettings(
                    formatter_key=cpp_interface.CPP_DEFAULT.formatter_key,
                    formatter_style=cpp_interface.CPP_DEFAULT.formatter_style,
                    file_extension="hip",
                    header_extension="h",
                )
            case core_defs.DeviceType.CPU:
                return cpp_interface.CPP_DEFAULT
            case _:
                raise self._not_implemented_for_device_type()

    def _process_regular_arguments(
        self,
        program: itir.Program,
        arg_types: tuple[ts.TypeSpec, ...],
        offset_provider_type: common.OffsetProviderType,
    ) -> tuple[list[interface.Parameter], list[str]]:
        parameters: list[interface.Parameter] = []
        arg_exprs: list[str] = []

        for arg_type, program_param in zip(arg_types, program.params, strict=True):
            # parameter
            parameter = get_param_description(program_param.id, arg_type)
            parameters.append(parameter)

            arg = f"std::forward<decltype({parameter.name})>({parameter.name})"

            if isinstance(parameter.type_, ts.FieldType):
                for dim in parameter.type_.dims:
                    if (
                        isinstance(
                            dim, fbuiltins.FieldOffset
                        )  # TODO(havogt): remove support for FieldOffset as Dimension
                        or dim.kind is common.DimensionKind.LOCAL
                    ):
                        # translate sparse dimensions to tuple dtype
                        dim_name = dim.value
                        connectivity = offset_provider_type[dim_name]
                        assert isinstance(connectivity, common.NeighborConnectivityType)
                        size = connectivity.max_neighbors
                        arg = f"gridtools::sid::dimension_to_tuple_like<generated::{dim_name}_t, {size}>({arg})"
            arg_exprs.append(arg)
        return parameters, arg_exprs

    def _process_connectivity_args(
        self, offset_provider_type: common.OffsetProviderType
    ) -> tuple[list[interface.Parameter], list[str]]:
        parameters: list[interface.Parameter] = []
        arg_exprs: list[str] = []

        for name, connectivity_type in offset_provider_type.items():
            if isinstance(connectivity_type, common.NeighborConnectivityType):
                if connectivity_type.dtype.scalar_type not in [np.int32, np.int64]:
                    raise ValueError(
                        "Neighbor table indices must be of type 'np.int32' or 'np.int64'."
                    )

                # parameter
                parameters.append(
                    interface.Parameter(
                        name=GENERATED_CONNECTIVITY_PARAM_PREFIX + name.lower(),
                        type_=ts.FieldType(
                            dims=list(connectivity_type.domain),
                            dtype=type_translation.from_dtype(connectivity_type.dtype),
                        ),
                    )
                )

                # connectivity argument expression
                nbtbl = (
                    f"gridtools::fn::sid_neighbor_table::as_neighbor_table<"
                    f"generated::{connectivity_type.source_dim.value}_t, "
                    f"generated::{name}_t, {connectivity_type.max_neighbors}"
                    f">(std::forward<decltype({GENERATED_CONNECTIVITY_PARAM_PREFIX}{name.lower()})>({GENERATED_CONNECTIVITY_PARAM_PREFIX}{name.lower()}))"
                )
                arg_exprs.append(
                    f"gridtools::hymap::keys<generated::{name}_t>::make_values({nbtbl})"
                )
            elif isinstance(connectivity_type, common.Dimension):
                pass
            else:
                raise AssertionError(
                    f"Expected offset provider type '{name}' to be a 'NeighborConnectivityType' or 'Dimension', "
                    f"got '{type(connectivity_type).__name__}'."
                )

        return parameters, arg_exprs

    def _preprocess_program(
        self,
        program: itir.Program,
        offset_provider: common.OffsetProvider,
    ) -> itir.Program:
        apply_common_transforms = functools.partial(
            pass_manager.apply_common_transforms,
            extract_temporaries=True,
            offset_provider=offset_provider,
            # sid::composite (via hymap) supports assigning from tuple with more elements to tuple with fewer elements
            unconditionally_collapse_tuples=True,
            symbolic_domain_sizes=self.symbolic_domain_sizes,
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
        program: itir.Program,
        offset_provider: common.OffsetProvider,
        column_axis: Optional[common.Dimension],
    ) -> str:
        if self.enable_itir_transforms:
            new_program = self._preprocess_program(program, offset_provider)
        else:
            assert isinstance(program, itir.Program)
            new_program = program

        gtfn_ir = GTFN_lowering.apply(
            new_program,
            offset_provider_type=common.offset_provider_to_type(offset_provider),
            column_axis=column_axis,
        )

        if self.use_imperative_backend:
            gtfn_im_ir = GTFN_IM_lowering().visit(node=gtfn_ir)
            generated_code = GTFNIMCodegen.apply(gtfn_im_ir)
        else:
            generated_code = GTFNCodegen.apply(gtfn_ir)

        return codegen.format_source("cpp", generated_code, style="LLVM")

    def __call__(
        self, inp: stages.CompilableProgram
    ) -> stages.ProgramSource[languages.NanobindSrcL, languages.LanguageWithHeaderFilesSettings]:
        """Generate GTFN C++ code from the ITIR definition."""
        program: itir.Program = inp.data

        # handle regular parameters and arguments of the program (i.e. what the user defined in
        #  the program)
        regular_parameters, regular_args_expr = self._process_regular_arguments(
            program, inp.args.args, inp.args.offset_provider_type
        )

        # handle connectivity parameters and arguments (i.e. what the user provided in the offset
        #  provider)
        connectivity_parameters, connectivity_args_expr = self._process_connectivity_args(
            inp.args.offset_provider_type
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
            inp.args.offset_provider,
            inp.args.column_axis,
        )
        source_code = interface.format_source(
            self._language_settings(),
            f"""
                    #include <{self._backend_header()}>
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
            implicit_domain=inp.data.implicit_domain,
        )
        return module

    def _backend_header(self) -> str:
        match self.device_type:
            case core_defs.DeviceType.CUDA | core_defs.DeviceType.ROCM:
                return "gridtools/fn/backend/gpu.hpp"
            case core_defs.DeviceType.CPU:
                return "gridtools/fn/backend/naive.hpp"
            case _:
                raise self._not_implemented_for_device_type()

    def _backend_type(self) -> str:
        match self.device_type:
            case core_defs.DeviceType.CUDA | core_defs.DeviceType.ROCM:
                return "gridtools::fn::backend::gpu<generated::block_sizes_t>{}"
            case core_defs.DeviceType.CPU:
                return "gridtools::fn::backend::naive{}"
            case _:
                raise self._not_implemented_for_device_type()

    def _language(self) -> type[languages.NanobindSrcL]:
        match self.device_type:
            case core_defs.DeviceType.CUDA:
                return languages.CUDA
            case core_defs.DeviceType.ROCM:
                return languages.HIP
            case core_defs.DeviceType.CPU:
                return languages.CPP
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
            case core_defs.DeviceType.CUDA | core_defs.DeviceType.ROCM:
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


translate_program_cpu: Final[step_types.TranslationStep] = GTFNTranslationStepFactory()

translate_program_gpu: Final[step_types.TranslationStep] = GTFNTranslationStepFactory(
    device_type=core_defs.DeviceType.CUDA
)
