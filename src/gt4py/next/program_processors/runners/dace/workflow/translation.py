# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Optional

import dace
import factory

from gt4py._core import definitions as core_defs
from gt4py.next import common, config
from gt4py.next.iterator import ir as itir, transforms as itir_transforms
from gt4py.next.otf import languages, stages, step_types, workflow
from gt4py.next.otf.binding import interface
from gt4py.next.otf.languages import LanguageSettings
from gt4py.next.program_processors.runners.dace import (
    gtir_sdfg,
    transformations as gtx_transformations,
    utils as gtx_dace_utils,
)
from gt4py.next.type_system import type_specifications as ts


def _find_constant_symbols(
    ir: itir.Program,
    sdfg: dace.SDFG,
    offset_provider_type: common.OffsetProviderType,
) -> dict[str, int]:
    """Helper function to find symbols to replace with constant values."""
    constant_symbols: dict[str, int] = {}
    if config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE:
        # Search the stride symbols corresponding to the horizontal dimension
        for p in ir.params:
            if isinstance(p.type, ts.FieldType):
                dims = p.type.dims
                if len(dims) == 0:
                    continue
                elif len(dims) == 1:
                    dim_index = 0
                elif len(dims) == 2:
                    dim_index = 0 if dims[0].kind == common.DimensionKind.HORIZONTAL else 1
                else:
                    raise ValueError(f"Unsupported field with dims={dims}.")
                stride_name = gtx_dace_utils.field_stride_symbol_name(p.id, dim_index)
                constant_symbols[stride_name] = 1
        # Same for connectivity tables, for which the first dimension is always horizontal
        for conn, desc in sdfg.arrays.items():
            if gtx_dace_utils.is_connectivity_identifier(conn, offset_provider_type):
                assert not desc.transient
                stride_name = gtx_dace_utils.field_stride_symbol_name(conn, 0)
                constant_symbols[stride_name] = 1

    return constant_symbols


@dataclasses.dataclass(frozen=True)
class DaCeTranslator(
    workflow.ChainableWorkflowMixin[
        stages.CompilableProgram, stages.ProgramSource[languages.SDFG, languages.LanguageSettings]
    ],
    step_types.TranslationStep[languages.SDFG, languages.LanguageSettings],
):
    device_type: core_defs.DeviceType
    auto_optimize: bool
    disable_itir_transforms: bool = False
    disable_field_origin_on_program_arguments: bool = False

    def _language_settings(self) -> languages.LanguageSettings:
        return languages.LanguageSettings(
            formatter_key="", formatter_style="", file_extension="sdfg"
        )

    def generate_sdfg(
        self,
        ir: itir.Program,
        offset_provider: common.OffsetProvider,
        column_axis: Optional[common.Dimension],
        auto_opt: bool,
        on_gpu: bool,
    ) -> dace.SDFG:
        if not self.disable_itir_transforms:
            ir = itir_transforms.apply_fieldview_transforms(ir, offset_provider=offset_provider)
        offset_provider_type = common.offset_provider_to_type(offset_provider)
        sdfg = gtir_sdfg.build_sdfg_from_gtir(
            ir,
            offset_provider_type,
            column_axis,
            disable_field_origin_on_program_arguments=self.disable_field_origin_on_program_arguments,
        )

        if auto_opt:
            unit_strides_kind = (
                common.DimensionKind.HORIZONTAL
                if config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE
                else None  # let `gt_auto_optimize` select `unit_strides_kind` based on `gpu` argument
            )
            constant_symbols = _find_constant_symbols(ir, sdfg, offset_provider_type)
            gtx_transformations.gt_auto_optimize(
                sdfg,
                gpu=on_gpu,
                gpu_block_size=None,  # TODO: set optimal block size
                unit_strides_kind=unit_strides_kind,
                constant_symbols=constant_symbols,
                assume_pointwise=True,
                make_persistent=False,
            )
        elif on_gpu:
            # We run simplify to bring the SDFG into a canonical form that the gpu transformations
            # can handle. This is a workaround for an issue with scalar expressions that are
            # promoted to symbolic expressions and computed on the host (CPU), but the intermediate
            # result is written to a GPU global variable (https://github.com/spcl/dace/issues/1773).
            gtx_transformations.gt_simplify(sdfg)
            gtx_transformations.gt_gpu_transformation(sdfg, try_removing_trivial_maps=True)

        return sdfg

    def __call__(
        self, inp: stages.CompilableProgram
    ) -> stages.ProgramSource[languages.SDFG, LanguageSettings]:
        """Generate DaCe SDFG file from the GTIR definition."""
        program: itir.Program = inp.data
        assert isinstance(program, itir.Program)

        sdfg = self.generate_sdfg(
            program,
            inp.args.offset_provider,  # TODO(havogt): should be offset_provider_type once the transformation don't require run-time info
            inp.args.column_axis,
            auto_opt=self.auto_optimize,
            on_gpu=(self.device_type == core_defs.CUPY_DEVICE_TYPE),
        )

        param_types = tuple(
            interface.Parameter(param, arg_type)
            for param, arg_type in zip(sdfg.arg_names, inp.args.args)
        )

        module: stages.ProgramSource[languages.SDFG, languages.LanguageSettings] = (
            stages.ProgramSource(
                entry_point=interface.Function(program.id, param_types),
                source_code=sdfg.to_json(),
                library_deps=tuple(),
                language=languages.SDFG,
                language_settings=self._language_settings(),
                implicit_domain=inp.data.implicit_domain,
            )
        )
        return module


class DaCeTranslationStepFactory(factory.Factory):
    class Meta:
        model = DaCeTranslator
