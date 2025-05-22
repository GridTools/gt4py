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
from gt4py.next.otf import arguments, languages, stages, step_types, workflow
from gt4py.next.otf.binding import interface
from gt4py.next.otf.languages import LanguageSettings
from gt4py.next.program_processors.runners.dace import (
    gtir_sdfg,
    transformations as gtx_transformations,
    utils as gtx_dace_utils,
)
from gt4py.next.type_system import type_specifications as ts


def find_constant_symbols(
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


def make_sdfg_async(sdfg: dace.SDFG) -> None:
    """Make an SDFG call immediatly return, without waiting for execution to complete.
    This allows to run the SDFG asynchronously, thus letting GPU kernel execution
    to overlap with host python code.

    The asynchronous call is implemented by the following changes to SDFG:

    - Set all cuda streams to the default stream. This allows to serialize all
      kernels on one GPU queue, avoiding synchronization between different cuda
      streams. Besides, device-to-host copies are performed on the default cuda
      stream, which allows to synchronize the last GPU kernel on the host side.

    - Set `nosync=True` on the states of the top-level SDFG. This flag is used
      by dace codegen to skip emission of `cudaStreamSynchronize()` at the end
      of each state. An exception is made for state transitions where data descriptors
      are accessed on an InterState edge. The typical example is a symbol set to
      the scalar value produced by the previous state, or a condition accessing
      some data descriptor. In this case, we have to wait for the previous state
      to complete.

    """

    has_gpu_code = any(getattr(node, "schedule", False) for node, _ in sdfg.all_nodes_recursive())

    if not has_gpu_code:
        # The async execution mode requires CUDA, therefore it is only available on GPU
        return

    # Loop over all states in the top-level SDFG
    is_nosync_used = False
    for state in sdfg.states():
        for oedge in sdfg.out_edges(state):
            # We check whether the expressions on an InterState edge (symbols assignment
            # and condition for state transition) do access any data descriptor.
            # If so, we break the loop and leave the default `state.nosync=False`.
            if any(
                sym_id in sdfg.arrays for sym_id in oedge.data.condition.get_free_symbols()
            ) or any(
                str(sym) in sdfg.arrays
                for v in oedge.data.assignments.values()
                for sym in dace.symbolic.pystr_to_symbolic(v).free_symbols
            ):
                break
        else:
            # No data descriptor is accessed on the InterState edge, we make the state async.
            state.nosync = True
            is_nosync_used = True

    if is_nosync_used:
        # Emit cuda code for the SDFG to use only the default cuda stream.
        # This allows to serialize all kernels on one GPU queue.
        sdfg.append_init_code(
            "__dace_gpu_set_all_streams(__state, cudaStreamDefault);", location="cuda"
        )


@dataclasses.dataclass(frozen=True)
class DaCeTranslator(
    workflow.ChainableWorkflowMixin[
        stages.CompilableProgram, stages.ProgramSource[languages.SDFG, languages.LanguageSettings]
    ],
    step_types.TranslationStep[languages.SDFG, languages.LanguageSettings],
):
    device_type: core_defs.DeviceType
    auto_optimize: bool
    async_sdfg_call: bool = False
    disable_itir_transforms: bool = False
    disable_field_origin_on_program_arguments: bool = False

    # auto-optimize arguments
    gpu_block_size: tuple[int, int, int] = (32, 8, 1)
    make_persistent: bool = False
    blocking_dim: Optional[common.Dimension] = None
    blocking_size: int = 10

    def generate_sdfg(
        self,
        ir: itir.Program,
        offset_provider: common.OffsetProvider,
        column_axis: Optional[common.Dimension],
    ) -> dace.SDFG:
        if not self.disable_itir_transforms:
            ir = itir_transforms.apply_fieldview_transforms(ir, offset_provider=offset_provider)
        offset_provider_type = common.offset_provider_to_type(offset_provider)
        on_gpu = self.device_type == core_defs.CUPY_DEVICE_TYPE

        # do not store transformation history in SDFG
        with dace.config.set_temporary("store_history", value=False):
            sdfg = gtir_sdfg.build_sdfg_from_gtir(
                ir,
                offset_provider_type,
                column_axis,
                disable_field_origin_on_program_arguments=self.disable_field_origin_on_program_arguments,
            )

            if self.auto_optimize:
                unit_strides_kind = (
                    common.DimensionKind.HORIZONTAL
                    if config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE
                    else None  # let `gt_auto_optimize` select `unit_strides_kind` based on `gpu` argument
                )
                constant_symbols = find_constant_symbols(ir, sdfg, offset_provider_type)
                gtx_transformations.gt_auto_optimize(
                    sdfg,
                    gpu=on_gpu,
                    gpu_block_size=self.gpu_block_size,
                    unit_strides_kind=unit_strides_kind,
                    constant_symbols=constant_symbols,
                    assume_pointwise=True,
                    make_persistent=self.make_persistent,
                    blocking_dim=self.blocking_dim,
                    blocking_size=self.blocking_size,
                )
            elif on_gpu:
                # We run simplify to bring the SDFG into a canonical form that the GPU transformations
                # can handle. This is a workaround for an issue with scalar expressions that are
                # promoted to symbolic expressions and computed on the host (CPU), but the intermediate
                # result is written to a GPU global variable (https://github.com/spcl/dace/issues/1773).
                gtx_transformations.gt_simplify(sdfg)
                gtx_transformations.gt_gpu_transformation(sdfg, try_removing_trivial_maps=True)

        if on_gpu and self.async_sdfg_call:
            make_sdfg_async(sdfg)

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
        )

        arg_types = tuple(
            arg.type_ if isinstance(arg, arguments.StaticArg) else arg for arg in inp.args.args
        )

        program_parameters = tuple(
            interface.Parameter(param.id, arg_type)
            for param, arg_type in zip(program.params, arg_types)
        )

        module: stages.ProgramSource[languages.SDFG, languages.LanguageSettings] = (
            stages.ProgramSource(
                entry_point=interface.Function(program.id, program_parameters),
                source_code=sdfg.to_json(),
                library_deps=tuple(),
                language=languages.SDFG,
                language_settings=languages.LanguageSettings(
                    formatter_key="", formatter_style="", file_extension="sdfg"
                ),
                implicit_domain=inp.data.implicit_domain,
            )
        )
        return module


class DaCeTranslationStepFactory(factory.Factory):
    class Meta:
        model = DaCeTranslator
