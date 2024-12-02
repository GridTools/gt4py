# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import warnings
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import field
from inspect import currentframe, getframeinfo
from pathlib import Path
from typing import Any, ClassVar, Optional

import dace
import numpy as np
from dace.sdfg import utils as sdutils
from dace.transformation.auto import auto_optimize as autoopt

import gt4py.next.iterator.ir as itir
from gt4py.next import common
from gt4py.next.ffront import decorator
from gt4py.next.iterator import transforms as itir_transforms
from gt4py.next.iterator.ir import SymRef
from gt4py.next.iterator.transforms import (
    pass_manager_legacy as legacy_itir_transforms,
    program_to_fencil,
)
from gt4py.next.iterator.type_system import inference as itir_type_inference
from gt4py.next.program_processors.runners.dace_common import utility as dace_utils
from gt4py.next.type_system import type_specifications as ts

from .itir_to_sdfg import ItirToSDFG


def preprocess_program(
    program: itir.FencilDefinition,
    offset_provider_type: common.OffsetProviderType,
    lift_mode: legacy_itir_transforms.LiftMode,
    symbolic_domain_sizes: Optional[dict[str, str]] = None,
    temporary_extraction_heuristics: Optional[
        Callable[[itir.StencilClosure], Callable[[itir.Expr], bool]]
    ] = None,
    unroll_reduce: bool = False,
):
    node = legacy_itir_transforms.apply_common_transforms(
        program,
        common_subexpression_elimination=False,
        force_inline_lambda_args=True,
        lift_mode=lift_mode,
        offset_provider_type=offset_provider_type,
        symbolic_domain_sizes=symbolic_domain_sizes,
        temporary_extraction_heuristics=temporary_extraction_heuristics,
        unroll_reduce=unroll_reduce,
    )

    node = itir_type_inference.infer(node, offset_provider_type=offset_provider_type)

    if isinstance(node, itir.Program):
        fencil_definition = program_to_fencil.program_to_fencil(node)
        tmps = node.declarations
        assert all(isinstance(tmp, itir.Temporary) for tmp in tmps)
    else:
        raise TypeError(f"Expected 'Program', got '{type(node).__name__}'.")

    return fencil_definition, tmps


def build_sdfg_from_itir(
    program: itir.FencilDefinition,
    arg_types: Sequence[ts.TypeSpec],
    offset_provider_type: common.OffsetProviderType,
    auto_optimize: bool = False,
    on_gpu: bool = False,
    column_axis: Optional[common.Dimension] = None,
    lift_mode: legacy_itir_transforms.LiftMode = legacy_itir_transforms.LiftMode.FORCE_INLINE,
    symbolic_domain_sizes: Optional[dict[str, str]] = None,
    temporary_extraction_heuristics: Optional[
        Callable[[itir.StencilClosure], Callable[[itir.Expr], bool]]
    ] = None,
    load_sdfg_from_file: bool = False,
    save_sdfg: bool = True,
    use_field_canonical_representation: bool = True,
) -> dace.SDFG:
    """Translate a Fencil into an SDFG.

    Args:
        program:             The Fencil that should be translated.
        arg_types:           Types of the arguments passed to the fencil.
        offset_provider:     The set of offset providers that should be used.
        auto_optimize:       Apply DaCe's `auto_optimize` heuristic.
        on_gpu:              Performs the translation for GPU, defaults to `False`.
        column_axis:         The column axis to be used, defaults to `None`.
        lift_mode:           Which lift mode should be used, defaults `FORCE_INLINE`.
        symbolic_domain_sizes: Used for generation of liskov bindings when temporaries are enabled.
        load_sdfg_from_file: Allows to read the SDFG from file, instead of generating it, for debug only.
        save_sdfg:           If `True`, the default the SDFG is stored as a file and can be loaded, this allows to skip the lowering step, requires `load_sdfg_from_file` set to `True`.
        use_field_canonical_representation: If `True`,  assume that the fields dimensions are sorted alphabetically.
    """

    sdfg_filename = f"_dacegraphs/gt4py/{program.id}.sdfg"
    if load_sdfg_from_file and Path(sdfg_filename).exists():
        sdfg: dace.SDFG = dace.SDFG.from_file(sdfg_filename)
        sdfg.validate()
        return sdfg

    # visit ITIR and generate SDFG
    program, tmps = preprocess_program(
        program,
        offset_provider_type,
        lift_mode,
        symbolic_domain_sizes,
        temporary_extraction_heuristics,
    )
    sdfg_genenerator = ItirToSDFG(
        list(arg_types),
        offset_provider_type,
        tmps,
        use_field_canonical_representation,
        column_axis,
    )
    sdfg = sdfg_genenerator.visit(program)
    if sdfg is None:
        raise RuntimeError(f"Visit failed for program {program.id}.")

    for nested_sdfg in sdfg.all_sdfgs_recursive():
        if not nested_sdfg.debuginfo:
            _, frameinfo = (
                warnings.warn(
                    f"{nested_sdfg.label} does not have debuginfo. Consider adding them in the corresponding nested sdfg.",
                    stacklevel=2,
                ),
                getframeinfo(currentframe()),  # type: ignore[arg-type]
            )
            nested_sdfg.debuginfo = dace.dtypes.DebugInfo(
                start_line=frameinfo.lineno, end_line=frameinfo.lineno, filename=frameinfo.filename
            )

    # TODO(edopao): remove `inline_loop_blocks` when DaCe transformations support LoopRegion construct
    sdutils.inline_loop_blocks(sdfg)

    # run DaCe transformations to simplify the SDFG
    sdfg.simplify()

    # run DaCe auto-optimization heuristics
    if auto_optimize:
        # TODO: Investigate performance improvement from SDFG specialization with constant symbols,
        #       for array shape and strides, although this would imply JIT compilation.
        symbols: dict[str, int] = {}
        device = dace.DeviceType.GPU if on_gpu else dace.DeviceType.CPU
        sdfg = autoopt.auto_optimize(sdfg, device, symbols=symbols, use_gpu_storage=on_gpu)
    elif on_gpu:
        autoopt.apply_gpu_storage(sdfg)

    if on_gpu:
        sdfg.apply_gpu_transformations()

    # Store the sdfg such that we can later reuse it.
    if save_sdfg:
        sdfg.save(sdfg_filename)

    return sdfg


@dataclasses.dataclass(frozen=True)
class Program(decorator.Program, dace.frontend.python.common.SDFGConvertible):
    """Extension of GT4Py Program implementing the SDFGConvertible interface."""

    sdfg_closure_vars: dict[str, Any] = field(default_factory=dict)

    # Being a ClassVar ensures that in an SDFG with multiple nested GT4Py Programs,
    # there is no name mangling of the connectivity tables used across the nested SDFGs
    # since they share the same memory address.
    connectivity_tables_data_descriptors: ClassVar[
        dict[str, dace.data.Array]
    ] = {}  # symbolically defined

    def __sdfg__(self, *args, **kwargs) -> dace.sdfg.sdfg.SDFG:
        if "dace" not in self.backend.name.lower():  # type: ignore[union-attr]
            raise ValueError("The SDFG can be generated only for the DaCe backend.")

        params = {str(p.id): p.type for p in self.itir.params}
        fields = {str(p.id): p.type for p in self.itir.params if hasattr(p.type, "dims")}
        arg_types = [*params.values()]

        dace_parsed_args = [*args, *kwargs.values()]
        gt4py_program_args = [*params.values()]
        _crosscheck_dace_parsing(dace_parsed_args, gt4py_program_args)

        if self.connectivities is None:
            raise ValueError(
                "[DaCe Orchestration] Connectivities -at compile time- are required to generate the SDFG. Use `with_connectivities` method."
            )
        offset_provider_type = {**self.connectivities, **self._implicit_offset_provider}

        sdfg = self.backend.executor.step.translation.generate_sdfg(  # type: ignore[union-attr]
            self.itir,
            arg_types,
            offset_provider_type=offset_provider_type,
            column_axis=kwargs.get("column_axis", None),
        )
        self.sdfg_closure_vars["sdfg.arrays"] = sdfg.arrays  # use it in __sdfg_closure__

        # Halo exchange related metadata, i.e. gt4py_program_input_fields, gt4py_program_output_fields, offset_providers_per_input_field
        # Add them as dynamic properties to the SDFG

        assert all(
            isinstance(in_field, SymRef)
            for closure in self.itir.closures
            for in_field in closure.inputs
        )  # backend only supports SymRef inputs, not `index` calls
        input_fields = [
            str(in_field.id)  # type: ignore[union-attr]  # ensured by assert
            for closure in self.itir.closures
            for in_field in closure.inputs
            if str(in_field.id) in fields  # type: ignore[union-attr]  # ensured by assert
        ]
        sdfg.gt4py_program_input_fields = {
            in_field: dim
            for in_field in input_fields
            for dim in fields[in_field].dims  # type: ignore[union-attr]
            if dim.kind == common.DimensionKind.HORIZONTAL
        }

        output_fields = []
        for closure in self.itir.closures:
            output = closure.output
            if isinstance(output, itir.SymRef):
                if str(output.id) in fields:
                    output_fields.append(str(output.id))
            else:
                for arg in output.args:
                    if str(arg.id) in fields:  # type: ignore[attr-defined]
                        output_fields.append(str(arg.id))  # type: ignore[attr-defined]
        sdfg.gt4py_program_output_fields = {
            output: dim
            for output in output_fields
            for dim in fields[output].dims  # type: ignore[union-attr]
            if dim.kind == common.DimensionKind.HORIZONTAL
        }

        sdfg.offset_providers_per_input_field = {}
        itir_tmp = legacy_itir_transforms.apply_common_transforms(
            self.itir, offset_provider_type=offset_provider_type
        )
        itir_tmp_fencil = program_to_fencil.program_to_fencil(itir_tmp)
        for closure in itir_tmp_fencil.closures:
            params_shifts = itir_transforms.trace_shifts.trace_stencil(
                closure.stencil, num_args=len(closure.inputs)
            )
            for param, shifts in zip(closure.inputs, params_shifts):
                assert isinstance(
                    param, SymRef
                )  # backend only supports SymRef inputs, not `index` calls
                if not isinstance(param.id, str):
                    continue
                if param.id not in sdfg.gt4py_program_input_fields:
                    continue
                sdfg.offset_providers_per_input_field.setdefault(param.id, []).extend(list(shifts))

        return sdfg

    def __sdfg_closure__(self, reevaluate: Optional[dict[str, str]] = None) -> dict[str, Any]:
        """
        Returns the closure arrays of the SDFG represented by this object
        as a mapping between array name and the corresponding value.

        The connectivity tables are defined symbolically, i.e. table sizes & strides are DaCe symbols.
        The need to define the connectivity tables in the `__sdfg_closure__` arises from the fact that
        the offset providers are not part of GT4Py Program's arguments.
        Keep in mind, that `__sdfg_closure__` is called after `__sdfg__` method.
        """
        offset_provider_type = self.connectivities

        # Define DaCe symbols
        connectivity_table_size_symbols = {
            dace_utils.field_size_symbol_name(
                dace_utils.connectivity_identifier(k), axis
            ): dace.symbol(
                dace_utils.field_size_symbol_name(dace_utils.connectivity_identifier(k), axis)
            )
            for k, v in offset_provider_type.items()  # type: ignore[union-attr]
            for axis in [0, 1]
            if isinstance(v, common.NeighborConnectivityType)
            and dace_utils.connectivity_identifier(k) in self.sdfg_closure_vars["sdfg.arrays"]
        }

        connectivity_table_stride_symbols = {
            dace_utils.field_stride_symbol_name(
                dace_utils.connectivity_identifier(k), axis
            ): dace.symbol(
                dace_utils.field_stride_symbol_name(dace_utils.connectivity_identifier(k), axis)
            )
            for k, v in offset_provider_type.items()  # type: ignore[union-attr]
            for axis in [0, 1]
            if isinstance(v, common.NeighborConnectivityType)
            and dace_utils.connectivity_identifier(k) in self.sdfg_closure_vars["sdfg.arrays"]
        }

        symbols = {**connectivity_table_size_symbols, **connectivity_table_stride_symbols}

        # Define the storage location (e.g. CPU, GPU) of the connectivity tables
        if "storage" not in Program.connectivity_tables_data_descriptors:
            for k, v in offset_provider_type.items():  # type: ignore[union-attr]
                if not isinstance(v, common.NeighborConnectivityType):
                    continue
                if dace_utils.connectivity_identifier(k) in self.sdfg_closure_vars["sdfg.arrays"]:
                    Program.connectivity_tables_data_descriptors["storage"] = (
                        self.sdfg_closure_vars[
                            "sdfg.arrays"
                        ][dace_utils.connectivity_identifier(k)].storage
                    )
                    break

        # Build the closure dictionary
        closure_dict = {}
        for k, v in offset_provider_type.items():  # type: ignore[union-attr]
            conn_id = dace_utils.connectivity_identifier(k)
            if (
                isinstance(v, common.NeighborConnectivityType)
                and conn_id in self.sdfg_closure_vars["sdfg.arrays"]
            ):
                if conn_id not in Program.connectivity_tables_data_descriptors:
                    Program.connectivity_tables_data_descriptors[conn_id] = dace.data.Array(
                        dtype=dace.int64 if v.dtype.scalar_type == np.int64 else dace.int32,
                        shape=[
                            symbols[dace_utils.field_size_symbol_name(conn_id, 0)],
                            symbols[dace_utils.field_size_symbol_name(conn_id, 1)],
                        ],
                        strides=[
                            symbols[dace_utils.field_stride_symbol_name(conn_id, 0)],
                            symbols[dace_utils.field_stride_symbol_name(conn_id, 1)],
                        ],
                        storage=Program.connectivity_tables_data_descriptors["storage"],
                    )
                closure_dict[conn_id] = Program.connectivity_tables_data_descriptors[conn_id]

        return closure_dict

    def __sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
        args = []
        for arg in self.past_stage.past_node.params:
            args.append(arg.id)
        return (args, [])


def _crosscheck_dace_parsing(dace_parsed_args: list[Any], gt4py_program_args: list[Any]) -> bool:
    for dace_parsed_arg, gt4py_program_arg in zip(dace_parsed_args, gt4py_program_args):
        if isinstance(dace_parsed_arg, dace.data.Scalar):
            assert dace_parsed_arg.dtype == dace_utils.as_dace_type(gt4py_program_arg)
        elif isinstance(
            dace_parsed_arg, (bool, int, float, str, np.bool_, np.integer, np.floating, np.str_)
        ):  # compile-time constant scalar
            assert isinstance(gt4py_program_arg, ts.ScalarType)
            if isinstance(dace_parsed_arg, (bool, np.bool_)):
                assert gt4py_program_arg.kind == ts.ScalarKind.BOOL
            elif isinstance(dace_parsed_arg, (int, np.integer)):
                assert gt4py_program_arg.kind in [ts.ScalarKind.INT32, ts.ScalarKind.INT64]
            elif isinstance(dace_parsed_arg, (float, np.floating)):
                assert gt4py_program_arg.kind in [ts.ScalarKind.FLOAT32, ts.ScalarKind.FLOAT64]
            elif isinstance(dace_parsed_arg, (str, np.str_)):
                assert gt4py_program_arg.kind == ts.ScalarKind.STRING
        elif isinstance(dace_parsed_arg, dace.data.Array):
            assert isinstance(gt4py_program_arg, ts.FieldType)
            assert len(dace_parsed_arg.shape) == len(gt4py_program_arg.dims)
            assert dace_parsed_arg.dtype == dace_utils.as_dace_type(gt4py_program_arg.dtype)
        elif isinstance(
            dace_parsed_arg, (dace.data.Structure, dict, OrderedDict)
        ):  # offset_provider
            continue
        else:
            raise ValueError(f"Unresolved case for {dace_parsed_arg} (==, !=) {gt4py_program_arg}")

    return True
