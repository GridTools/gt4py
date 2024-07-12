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
import dataclasses
import warnings
from collections.abc import Callable, Mapping, Sequence
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
from gt4py.next.iterator.type_system import inference as itir_type_inference
from gt4py.next.type_system import type_specifications as ts

from .itir_to_sdfg import ItirToSDFG
from .utility import connectivity_identifier, filter_connectivities, get_sorted_dims


try:
    import cupy as cp
except ImportError:
    cp = None


def convert_arg(arg: Any, sdfg_param: str, use_field_canonical_representation: bool):
    if not isinstance(arg, common.Field):
        return arg
    # field domain offsets are not supported
    non_zero_offsets = [
        (dim, dim_range)
        for dim, dim_range in zip(arg.domain.dims, arg.domain.ranges)
        if dim_range.start != 0
    ]
    if non_zero_offsets:
        dim, dim_range = non_zero_offsets[0]
        raise RuntimeError(
            f"Field '{sdfg_param}' passed as array slice with offset {dim_range.start} on dimension {dim.value}."
        )
    if not use_field_canonical_representation:
        return arg.ndarray
    # the canonical representation requires alphabetical ordering of the dimensions in field domain definition
    sorted_dims = get_sorted_dims(arg.domain.dims)
    ndim = len(sorted_dims)
    dim_indices = [dim_index for dim_index, _ in sorted_dims]
    if isinstance(arg.ndarray, np.ndarray):
        return np.moveaxis(arg.ndarray, range(ndim), dim_indices)
    else:
        assert cp is not None and isinstance(arg.ndarray, cp.ndarray)
        return cp.moveaxis(arg.ndarray, range(ndim), dim_indices)


def preprocess_program(
    program: itir.FencilDefinition,
    offset_provider: Mapping[str, Any],
    lift_mode: itir_transforms.LiftMode,
    symbolic_domain_sizes: Optional[dict[str, str]] = None,
    temporary_extraction_heuristics: Optional[
        Callable[[itir.StencilClosure], Callable[[itir.Expr], bool]]
    ] = None,
    unroll_reduce: bool = False,
):
    node = itir_transforms.apply_common_transforms(
        program,
        common_subexpression_elimination=False,
        force_inline_lambda_args=True,
        lift_mode=lift_mode,
        offset_provider=offset_provider,
        symbolic_domain_sizes=symbolic_domain_sizes,
        temporary_extraction_heuristics=temporary_extraction_heuristics,
        unroll_reduce=unroll_reduce,
    )

    node = itir_type_inference.infer(node, offset_provider=offset_provider)

    if isinstance(node, itir_transforms.global_tmps.FencilWithTemporaries):
        fencil_definition = node.fencil
        tmps = node.tmps

    elif isinstance(node, itir.FencilDefinition):
        fencil_definition = node
        tmps = []

    else:
        raise TypeError(
            f"Expected 'FencilDefinition' or 'FencilWithTemporaries', got '{type(program).__name__}'."
        )

    return fencil_definition, tmps


def get_args(
    sdfg: dace.SDFG, args: Sequence[Any], use_field_canonical_representation: bool
) -> dict[str, Any]:
    sdfg_params: Sequence[str] = sdfg.arg_names
    return {
        sdfg_param: convert_arg(arg, sdfg_param, use_field_canonical_representation)
        for sdfg_param, arg in zip(sdfg_params, args)
    }


def _ensure_is_on_device(
    connectivity_arg: np.typing.NDArray, device: dace.dtypes.DeviceType
) -> np.typing.NDArray:
    if device == dace.dtypes.DeviceType.GPU:
        if not isinstance(connectivity_arg, cp.ndarray):
            warnings.warn(
                "Copying connectivity to device. For performance make sure connectivity is provided on device.",
                stacklevel=2,
            )
            return cp.asarray(connectivity_arg)
    return connectivity_arg


def get_connectivity_args(
    neighbor_tables: Mapping[str, common.NeighborTable], device: dace.dtypes.DeviceType
) -> dict[str, Any]:
    return {
        connectivity_identifier(offset): _ensure_is_on_device(offset_provider.table, device)
        for offset, offset_provider in neighbor_tables.items()
    }


def get_shape_args(
    arrays: Mapping[str, dace.data.Array], args: Mapping[str, Any]
) -> Mapping[str, int]:
    shape_args: dict[str, int] = {}
    for name, value in args.items():
        for sym, size in zip(arrays[name].shape, value.shape):
            if isinstance(sym, dace.symbol):
                assert sym.name not in shape_args
                shape_args[sym.name] = size
            elif sym != size:
                raise RuntimeError(
                    f"Expected shape {arrays[name].shape} for arg {name}, got {value.shape}."
                )
    return shape_args


def get_stride_args(
    arrays: Mapping[str, dace.data.Array], args: Mapping[str, Any]
) -> Mapping[str, int]:
    stride_args = {}
    for name, value in args.items():
        for sym, stride_size in zip(arrays[name].strides, value.strides):
            stride, remainder = divmod(stride_size, value.itemsize)
            if remainder != 0:
                raise ValueError(
                    f"Stride ({stride_size} bytes) for argument '{sym}' must be a multiple of item size ({value.itemsize} bytes)."
                )
            if isinstance(sym, dace.symbol):
                assert sym.name not in stride_args
                stride_args[str(sym)] = stride
            elif sym != stride:
                raise RuntimeError(
                    f"Expected stride {arrays[name].strides} for arg {name}, got {value.strides}."
                )
    return stride_args


def get_sdfg_args(
    sdfg: dace.SDFG,
    *args,
    check_args: bool = False,
    on_gpu: bool = False,
    use_field_canonical_representation: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Extracts the arguments needed to call the SDFG.

    This function can handle the same arguments that are passed to dace runner.

    Args:
        sdfg:               The SDFG for which we want to get the arguments.
    """
    offset_provider = kwargs["offset_provider"]

    neighbor_tables: dict[str, common.NeighborTable] = {}
    for offset, connectivity in filter_connectivities(offset_provider).items():
        assert isinstance(connectivity, common.NeighborTable)
        neighbor_tables[offset] = connectivity
    device = dace.DeviceType.GPU if on_gpu else dace.DeviceType.CPU

    dace_args = get_args(sdfg, args, use_field_canonical_representation)
    dace_field_args = {n: v for n, v in dace_args.items() if not np.isscalar(v)}
    dace_conn_args = get_connectivity_args(neighbor_tables, device)
    # keep only connectivity tables that are used in the sdfg
    dace_conn_args = {n: v for n, v in dace_conn_args.items() if n in sdfg.arrays}
    dace_shapes = get_shape_args(sdfg.arrays, dace_field_args)
    dace_conn_shapes = get_shape_args(sdfg.arrays, dace_conn_args)
    dace_strides = get_stride_args(sdfg.arrays, dace_field_args)
    dace_conn_strides = get_stride_args(sdfg.arrays, dace_conn_args)
    all_args = {
        **dace_args,
        **dace_conn_args,
        **dace_shapes,
        **dace_conn_shapes,
        **dace_strides,
        **dace_conn_strides,
    }

    if check_args:
        # return only arguments expected in SDFG signature (note hat `signature_arglist` takes time)
        sdfg_sig = sdfg.signature_arglist(with_types=False)
        return {key: all_args[key] for key in sdfg_sig}

    return all_args


def build_sdfg_from_itir(
    program: itir.FencilDefinition,
    arg_types: list[ts.TypeSpec],
    offset_provider: dict[str, Any],
    auto_optimize: bool = False,
    on_gpu: bool = False,
    column_axis: Optional[common.Dimension] = None,
    lift_mode: itir_transforms.LiftMode = itir_transforms.LiftMode.FORCE_INLINE,
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
        program, offset_provider, lift_mode, symbolic_domain_sizes, temporary_extraction_heuristics
    )
    sdfg_genenerator = ItirToSDFG(
        arg_types, offset_provider, tmps, use_field_canonical_representation, column_axis
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

    # Defined as class variable, otherwise the same connectivity table -name & data descriptor- across multiple GT4Py Programs
    # will be considered as a different table (with subsequent name mangling). Therefore, the memory address of the table's data descriptor needs to be the same.
    # This affects the case where a DaCe.Program calls multiple GT4Py Programs (fused SDFG).
    connectivity_tables_data_descriptors: ClassVar[dict[str, Any]] = {}

    def __sdfg__(self, *args, **kwargs) -> dace.sdfg.sdfg.SDFG:
        params = {str(p.id): p.type for p in self.itir.params}
        fields = {str(p.id): p.type for p in self.itir.params if hasattr(p.type, "dims")}
        arg_types = list(params.values())

        # Do this because DaCe converts the offset_provider to an OrderedDict with StringLiteral keys
        offset_provider = {str(k): v for k, v in kwargs.get("offset_provider", {}).items()}
        self.sdfg_closure_vars["offset_provider"] = offset_provider

        sdfg = self.backend.executor.otf_workflow.step.translation.generate_sdfg(  # type: ignore[union-attr]
            self.itir,
            arg_types,
            offset_provider=offset_provider,
            column_axis=kwargs.get("column_axis", None),
        )

        self.sdfg_closure_vars["sdfg.arrays"] = sdfg.arrays

        # TODO(kotsaloscv): Keep halo exchange related metadata here?
        # Could possibly be computed directly from the SDFG.

        input_fields = [
            str(inpt.id)
            for closure in self.itir.closures
            for inpt in closure.inputs
            if str(inpt.id) in sdfg.arrays
        ]
        sdfg.gt4py_program_input_fields = {
            inpt: dim
            for inpt in input_fields
            for dim in fields[inpt].dims  # type: ignore[union-attr]
            if dim.kind == common.DimensionKind.HORIZONTAL
        }

        output_fields = []
        for closure in self.itir.closures:
            output = closure.output
            if isinstance(output, itir.SymRef):
                if str(output.id) in sdfg.arrays:
                    output_fields.append(str(output.id))
            else:
                for arg in output.args:
                    if str(arg.id) in sdfg.arrays:  # type: ignore[attr-defined]
                        output_fields.append(str(arg.id))  # type: ignore[attr-defined]
        sdfg.gt4py_program_output_fields = {
            output: dim
            for output in output_fields
            for dim in fields[output].dims  # type: ignore[union-attr]
            if dim.kind == common.DimensionKind.HORIZONTAL
        }

        sdfg.offset_providers_per_input_field = {}
        itir_tmp = itir_transforms.apply_common_transforms(
            self.itir, offset_provider=offset_provider
        )
        for closure in itir_tmp.closures:  # type: ignore[union-attr]
            shifts = itir_transforms.trace_shifts.TraceShifts.apply(closure)
            for k, v in shifts.items():
                if k not in sdfg.gt4py_program_input_fields:
                    continue
                sdfg.offset_providers_per_input_field.setdefault(k, []).extend(list(v))

        sdfg.gt4py_program_kwargs = {
            key: value for key, value in kwargs.items() if key != "offset_provider"
        }

        return sdfg

    def __sdfg_closure__(self, reevaluate: Optional[dict[str, str]] = None) -> dict[str, Any]:
        """
        The connectivity tables are defined here symbolically (runtime args and not compile time).
        They need to be defined in `__sdfg_closure__` due to the fact that they are not part of GT4Py Program's arguments.
        Actually, they are already defined in the construction of the sdfg, but some symbols are missing because they are not part of GT4Py Program's arguments -parsing issue-.
        Here, we define them symbolically (along with the correct dtypes), and let DaCe fill the missing symbols.
        Keep in mind, that __sdfg_closure__ is called after __sdfg__.
        """
        symbols = {
            f"__{connectivity_identifier(k)}_{sym}": dace.symbol(
                f"__{connectivity_identifier(k)}_{sym}"
            )
            for k, v in self.sdfg_closure_vars.get("offset_provider", {}).items()
            for sym in ["size_0", "size_1", "stride_0", "stride_1"]
            if hasattr(v, "table")
            and connectivity_identifier(k) in self.sdfg_closure_vars.get("sdfg.arrays", {})
        }

        closure_dict = {}
        for k, v in self.sdfg_closure_vars.get("offset_provider", {}).items():
            conn_id = connectivity_identifier(k)
            if hasattr(v, "table") and conn_id in self.sdfg_closure_vars.get("sdfg.arrays", {}):
                if conn_id not in Program.connectivity_tables_data_descriptors:
                    Program.connectivity_tables_data_descriptors[conn_id] = dace.data.Array(
                        dtype=dace.int64 if v.table.dtype == np.int64 else dace.int32,
                        shape=[symbols[f"__{conn_id}_size_0"], symbols[f"__{conn_id}_size_1"]],
                        strides=[
                            symbols[f"__{conn_id}_stride_0"],
                            symbols[f"__{conn_id}_stride_1"],
                        ],
                    )
                closure_dict[conn_id] = Program.connectivity_tables_data_descriptors[conn_id]

        return closure_dict

    def __sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
        args = []
        for arg in self.past_stage.past_node.params:
            args.append(arg.id)
        return (args, [])
