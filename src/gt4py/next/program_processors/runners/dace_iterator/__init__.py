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
import warnings
from inspect import currentframe, getframeinfo
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import dace
import numpy as np
from dace.sdfg import utils as sdutils
from dace.transformation.auto import auto_optimize as autoopt

import gt4py.next.iterator.ir as itir
from gt4py.next import common
from gt4py.next.iterator import transforms as itir_transforms
from gt4py.next.type_system import type_translation

from .itir_to_sdfg import ItirToSDFG
from .utility import connectivity_identifier, filter_neighbor_tables, get_sorted_dims


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
    unroll_reduce: bool = False,
):
    node = itir_transforms.apply_common_transforms(
        program,
        common_subexpression_elimination=False,
        force_inline_lambda_args=True,
        lift_mode=lift_mode,
        offset_provider=offset_provider,
        unroll_reduce=unroll_reduce,
    )

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

    neighbor_tables = filter_neighbor_tables(offset_provider)
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
    *args,
    offset_provider: dict[str, Any],
    auto_optimize: bool = False,
    on_gpu: bool = False,
    column_axis: Optional[common.Dimension] = None,
    lift_mode: itir_transforms.LiftMode = itir_transforms.LiftMode.FORCE_INLINE,
    load_sdfg_from_file: bool = False,
    save_sdfg: bool = True,
    use_field_canonical_representation: bool = True,
) -> dace.SDFG:
    """Translate a Fencil into an SDFG.

    Args:
        program:             The Fencil that should be translated.
        *args:               Arguments for which the fencil should be called.
        offset_provider:     The set of offset providers that should be used.
        auto_optimize:       Apply DaCe's `auto_optimize` heuristic.
        on_gpu:              Performs the translation for GPU, defaults to `False`.
        column_axis:         The column axis to be used, defaults to `None`.
        lift_mode:           Which lift mode should be used, defaults `FORCE_INLINE`.
        load_sdfg_from_file: Allows to read the SDFG from file, instead of generating it, for debug only.
        save_sdfg:           If `True`, the default the SDFG is stored as a file and can be loaded, this allows to skip the lowering step, requires `load_sdfg_from_file` set to `True`.
        use_field_canonical_representation: If `True`,  assume that the fields dimensions are sorted alphabetically.

    Notes:
        Currently only the `FORCE_INLINE` liftmode is supported and the value of `lift_mode` is ignored.
    """

    sdfg_filename = f"_dacegraphs/gt4py/{program.id}.sdfg"
    if load_sdfg_from_file and Path(sdfg_filename).exists():
        sdfg: dace.SDFG = dace.SDFG.from_file(sdfg_filename)
        sdfg.validate()
        return sdfg

    arg_types = [type_translation.from_value(arg) for arg in args]

    # visit ITIR and generate SDFG
    program, tmps = preprocess_program(program, offset_provider, lift_mode)
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
