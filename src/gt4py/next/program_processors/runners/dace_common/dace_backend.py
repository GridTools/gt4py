# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Iterable

import dace
import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common as gtx_common, utils as gtx_utils

from . import utility as dace_utils


try:
    import cupy as cp
except ImportError:
    cp = None


def _convert_arg(arg: Any, sdfg_param: str, use_field_canonical_representation: bool) -> Any:
    if not isinstance(arg, gtx_common.Field):
        return arg
    if len(arg.domain.dims) == 0:
        # Pass zero-dimensional fields as scalars.
        return arg.as_scalar()
    # field domain offsets are not supported
    non_zero_offsets = [
        (dim, dim_range)
        for dim, dim_range in zip(arg.domain.dims, arg.domain.ranges, strict=True)
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
    sorted_dims = dace_utils.get_sorted_dims(arg.domain.dims)
    ndim = len(sorted_dims)
    dim_indices = [dim_index for dim_index, _ in sorted_dims]
    if isinstance(arg.ndarray, np.ndarray):
        return np.moveaxis(arg.ndarray, range(ndim), dim_indices)
    else:
        assert cp is not None and isinstance(arg.ndarray, cp.ndarray)
        return cp.moveaxis(arg.ndarray, range(ndim), dim_indices)


def _get_args(
    sdfg: dace.SDFG, args: Sequence[Any], use_field_canonical_representation: bool
) -> dict[str, Any]:
    sdfg_params: Sequence[str] = sdfg.arg_names
    flat_args: Iterable[Any] = gtx_utils.flatten_nested_tuple(tuple(args))
    return {
        sdfg_param: _convert_arg(arg, sdfg_param, use_field_canonical_representation)
        for sdfg_param, arg in zip(sdfg_params, flat_args, strict=True)
    }


def _ensure_is_on_device(
    connectivity_arg: core_defs.NDArrayObject, device: dace.dtypes.DeviceType
) -> core_defs.NDArrayObject:
    if device == dace.dtypes.DeviceType.GPU:
        if not isinstance(connectivity_arg, cp.ndarray):
            warnings.warn(
                "Copying connectivity to device. For performance make sure connectivity is provided on device.",
                stacklevel=2,
            )
            return cp.asarray(connectivity_arg)
    return connectivity_arg


def _get_shape_args(
    arrays: Mapping[str, dace.data.Array], args: Mapping[str, core_defs.NDArrayObject]
) -> dict[str, int]:
    shape_args: dict[str, int] = {}
    for name, value in args.items():
        for sym, size in zip(arrays[name].shape, value.shape, strict=True):
            if isinstance(sym, dace.symbol):
                if sym.name not in shape_args:
                    shape_args[sym.name] = size
                elif shape_args[sym.name] != size:
                    # The same shape symbol is used by all fields of a tuple, because the current assumption is that all fields
                    # in a tuple have the same dimensions and sizes. Therefore, this if-branch only exists to ensure that array
                    # size (i.e. the value assigned to the shape symbol) is the same for all fields in a tuple.
                    # TODO(edopao): change to `assert sym.name not in shape_args` to ensure that shape symbols are unique,
                    # once the assumption on tuples is removed.
                    raise ValueError(
                        f"Expected array size {sym.name} for arg {name} to be {shape_args[sym.name]}, got {size}."
                    )
            elif sym != size:
                raise ValueError(
                    f"Expected shape {arrays[name].shape} for arg {name}, got {value.shape}."
                )
    return shape_args


def _get_stride_args(
    arrays: Mapping[str, dace.data.Array], args: Mapping[str, core_defs.NDArrayObject]
) -> dict[str, int]:
    stride_args = {}
    for name, value in args.items():
        for sym, stride_size in zip(arrays[name].strides, value.strides, strict=True):
            stride, remainder = divmod(stride_size, value.itemsize)
            if remainder != 0:
                raise ValueError(
                    f"Stride ({stride_size} bytes) for argument '{sym}' must be a multiple of item size ({value.itemsize} bytes)."
                )
            if isinstance(sym, dace.symbol):
                if sym.name not in stride_args:
                    stride_args[str(sym)] = stride
                elif stride_args[sym.name] != stride:
                    # See above comment in `_get_shape_args`, same for stride symbols of fields in a tuple.
                    # TODO(edopao): change to `assert sym.name not in stride_args` to ensure that stride symbols are unique,
                    # once the assumption on tuples is removed.
                    raise ValueError(
                        f"Expected array stride {sym.name} for arg {name} to be {stride_args[sym.name]}, got {stride}."
                    )
            elif sym != stride:
                raise ValueError(
                    f"Expected stride {arrays[name].strides} for arg {name}, got {value.strides}."
                )
    return stride_args


def get_sdfg_conn_args(
    sdfg: dace.SDFG,
    offset_provider: gtx_common.OffsetProvider,
    on_gpu: bool,
) -> dict[str, core_defs.NDArrayObject]:
    """
    Extracts the connectivity tables that are used in the sdfg and ensures
    that the memory buffers are allocated for the target device.
    """
    device = dace.DeviceType.GPU if on_gpu else dace.DeviceType.CPU

    connectivity_args = {}
    for offset, connectivity in offset_provider.items():
        if gtx_common.is_neighbor_table(connectivity):
            param = dace_utils.connectivity_identifier(offset)
            if param in sdfg.arrays:
                connectivity_args[param] = _ensure_is_on_device(connectivity.ndarray, device)

    return connectivity_args


def get_sdfg_args(
    sdfg: dace.SDFG,
    *args: Any,
    check_args: bool = False,
    on_gpu: bool = False,
    use_field_canonical_representation: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Extracts the arguments needed to call the SDFG.

    This function can handle the same arguments that are passed to dace runner.

    Args:
        sdfg:               The SDFG for which we want to get the arguments.
    """
    offset_provider = kwargs["offset_provider"]

    dace_args = _get_args(sdfg, args, use_field_canonical_representation)
    dace_field_args = {n: v for n, v in dace_args.items() if not np.isscalar(v)}
    dace_conn_args = get_sdfg_conn_args(sdfg, offset_provider, on_gpu)
    dace_shapes = _get_shape_args(sdfg.arrays, dace_field_args)
    dace_conn_shapes = _get_shape_args(sdfg.arrays, dace_conn_args)
    dace_strides = _get_stride_args(sdfg.arrays, dace_field_args)
    dace_conn_strides = _get_stride_args(sdfg.arrays, dace_conn_args)
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
