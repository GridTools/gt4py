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
        # pass zero-dimensional fields as scalars
        return arg.asnumpy().item()
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


def _get_shape_args(
    arrays: Mapping[str, dace.data.Array], args: Mapping[str, np.typing.NDArray]
) -> dict[str, int]:
    shape_args: dict[str, int] = {}
    for name, value in args.items():
        for sym, size in zip(arrays[name].shape, value.shape, strict=True):
            if isinstance(sym, dace.symbol):
                assert sym.name not in shape_args
                shape_args[sym.name] = size
            elif sym != size:
                raise RuntimeError(
                    f"Expected shape {arrays[name].shape} for arg {name}, got {value.shape}."
                )
    return shape_args


def _get_stride_args(
    arrays: Mapping[str, dace.data.Array], args: Mapping[str, np.typing.NDArray]
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
                assert sym.name not in stride_args
                stride_args[str(sym)] = stride
            elif sym != stride:
                raise RuntimeError(
                    f"Expected stride {arrays[name].strides} for arg {name}, got {value.strides}."
                )
    return stride_args


def get_sdfg_conn_args(
    sdfg: dace.SDFG,
    offset_provider: gtx_common.OffsetProvider,
    on_gpu: bool,
) -> dict[str, np.typing.NDArray]:
    """
    Extracts the connectivity tables that are used in the sdfg and ensures
    that the memory buffers are allocated for the target device.
    """
    device = dace.DeviceType.GPU if on_gpu else dace.DeviceType.CPU

    connectivity_args = {}
    for offset, connectivity in dace_utils.filter_connectivities(offset_provider).items():
        assert isinstance(connectivity, gtx_common.NeighborTable)
        param = dace_utils.connectivity_identifier(offset)
        if param in sdfg.arrays:
            connectivity_args[param] = _ensure_is_on_device(connectivity.table, device)

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
