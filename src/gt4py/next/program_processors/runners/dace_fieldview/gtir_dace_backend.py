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
from collections.abc import Mapping, Sequence
from typing import Any

import dace
import numpy as np

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace_fieldview import utility as dace_fieldview_util


try:
    import cupy as cp
except ImportError:
    cp = None


def convert_arg(arg: Any, sdfg_param: str) -> Any:
    if not isinstance(arg, gtx_common.Field):
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
    return arg.ndarray


def get_args(sdfg: dace.SDFG, args: Sequence[Any]) -> dict[str, Any]:
    sdfg_params: Sequence[str] = sdfg.arg_names
    return {sdfg_param: convert_arg(arg, sdfg_param) for sdfg_param, arg in zip(sdfg_params, args)}


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
    neighbor_tables: Mapping[str, gtx_common.NeighborTable], device: dace.dtypes.DeviceType
) -> dict[str, Any]:
    return {
        dace_fieldview_util.connectivity_identifier(offset): _ensure_is_on_device(
            offset_provider.table, device
        )
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
    *args: Any,
    check_args: bool = False,
    on_gpu: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Extracts the arguments needed to call the SDFG.

    This function can handle the same arguments that are passed to dace runner.

    Args:
        sdfg:               The SDFG for which we want to get the arguments.
    """
    offset_provider = kwargs["offset_provider"]

    neighbor_tables: dict[str, gtx_common.NeighborTable] = {}
    for offset, connectivity in dace_fieldview_util.filter_connectivities(offset_provider).items():
        assert isinstance(connectivity, gtx_common.NeighborTable)
        neighbor_tables[offset] = connectivity
    device = dace.DeviceType.GPU if on_gpu else dace.DeviceType.CPU

    dace_args = get_args(sdfg, args)
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
