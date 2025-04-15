# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import dace
import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common as gtx_common, field_utils

from . import utils as gtx_dace_utils


try:
    import cupy as cp
except ImportError:
    cp = None


def _convert_arg(arg: Any) -> tuple[Any, Optional[gtx_common.Domain]]:
    if not isinstance(arg, gtx_common.Field):
        return arg, None
    if len(arg.domain.dims) == 0:
        # Pass zero-dimensional fields as scalars.
        return arg.as_scalar(), None
    return arg.ndarray, arg.domain


def _get_args(sdfg: dace.SDFG, args: Sequence[Any]) -> dict[str, Any]:
    sdfg_params: Sequence[str] = sdfg.arg_names
    sdfg_arguments = {}
    range_symbols: dict[str, int] = {}
    for sdfg_param, arg in zip(sdfg_params, args, strict=True):
        sdfg_arg, domain = _convert_arg(arg)
        sdfg_arguments[sdfg_param] = sdfg_arg
        if domain:
            assert gtx_common.Domain.is_finite(domain)
            range_symbols |= {
                gtx_dace_utils.range_start_symbol(sdfg_param, i): r.start
                for i, r in enumerate(domain.ranges)
            }
            range_symbols |= {
                gtx_dace_utils.range_stop_symbol(sdfg_param, i): r.stop
                for i, r in enumerate(domain.ranges)
            }
    # sanity check in case range symbols are passed as explicit program arguments
    for range_symbol, value in range_symbols.items():
        if (sdfg_arg := sdfg_arguments.get(range_symbol, None)) is not None:
            if sdfg_arg != value:
                raise ValueError(
                    f"Received program argument {range_symbol} with value {sdfg_arg}, expected {value}."
                )
    return sdfg_arguments | range_symbols


def _get_shape_args(
    arrays: Mapping[str, dace.data.Array], args: Mapping[str, core_defs.NDArrayObject]
) -> dict[str, int]:
    shape_args: dict[str, int] = {}
    for name, value in args.items():
        for sym, size in zip(arrays[name].shape, value.shape, strict=True):
            if isinstance(sym, dace.symbol):
                assert sym.name not in shape_args
                shape_args[sym.name] = size
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
                assert sym.name not in stride_args
                stride_args[sym.name] = stride
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
    connectivity_args = {}
    for offset, connectivity in offset_provider.items():
        if gtx_common.is_neighbor_table(connectivity):
            param = gtx_dace_utils.connectivity_identifier(offset)
            if param in sdfg.arrays:
                assert field_utils.verify_device_field_type(
                    connectivity,
                    core_defs.CUPY_DEVICE_TYPE if on_gpu else core_defs.DeviceType.CPU,  # type: ignore[arg-type] # if we are `on_gpu` we expect `CUPY_DEVICE_TYPE` to be not `None`
                )
                connectivity_args[param] = connectivity.ndarray

    return connectivity_args


def get_sdfg_args(
    sdfg: dace.SDFG,
    offset_provider: gtx_common.OffsetProvider,
    *args: Any,
    check_args: bool = False,
    on_gpu: bool = False,
) -> dict[str, Any]:
    """Extracts the arguments needed to call the SDFG.

    This function can handle the arguments that are passed to the dace runner
    and that end up in the decoration stage of the dace backend workflow.

    Args:
        sdfg:               The SDFG for which we want to get the arguments.
        offset_provider:    The offset provider.
        args:               The list of arguments passed to the dace runner.
        check_args:         If True, return only the arguments that are expected
                            according to the SDFG signature.
        on_gpu:             If True, this method ensures that the arrays for the
                            connectivity tables are allocated in GPU memory.

    Returns:
        A dictionary of keyword arguments to be passed in the SDFG call.
    """

    dace_args = _get_args(sdfg, args)
    dace_field_args = {n: v for n, v in dace_args.items() if not np.isscalar(v)}
    dace_field_strides = _get_stride_args(sdfg.arrays, dace_field_args)
    dace_conn_args = get_sdfg_conn_args(sdfg, offset_provider, on_gpu)
    dace_conn_shapes = _get_shape_args(sdfg.arrays, dace_conn_args)
    dace_conn_strides = _get_stride_args(sdfg.arrays, dace_conn_args)
    all_args = {
        **dace_args,
        **dace_conn_args,
        **dace_conn_shapes,
        **dace_conn_strides,
        **dace_field_strides,
    }

    if check_args:
        # return only arguments expected in SDFG signature (note hat `signature_arglist` takes time)
        sdfg_sig = sdfg.signature_arglist(with_types=False)
        return {key: all_args[key] for key in sdfg_sig}

    return all_args
