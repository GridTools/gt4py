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


def get_field_shape_symbols(field_name: str, field_domain: gtx_common.Domain) -> dict[str, Any]:
    assert gtx_common.Domain.is_finite(field_domain)
    if len(field_domain.dims) == 0:
        return {}
    return {
        gtx_dace_utils.range_start_symbol(field_name, i): r.start
        for i, r in enumerate(field_domain.ranges)
    } | {
        gtx_dace_utils.range_stop_symbol(field_name, i): r.stop
        for i, r in enumerate(field_domain.ranges)
    }


def get_field_stride_symbols(
    array_desc: dace.data.Array, ndarray: core_defs.NDArrayObject
) -> dict[str, int]:
    stride_symbols = {}
    for sym, stride_size in zip(array_desc.strides, ndarray.strides, strict=True):
        stride, remainder = divmod(stride_size, ndarray.itemsize)
        assert remainder == 0
        if (sym == stride) == True:  # noqa: E712 [true-false-comparison]  # SymPy Fancy comparison.
            pass
        else:
            assert isinstance(sym, dace.symbol)
            stride_symbols[sym.name] = stride
    return stride_symbols


def _convert_arg(arg: Any) -> tuple[Any, Optional[gtx_common.Domain]]:
    if (domain := getattr(arg, "domain", None)) is None:
        return arg, None
    if len(domain.dims) == 0:
        # Pass zero-dimensional fields as scalars.
        return arg.as_scalar(), None
    return arg.ndarray, domain


def _get_args(sdfg: dace.SDFG, args: Sequence[Any]) -> dict[str, Any]:
    sdfg_params: Sequence[str] = sdfg.arg_names
    sdfg_arguments = {}
    range_symbols: dict[str, int] = {}
    for sdfg_param, arg in zip(sdfg_params, args, strict=True):
        sdfg_arg, domain = _convert_arg(arg)
        sdfg_arguments[sdfg_param] = sdfg_arg
        if domain:
            range_symbols |= get_field_shape_symbols(sdfg_param, domain)
    # sanity check in case range symbols are passed as explicit program arguments
    for range_symbol, value in range_symbols.items():
        if (sdfg_arg := sdfg_arguments.get(range_symbol, None)) is not None:
            if sdfg_arg != value:
                raise ValueError(
                    f"Received program argument {range_symbol} with value {sdfg_arg}, expected {value}."
                )
    return sdfg_arguments | range_symbols


def _get_shape_args(sdfg: dace.SDFG, args: Mapping[str, core_defs.NDArrayObject]) -> dict[str, int]:
    shape_args: dict[str, int] = {}
    for name, value in args.items():
        for sym, size in zip(sdfg.arrays[name].shape, value.shape, strict=True):
            if (sym == size) == True:  # noqa: E712 [true-false-comparison]  # SymPy Fancy comparison.
                pass
            else:
                assert isinstance(sym, dace.symbol)
                shape_args[sym.name] = size
    return shape_args


def _get_stride_args(
    sdfg: dace.SDFG, args: Mapping[str, core_defs.NDArrayObject]
) -> dict[str, int]:
    stride_args: dict[str, int] = {}
    for name, value in args.items():
        stride_args |= get_field_stride_symbols(sdfg.arrays[name], value)
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
        param = gtx_dace_utils.connectivity_identifier(offset)
        if param in sdfg.arrays:
            assert gtx_common.is_neighbor_connectivity(connectivity)
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
    filter_args: bool = False,
    on_gpu: bool = False,
) -> dict[str, Any]:
    """Extracts the arguments needed to call the SDFG.

    This function can handle the arguments that are passed to the dace runner
    and that end up in the decoration stage of the dace backend workflow.

    Args:
        sdfg:               The SDFG for which we want to get the arguments.
        offset_provider:    The offset provider.
        args:               The list of arguments passed to the dace runner.
        filter_args:        If True, return only the arguments that are expected
                            according to the SDFG signature.
        on_gpu:             If True, this method ensures that the arrays for the
                            connectivity tables are allocated in GPU memory.

    Returns:
        A dictionary of keyword arguments to be passed in the SDFG call.
    """

    dace_args = _get_args(sdfg, args)
    dace_field_args = {n: v for n, v in dace_args.items() if not np.isscalar(v)}
    dace_field_strides = _get_stride_args(sdfg, dace_field_args)
    dace_conn_args = get_sdfg_conn_args(sdfg, offset_provider, on_gpu)
    dace_conn_shapes = _get_shape_args(sdfg, dace_conn_args)
    dace_conn_strides = _get_stride_args(sdfg, dace_conn_args)
    all_args = {
        **dace_args,
        **dace_conn_args,
        **dace_conn_shapes,
        **dace_conn_strides,
        **dace_field_strides,
    }

    if filter_args:
        # return only arguments expected in SDFG signature (note hat `signature_arglist` takes time)
        sdfg_sig = sdfg.signature_arglist(with_types=False)
        return {key: all_args[key] for key in sdfg_sig}

    return all_args
