# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import dace

from gt4py._core import definitions as core_defs
from gt4py.next import common as gtx_common, field_utils

from . import utils as gtx_dace_utils


def get_field_domain_symbols(name: str, domain: gtx_common.Domain) -> dict[str, int]:
    assert gtx_common.Domain.is_finite(domain)
    return {
        gtx_dace_utils.range_start_symbol(name, dim): r.start
        for dim, r in zip(domain.dims, domain.ranges, strict=True)
    } | {
        gtx_dace_utils.range_stop_symbol(name, dim): r.stop
        for dim, r in zip(domain.dims, domain.ranges, strict=True)
    }


def get_array_shape_symbols(
    array_desc: dace.data.Array, ndarray: core_defs.NDArrayObject
) -> dict[str, int]:
    array_symbols = {}
    for array_size, size in zip(array_desc.shape, ndarray.shape, strict=True):
        if (array_size == size) == True:  # noqa: E712 [true-false-comparison]  # SymPy Fancy comparison.
            pass
        else:
            assert isinstance(array_size, dace.symbol)
            array_symbols[array_size.name] = size
    return array_symbols


def get_array_stride_symbols(
    array_desc: dace.data.Array, ndarray: core_defs.NDArrayObject
) -> dict[str, int]:
    array_symbols = {}
    for array_stride, value in zip(array_desc.strides, ndarray.strides, strict=True):
        assert divmod(value, ndarray.itemsize)[1] == 0
        stride = value // ndarray.itemsize
        if (array_stride == stride) == True:  # noqa: E712 [true-false-comparison]  # SymPy Fancy comparison.
            pass
        else:
            assert isinstance(array_stride, dace.symbol)
            array_symbols[array_stride.name] = stride
    return array_symbols


def _convert_arg(arg: Any) -> tuple[Any, Optional[gtx_common.Domain]]:
    if (domain := getattr(arg, "domain", None)) is None:
        return arg, None
    if len(domain.dims) == 0:
        # Pass zero-dimensional fields as scalars.
        return arg.as_scalar(), None
    return arg.ndarray, domain


def _get_args(sdfg: dace.SDFG, args: Sequence[Any]) -> dict[str, Any]:
    call_args = {}
    range_symbols: dict[str, int] = {}
    stride_symbols: dict[str, int] = {}
    for name, arg in zip(sdfg.arg_names, args, strict=True):
        call_arg, domain = _convert_arg(arg)
        call_args[name] = call_arg
        if domain is not None:
            range_symbols |= get_field_domain_symbols(name, domain)
            stride_symbols |= get_array_stride_symbols(sdfg.arrays[name], call_arg)
    # sanity check in case range symbols are passed as explicit program arguments
    assert all(
        call_arg == range_symbols[param]
        for param, call_arg in call_args.items()
        if param in range_symbols
    )
    return call_args | range_symbols | stride_symbols


def get_sdfg_conn_args(
    sdfg: dace.SDFG,
    offset_provider: gtx_common.OffsetProvider,
) -> dict[str, core_defs.NDArrayObject]:
    """
    Extracts the connectivity tables that are used in the sdfg and ensures
    that the memory buffers are allocated for the target device.
    """
    connectivity_args = {}
    for offset, connectivity in offset_provider.items():
        name = gtx_dace_utils.connectivity_identifier(offset)
        if name in sdfg.arrays:
            assert gtx_common.is_neighbor_connectivity(connectivity)
            assert field_utils.verify_device_field_type(
                connectivity,
                core_defs.CUPY_DEVICE_TYPE
                if sdfg.arrays[name].storage == dace.dtypes.StorageType.GPU_Global
                else core_defs.DeviceType.CPU,  # type: ignore[arg-type] # if we are `on_gpu` we expect `CUPY_DEVICE_TYPE` to be not `None`
            )
            connectivity_args[name] = connectivity.ndarray

    return connectivity_args


def get_sdfg_args(
    sdfg: dace.SDFG,
    offset_provider: gtx_common.OffsetProvider,
    *args: Any,
    filter_args: bool = False,
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

    Returns:
        A dictionary of keyword arguments to be passed in the SDFG call.
    """

    call_args = _get_args(sdfg, args)
    connectivity_args = get_sdfg_conn_args(sdfg, offset_provider)
    for conn, ndarray in connectivity_args.items():
        call_args |= get_array_shape_symbols(sdfg.arrays[conn], ndarray)
        call_args |= get_array_stride_symbols(sdfg.arrays[conn], ndarray)
    call_args |= connectivity_args

    if filter_args:
        # return only arguments expected in SDFG signature (note hat `signature_arglist` takes time)
        sdfg_sig = sdfg.signature_arglist(with_types=False)
        return {key: call_args[key] for key in sdfg_sig}

    return call_args
