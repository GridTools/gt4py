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

import numbers
from typing import Optional, Sequence

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.storage import common as storage_common
from gt4py.storage import allocators
from gt4py.storage.cartesian import utils as storage_utils


# Public interface
def empty(
    domain: common.Domain,
    dtype: core_defs.DTypeLike = np.float64,
    *,
    storage_info: storage_common.StorageInfoInterface,
    aligned_index: Optional[Sequence[int]] = None,
) -> common.Field:
    """Allocate an array of uninitialized (undefined) values with performance-optimal strides and alignment.

    !!!TODO!!!

    Parameters
    ----------
        shape : `Sequence` of `int`
            The shape of the resulting `ndarray`
        dtype :  DTypeLike, optional
            The dtype of the resulting `ndarray`

    Keyword Arguments
    -----------------
        backend : `str`
            The target backend for which the allocation is optimized.
        aligned_index: `Sequence` of `int`, optional
            Indicate the index of the resulting array that most commonly corresponds to the origin of the compute
            domain. If not passed, it is aligned at the data origin.
        dimensions: `Sequence` of `str`, optional
            Indicate the semantic meaning of the dimensions in the provided array. Only used for determining optimal
            strides, the information is not stored.

    Returns
    -------
        NumPy or CuPy ndarray
            With uninitialized values, padded and aligned to provide optimal performance for the given `backend` and
            `aligned_index`

    Raises
    -------
        TypeError
            If arguments of an unexpected type are specified.
        ValueError
            If illegal or inconsistent arguments are specified.
    """
    dtype = core_defs.dtype(dtype)
    shape = domain.shape
    layout_map = list(range(len(shape)))
    device = storage_info.device  # TODO define LayoutInfo

    assert allocators.is_valid_layout_map(layout_map)
    buffer = allocators.allocate(shape, dtype, layout_map, byte_alignment=1, device=device)

    return common.field(buffer.ndarray, domain=domain)  # TODO alignment etc


def zeros(
    domain: common.Domain,
    dtype: core_defs.DTypeLike = np.float64,
    *,
    storage_info: storage_common.StorageInfoInterface,
    aligned_index: Optional[Sequence[int]] = None,
) -> common.Field:
    field = empty(
        domain=domain,
        dtype=dtype,
        storage_info=storage_info,
        aligned_index=aligned_index,
    )
    field.ndarray[...] = field.value_type(0)
    return field


def ones(
    domain: common.Domain,
    dtype: core_defs.DTypeLike = np.float64,
    *,
    storage_info: storage_common.StorageInfoInterface,
    aligned_index: Optional[Sequence[int]] = None,
) -> common.Field:
    field = empty(
        domain=domain,
        dtype=dtype,
        storage_info=storage_info,
        aligned_index=aligned_index,
    )
    field.ndarray[...] = field.value_type(1)
    return field


def fill(
    domain: common.Domain,
    fill_value: numbers.Number,
    dtype: core_defs.DTypeLike = np.float64,
    *,
    storage_info: storage_common.StorageInfoInterface,
    aligned_index: Optional[Sequence[int]] = None,
) -> common.Field:
    field = empty(
        domain=domain,
        dtype=dtype,
        storage_info=storage_info,
        aligned_index=aligned_index,
    )
    field.ndarray[...] = field.value_type(fill_value)
    return field


def from_array(
    data: core_defs.NDArrayObject,
    dimensions: Sequence[common.Dimension],
    dtype: core_defs.DTypeLike = np.float64,
    *,
    storage_info: storage_common.StorageInfoInterface,
    aligned_index: Optional[Sequence[int]] = None,
) -> common.Field:
    shape = storage_utils.asarray(data).shape
    if dtype is None:
        dtype = storage_utils.asarray(data).dtype
    dtype = np.dtype(dtype)
    if dtype.shape:
        if dtype.shape and not shape[-dtype.ndim :] == dtype.shape:
            raise ValueError(f"Incompatible data shape {shape} with dtype of shape {dtype.shape}.")
        shape = shape[: -dtype.ndim]
    storage = empty(
        domain=common.Domain(tuple(dimensions), tuple(common.UnitRange(0, s) for s in shape)),
        dtype=dtype,
        storage_info=storage_info,
        aligned_index=aligned_index,
    )

    device_str = (
        "cpu" if storage_info.device.device_type == core_defs.DeviceType.CPU else "gpu"
    )  # TODO
    storage[...] = storage_utils.asarray(data, device=device_str)

    return storage
