# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import collections.abc
import math
import numbers
from typing import Optional, Sequence, Union

import numpy as np

import gt4py.utils as gt_util
from gtc.definitions import Index, Shape


try:
    import cupy as cp
    from cupy.lib.stride_tricks import as_strided
except ImportError:
    cp = None


def idx_from_order(order):
    return list(np.argsort(order))


def normalize_storage_spec(aligned_index, shape, dtype, mask):
    """Normalize the fields of the storage spec in a homogeneous representation.

    Returns
    -------

    tuple(aligned_index, shape, dtype, mask)
        The output tuple fields verify the following semantics:

            - aligned_index: tuple of ints with default origin values for the non-masked dimensions
            - shape: tuple of ints with shape values for the non-masked dimensions
            - dtype: scalar numpy.dtype (non-structured and without subarrays)
            - backend: backend identifier string (numpy, gt:cpu_kfirst, gt:gpu, ...)
            - mask: a tuple of bools (at least 3d)
    """

    if mask is None:
        mask = tuple(True if i < len(shape) else False for i in range(max(len(shape), 3)))
    elif not gt_util.is_iterable_of(mask, bool):
        # User-friendly axes specification (e.g. "IJK" or gtscript.IJK)
        str_kind = "".join(str(i) for i in mask) if isinstance(mask, (tuple, list)) else str(mask)
        axes_set = set(str_kind)
        if axes_set - {"I", "J", "K"}:
            raise ValueError(f"Invalid axes names in mask specification: '{mask}'")
        if len(axes_set) != len(str_kind):
            raise ValueError(f"Repeated axes names in mask specification: '{mask}'")
        mask = ("I" in axes_set, "J" in axes_set, "K" in axes_set)
    elif len(mask) < 3 or not sum(mask):
        raise ValueError(f"Invalid mask definition: '{mask}'")

    assert len(mask) >= 3

    if shape is not None:
        if not gt_util.is_iterable_of(shape, numbers.Integral):
            raise TypeError("shape must be an iterable of ints.")
        if len(shape) not in (sum(mask), len(mask)):
            raise ValueError(
                f"Mask ({mask}) and shape ({shape}) have non-matching sizes."
                f"len(shape)(={len(shape)}) must be equal to len(mask)(={len(mask)}) "
                f"or the number of 'True' entries in mask '{mask}'."
            )

        if sum(mask) < len(shape):
            shape = tuple(int(d) for i, d in enumerate(shape) if mask[i])
        else:
            shape = tuple(shape)

        if any(i <= 0 for i in shape):
            raise ValueError(f"shape ({shape}) contains non-positive value.")
    else:
        raise TypeError("shape must be an iterable of ints.")

    if aligned_index is not None:
        if not gt_util.is_iterable_of(aligned_index, numbers.Integral):
            raise TypeError("aligned_index must be an iterable of ints.")
        if len(aligned_index) not in (sum(mask), len(mask)):
            raise ValueError(
                f"Mask ({mask}) and aligned_index ({aligned_index}) have non-matching sizes."
                f"len(aligned_index)(={len(aligned_index)}) must be equal to len(mask)(={len(mask)}) "
                f"or the number of 'True' entries in mask '{mask}'."
            )

        if sum(mask) < len(aligned_index):
            aligned_index = tuple(d for i, d in enumerate(aligned_index) if mask[i])
        else:
            aligned_index = tuple(aligned_index)

        if any(i < 0 for i in aligned_index):
            raise ValueError("aligned_index ({}) contains negative value.".format(aligned_index))
    else:
        raise TypeError("aligned_index must be an iterable of ints.")

    dtype = np.dtype(dtype)
    if dtype.shape:
        # Subarray dtype
        aligned_index = (*aligned_index, *((0,) * dtype.ndim))
        shape = (*shape, *(dtype.subdtype[1]))
        mask = (*mask, *((True,) * dtype.ndim))
        dtype = dtype.subdtype[0]

    return aligned_index, shape, dtype, mask


def compute_padded_shape(shape, items_per_alignment, order_idx):
    padded_shape = list(shape)
    if len(order_idx) > 0:
        padded_shape[order_idx[-1]] = int(
            math.ceil(padded_shape[order_idx[-1]] / items_per_alignment) * items_per_alignment
        )
    return padded_shape


def strides_from_padded_shape(padded_size, order_idx, itemsize):
    stride_accumulator = 1
    strides = [0] * len(padded_size)
    for idx in reversed(order_idx):
        strides[idx] = stride_accumulator * itemsize
        stride_accumulator = stride_accumulator * padded_size[idx]
    return list(strides)


def allocate(aligned_index, shape, layout_map, dtype, alignment_bytes, allocate_f):
    dtype = np.dtype(dtype)
    assert (
        alignment_bytes % dtype.itemsize
    ) == 0, "Alignment must be a multiple of byte-width of dtype."
    itemsize = dtype.itemsize
    items_per_alignment = int(alignment_bytes / itemsize)

    order_idx = idx_from_order([i for i in layout_map if i is not None])
    padded_shape = compute_padded_shape(shape, items_per_alignment, order_idx)

    strides = strides_from_padded_shape(padded_shape, order_idx, itemsize)
    if len(order_idx) > 0:
        halo_offset = (
            int(math.ceil(aligned_index[order_idx[-1]] / items_per_alignment)) * items_per_alignment
            - aligned_index[order_idx[-1]]
        )
    else:
        halo_offset = 0

    padded_size = int(np.prod(padded_shape))
    buffer_size = padded_size + items_per_alignment - 1
    array, raw_buffer = allocate_f(buffer_size, dtype=dtype)

    allocation_mismatch = int((array.ctypes.data % alignment_bytes) / itemsize)

    alignment_offset = (halo_offset - allocation_mismatch) % items_per_alignment

    field = np.reshape(array[alignment_offset : alignment_offset + padded_size], padded_shape)
    if field.ndim > 0:
        field.strides = strides
        field = field[tuple(slice(0, s, None) for s in shape)]
    return raw_buffer, field


def allocate_gpu(aligned_index, shape, layout_map, dtype, alignment_bytes):
    dtype = np.dtype(dtype)
    assert (
        alignment_bytes % dtype.itemsize
    ) == 0, "Alignment must be a multiple of byte-width of dtype."
    itemsize = dtype.itemsize
    items_per_alignment = int(alignment_bytes / itemsize)

    order_idx = idx_from_order([i for i in layout_map if i is not None])
    padded_shape = compute_padded_shape(shape, items_per_alignment, order_idx)

    strides = strides_from_padded_shape(padded_shape, order_idx, itemsize)
    if len(order_idx) > 0:
        halo_offset = (
            int(math.ceil(aligned_index[order_idx[-1]] / items_per_alignment)) * items_per_alignment
            - aligned_index[order_idx[-1]]
        )
    else:
        halo_offset = 0

    padded_size = int(np.prod(padded_shape))
    buffer_size = padded_size + items_per_alignment - 1

    device_raw_buffer = cp.empty((buffer_size,), dtype=dtype)

    allocation_mismatch = int((device_raw_buffer.data.ptr % alignment_bytes) / itemsize)
    alignment_offset = (halo_offset - allocation_mismatch) % items_per_alignment

    device_field = as_strided(
        device_raw_buffer[alignment_offset : alignment_offset + padded_size],
        shape=padded_shape,
        strides=strides,
    )
    if device_field.ndim > 0:
        device_field = device_field[tuple(slice(0, s, None) for s in shape)]

    return device_raw_buffer, device_field


def allocate_cpu(aligned_index, shape, layout_map, dtype, alignment_bytes):
    dtype = np.dtype(dtype)
    assert (
        alignment_bytes % dtype.itemsize
    ) == 0, "Alignment must be a multiple of byte-width of dtype."
    itemsize = dtype.itemsize
    items_per_alignment = int(alignment_bytes / itemsize)

    order_idx = idx_from_order([i for i in layout_map if i is not None])
    padded_shape = compute_padded_shape(shape, items_per_alignment, order_idx)

    strides = strides_from_padded_shape(padded_shape, order_idx, itemsize)
    if len(order_idx) > 0:
        halo_offset = (
            int(math.ceil(aligned_index[order_idx[-1]] / items_per_alignment)) * items_per_alignment
            - aligned_index[order_idx[-1]]
        )
    else:
        halo_offset = 0

    padded_size = int(np.prod(padded_shape))
    buffer_size = padded_size + items_per_alignment - 1
    raw_buffer = np.empty(buffer_size, dtype=dtype)

    allocation_mismatch = int((raw_buffer.ctypes.data % alignment_bytes) / itemsize)

    alignment_offset = (halo_offset - allocation_mismatch) % items_per_alignment

    field = np.reshape(raw_buffer[alignment_offset : alignment_offset + padded_size], padded_shape)
    if field.ndim > 0:
        field.strides = strides
        field = field[tuple(slice(0, s, None) for s in shape)]
    return raw_buffer, field


def cpu_copy(array: Union[np.ndarray, "cp.ndarray"]):
    if cp is not None:
        # it's not clear from the documentation if cp.asnumpy guarantees a copy.
        # worst case, this copies twice.
        return np.array(cp.asnumpy(array))
    else:
        return np.array(array)
