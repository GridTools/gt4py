# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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

import math
import numbers
from typing import Optional, Sequence

import numpy as np

import gt4py.utils as gt_util
from gt4py.definitions import Index, Shape


try:
    import cupy as cp
    from cupy.lib.stride_tricks import as_strided
except ImportError:
    pass


def idx_from_order(order):
    return list(np.argsort(order))


def check_mask(mask):
    if not gt_util.is_iterable_of(mask, bool) and not mask is None:
        raise TypeError("Mask must be an iterable of booleans.")


def normalize_shape(
    shape: Optional[Sequence[int]], mask: Optional[Sequence[bool]] = None
) -> Optional[Shape]:

    check_mask(mask)

    if shape is None:
        return None
    if mask is None:
        mask = (True,) * len(shape)

    if sum(mask) != len(shape) and len(mask) != len(shape):
        raise ValueError(
            "len(shape) must be equal to len(mask) or the number of 'True' entries in mask."
        )

    if not gt_util.is_iterable_of(shape, numbers.Integral):
        raise TypeError("shape must be a tuple of ints or pairs of ints.")
    if any(o <= 0 for o in shape):
        raise ValueError("shape ({}) contains non-positive value.".format(shape))

    new_shape = list(shape)
    if sum(mask) < len(shape):
        new_shape = [int(h) for i, h in enumerate(new_shape) if mask[i]]

    return Shape(new_shape)


def normalize_default_origin(
    default_origin: Optional[Sequence[int]], mask: Optional[Sequence[bool]] = None
) -> Optional[Index]:

    check_mask(mask)

    if default_origin is None:
        return None
    if mask is None:
        mask = (True,) * len(default_origin)

    if sum(mask) != len(default_origin) and len(mask) != len(default_origin):
        raise ValueError(
            "len(default_origin) must be equal to len(mask) or the number of 'True' entries in mask."
        )

    if not gt_util.is_iterable_of(default_origin, numbers.Integral):
        raise TypeError("default_origin must be a tuple of ints or pairs of ints.")
    if any(o < 0 for o in default_origin):
        raise ValueError("default_origin ({}) contains negative value.".format(default_origin))

    new_default_origin = list(default_origin)
    if sum(mask) < len(default_origin):
        new_default_origin = [h for i, h in enumerate(new_default_origin) if mask[i]]

    return Index(new_default_origin)


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


def allocate(default_origin, shape, layout_map, dtype, alignment_bytes, allocate_f):
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
            int(math.ceil(default_origin[order_idx[-1]] / items_per_alignment))
            * items_per_alignment
            - default_origin[order_idx[-1]]
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


def allocate_gpu_unmanaged(default_origin, shape, layout_map, dtype, alignment_bytes):
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
            int(math.ceil(default_origin[order_idx[-1]] / items_per_alignment))
            * items_per_alignment
            - default_origin[order_idx[-1]]
        )
    else:
        halo_offset = 0

    padded_size = int(np.prod(padded_shape))
    buffer_size = padded_size + items_per_alignment - 1

    ptr = cp.cuda.alloc_pinned_memory(buffer_size * itemsize)
    raw_buffer = np.frombuffer(ptr, dtype, buffer_size)
    device_raw_buffer = cp.empty((buffer_size,), dtype=dtype)

    allocation_mismatch = int((raw_buffer.ctypes.data % alignment_bytes) / itemsize)
    alignment_offset = (halo_offset - allocation_mismatch) % items_per_alignment
    field = np.reshape(raw_buffer[alignment_offset : alignment_offset + padded_size], padded_shape)
    if field.ndim > 0:
        field.strides = strides
        field = field[tuple(slice(0, s, None) for s in shape)]

    allocation_mismatch = int((device_raw_buffer.data.ptr % alignment_bytes) / itemsize)
    alignment_offset = (halo_offset - allocation_mismatch) % items_per_alignment

    device_field = as_strided(
        device_raw_buffer[alignment_offset : alignment_offset + padded_size],
        shape=padded_shape,
        strides=strides,
    )
    if device_field.ndim > 0:
        device_field = device_field[tuple(slice(0, s, None) for s in shape)]

    return raw_buffer, field, device_raw_buffer, device_field


def allocate_cpu(default_origin, shape, layout_map, dtype, alignment_bytes):
    def allocate_f(size, dtype):
        raw_buffer = np.empty(size, dtype)
        return raw_buffer, raw_buffer

    return allocate(default_origin, shape, layout_map, dtype, alignment_bytes, allocate_f)


def allocate_gpu(default_origin, shape, layout_map, dtype, alignment_bytes):
    def allocate_f(size, dtype):
        cp.cuda.set_allocator(cp.cuda.malloc_managed)
        device_buffer = cp.empty(size, dtype)
        array = cpu_view(device_buffer)
        return array, device_buffer

    return allocate(default_origin, shape, layout_map, dtype, alignment_bytes, allocate_f)


def gpu_view(cpu_array):
    array_interface = cpu_array.__array_interface__
    array_interface["version"] = 2
    array_interface["strides"] = cpu_array.strides
    array_interface.pop("offset", None)

    class _cuda_array_interface:
        __cuda_array_interface__ = array_interface

    return cp.asarray(_cuda_array_interface())


def cpu_view(gpu_array):
    array_interface = gpu_array.__cuda_array_interface__
    array_interface["version"] = 3

    class _array_interface:
        __array_interface__ = array_interface

    return np.asarray(_array_interface())
