# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

import collections.abc
import math
import numbers
from typing import Any, Dict, Iterable, Optional, Protocol, Sequence, Tuple, Union, cast

import numpy as np
import numpy.typing as npt


if np.lib.NumpyVersion(np.__version__) >= "1.20.0":
    from numpy.typing import DTypeLike
else:
    DTypeLike = Any  # type: ignore[misc]  # assign multiple types in both branches

try:
    import cupy as cp
    from cupy.lib.stride_tricks import as_strided
except ImportError:
    cp = None


class ArrayInterfaceType(Protocol):
    __array_interface__: Dict[str, Any]


class CudaArrayInterfaceType(Protocol):
    __cuda_array_interface__: Dict[str, Any]


class GtDimsInterface(Protocol):
    __gt_dims__: Tuple[str, ...]


class GtOriginInterface(Protocol):
    __gt_origin__: Tuple[int, ...]


FieldLike = Union["cp.ndarray", np.ndarray, ArrayInterfaceType, CudaArrayInterfaceType]


def idx_from_order(order):
    return list(np.argsort(order))


def dimensions_to_mask(dimensions: Tuple[str, ...]) -> Tuple[bool, ...]:
    ndata_dims = sum(d.isdigit() for d in dimensions)
    mask = [(d in dimensions) for d in "IJK"] + [True for _ in range(ndata_dims)]
    return tuple(mask)


def normalize_storage_spec(
    aligned_index: Optional[Sequence[int]],
    shape: Sequence[int],
    dtype: DTypeLike,
    dimensions: Optional[Sequence[str]],
) -> Tuple[Sequence[int], Sequence[int], np.dtype, Tuple[str, ...]]:
    """Normalize the fields of the storage spec in a homogeneous representation.

    Returns
    -------
    tuple(aligned_index, shape, dtype, mask)
        The output tuple fields verify the following semantics:
            - aligned_index: tuple of ints with default origin values for the non-masked dimensions
            - shape: tuple of ints with shape values for the non-masked dimensions
            - dtype: scalar numpy.dtype (non-structured and without subarrays)
            - backend: backend identifier string (numpy, gt:cpu_kfirst, gt:gpu, ...)
            - dimensions: a tuple of dimension identifier strings
    """
    if dimensions is None:
        dimensions = (
            list("IJK"[: len(shape)])
            if len(shape) <= 3
            else list("IJK") + [str(d) for d in range(len(shape) - 3)]
        )

    if aligned_index is None:
        aligned_index = [0] * len(shape)

    dimensions = tuple(getattr(d, "__gt_axis_name__", d) for d in dimensions)
    if not all(isinstance(d, str) and (d.isdigit() or d in "IJK") for d in dimensions):
        raise ValueError(f"Invalid dimensions definition: '{dimensions}'")
    else:
        dimensions = tuple(str(d) for d in dimensions)
    if shape is not None:
        if not (
            isinstance(shape, collections.abc.Sequence)
            and all(isinstance(s, numbers.Integral) for s in shape)
        ):
            raise TypeError("shape must be an iterable of ints.")
        if len(shape) != len(dimensions):
            raise ValueError(
                f"Dimensions ({dimensions}) and shape ({shape}) have non-matching sizes."
                f"len(shape)(={len(shape)}) must be equal to len(dimensions)(={len(dimensions)})."
            )

        else:
            shape = tuple(shape)

        if any(i <= 0 for i in shape):
            raise ValueError(f"shape ({shape}) contains non-positive value.")
    else:
        raise TypeError("shape must be an iterable of ints.")

    if aligned_index is not None:
        if not (
            isinstance(aligned_index, collections.abc.Sequence)
            and all(isinstance(i, numbers.Integral) for i in aligned_index)
        ):
            raise TypeError("aligned_index must be an iterable of ints.")
        if len(aligned_index) != len(shape):
            raise ValueError(
                f"Shape ({shape}) and aligned_index ({aligned_index}) have non-matching sizes."
                f"len(aligned_index)(={len(aligned_index)}) must be equal to len(shape)(={len(shape)})."
            )

        aligned_index = tuple(aligned_index)

        if any(i < 0 for i in aligned_index):
            raise ValueError("aligned_index ({}) contains negative value.".format(aligned_index))
    else:
        raise TypeError("aligned_index must be an iterable of ints.")

    dtype = np.dtype(dtype)
    if dtype.shape:
        # Subarray dtype
        sub_dtype, sub_shape = cast(Tuple[np.dtype, Tuple[int, ...]], dtype.subdtype)
        aligned_index = (*aligned_index, *((0,) * dtype.ndim))
        shape = (*shape, *sub_shape)
        dimensions = (*dimensions, *(str(d) for d in range(dtype.ndim)))
        dtype = sub_dtype

    return aligned_index, shape, dtype, dimensions


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
    if not (alignment_bytes % dtype.itemsize) == 0:
        raise ValueError("Alignment must be a multiple of byte-width of dtype.")
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


def allocate_gpu(
    shape: Sequence[int],
    layout_map: Iterable[Optional[int]],
    dtype: DTypeLike,
    alignment_bytes: int,
    aligned_index: Optional[Sequence[int]],
) -> Tuple["cp.ndarray", "cp.ndarray"]:
    dtype = np.dtype(dtype)
    assert (
        alignment_bytes % dtype.itemsize
    ) == 0, "Alignment must be a multiple of byte-width of dtype."
    itemsize = dtype.itemsize
    items_per_alignment = int(alignment_bytes / itemsize)

    order_idx = idx_from_order([i for i in layout_map if i is not None])
    padded_shape = compute_padded_shape(shape, items_per_alignment, order_idx)

    if aligned_index is None:
        aligned_index = [0] * len(shape)

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


def allocate_cpu(
    shape: Sequence[int],
    layout_map: Iterable[Optional[int]],
    dtype: DTypeLike,
    alignment_bytes: int,
    aligned_index: Optional[Sequence[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    dtype = np.dtype(dtype)
    assert (
        alignment_bytes % dtype.itemsize
    ) == 0, "Alignment must be a multiple of byte-width of dtype."
    itemsize = dtype.itemsize
    items_per_alignment = int(alignment_bytes / itemsize)

    order_idx = idx_from_order([i for i in layout_map if i is not None])
    padded_shape = compute_padded_shape(shape, items_per_alignment, order_idx)

    if aligned_index is None:
        aligned_index = [0] * len(shape)

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


def cpu_copy(array: Union[np.ndarray, "cp.ndarray"]) -> np.ndarray:
    if cp is not None:
        # it's not clear from the documentation if cp.asnumpy guarantees a copy.
        # worst case, this copies twice.
        return np.array(cp.asnumpy(array))
    else:
        return np.array(array)


def as_numpy(array: FieldLike) -> np.ndarray:
    return np.asarray(array)


def as_cupy(array: FieldLike) -> "cp.ndarray":
    return cp.asarray(array)


def get_dims(obj: Union[GtDimsInterface, npt.NDArray]) -> Optional[Tuple[str, ...]]:
    dims = getattr(obj, "__gt_dims__", None)
    if dims is None:
        return dims
    return tuple(str(d) for d in dims)


def get_origin(obj: Union[GtDimsInterface, npt.NDArray]) -> Optional[Tuple[int, ...]]:
    origin = getattr(obj, "__gt_origin__", None)
    if origin is None:
        return origin
    return tuple(int(o) for o in origin)
