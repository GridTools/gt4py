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

from __future__ import annotations

import collections.abc
import math
import numbers
from typing import Any, Final, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

from gt4py._core import definitions as core_defs
from gt4py.cartesian import config as gt_config
from gt4py.eve.extended_typing import ArrayInterface, CUDAArrayInterface
from gt4py.storage import allocators


if np.lib.NumpyVersion(np.__version__) >= "1.20.0":
    from numpy.typing import DTypeLike
else:
    DTypeLike = Any  # type: ignore[misc]  # assign multiple types in both branches

try:
    import cupy as cp
except ImportError:
    cp = None


CUPY_DEVICE: Final[Literal[None, core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM]] = (
    None
    if not cp
    else (core_defs.DeviceType.ROCM if cp.cuda.get_hipcc_path() else core_defs.DeviceType.CUDA)
)


FieldLike = Union["cp.ndarray", np.ndarray, ArrayInterface, CUDAArrayInterface]

assert allocators.is_valid_nplike_allocation_ns(np)

_CPUBufferAllocator = allocators.NDArrayBufferAllocator(
    device_type=core_defs.DeviceType.CPU,
    array_ns=np,
)

_GPUBufferAllocator: Optional[allocators.NDArrayBufferAllocator] = None
if cp:
    assert allocators.is_valid_nplike_allocation_ns(cp)
    if CUPY_DEVICE == core_defs.DeviceType.CUDA:
        _GPUBufferAllocator = allocators.NDArrayBufferAllocator(
            device_type=core_defs.DeviceType.CUDA, array_ns=cp
        )
    else:
        _GPUBufferAllocator = allocators.NDArrayBufferAllocator(
            device_type=core_defs.DeviceType.ROCM, array_ns=cp
        )


def _idx_from_order(order):
    return list(np.argsort(order))


def _compute_padded_shape(shape, items_per_alignment, order_idx):
    padded_shape = list(shape)
    if len(order_idx) > 0:
        padded_shape[order_idx[-1]] = int(
            math.ceil(padded_shape[order_idx[-1]] / items_per_alignment) * items_per_alignment
        )
    return padded_shape


def _strides_from_padded_shape(padded_size, order_idx, itemsize):
    stride_accumulator = 1
    strides = [0] * len(padded_size)
    for idx in reversed(order_idx):
        strides[idx] = stride_accumulator * itemsize
        stride_accumulator = stride_accumulator * padded_size[idx]
    return list(strides)


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


def cpu_copy(array: Union[np.ndarray, "cp.ndarray"]) -> np.ndarray:
    if cp is not None:
        # it's not clear from the documentation if cp.asnumpy guarantees a copy.
        # worst case, this copies twice.
        return np.array(cp.asnumpy(array))
    else:
        return np.array(array)


def asarray(
    array: FieldLike, *, device: Literal["cpu", "gpu", None] = None
) -> np.ndarray | cp.ndarray:
    if hasattr(array, "ndarray"):
        # extract the buffer from a gt4py.next.Field
        # TODO(havogt): probably `Field` should provide the array interface methods when applicable
        array = array.ndarray
    if device == "gpu" or (not device and hasattr(array, "__cuda_array_interface__")):
        return cp.asarray(array)
    if device == "cpu" or (
        not device and (hasattr(array, "__array_interface__") or hasattr(array, "__array__"))
    ):
        return np.asarray(array)

    if device:
        raise ValueError(f"Invalid device: {device!s}")

    raise TypeError(f"Cannot convert {type(array)} to ndarray")


def get_dims(obj: Union[core_defs.GTDimsInterface, npt.NDArray]) -> Optional[Tuple[str, ...]]:
    dims = getattr(obj, "__gt_dims__", None)
    if dims is None:
        return dims
    return tuple(str(d) for d in dims)


def get_origin(obj: Union[core_defs.GTDimsInterface, npt.NDArray]) -> Optional[Tuple[int, ...]]:
    origin = getattr(obj, "__gt_origin__", None)
    if origin is None:
        return origin
    return tuple(int(o) for o in origin)


def allocate_cpu(
    shape: Sequence[int],
    layout_map: allocators.BufferLayoutMap,
    dtype: DTypeLike,
    alignment_bytes: int,
    aligned_index: Optional[Sequence[int]],
) -> Tuple[allocators._NDBuffer, np.ndarray]:
    device = core_defs.Device(core_defs.DeviceType.CPU, 0)
    buffer = _CPUBufferAllocator.allocate(
        shape,
        core_defs.dtype(dtype),
        device_id=device.device_id,
        layout_map=layout_map,
        byte_alignment=alignment_bytes,
        aligned_index=aligned_index,
    )
    return buffer.buffer, cast(np.ndarray, buffer.ndarray)


def allocate_gpu(
    shape: Sequence[int],
    layout_map: allocators.BufferLayoutMap,
    dtype: DTypeLike,
    alignment_bytes: int,
    aligned_index: Optional[Sequence[int]],
) -> Tuple["cp.ndarray", "cp.ndarray"]:
    assert _GPUBufferAllocator is not None, "GPU allocation library or device not found"
    device = core_defs.Device(  # type: ignore[type-var]
        core_defs.DeviceType.ROCM if gt_config.GT4PY_USE_HIP else core_defs.DeviceType.CUDA, 0
    )
    buffer = _GPUBufferAllocator.allocate(
        shape,
        core_defs.dtype(dtype),
        device_id=device.device_id,
        layout_map=layout_map,
        byte_alignment=alignment_bytes,
        aligned_index=aligned_index,
    )
    return buffer.buffer, cast("cp.ndarray", buffer.ndarray)
