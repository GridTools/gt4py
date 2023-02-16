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

import numbers
from typing import Any, Optional, Sequence, Union

import numpy as np


try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import dace
except ImportError:
    dace = None

if np.lib.NumpyVersion(np.__version__) >= "1.20.0":
    from numpy.typing import ArrayLike, DTypeLike
else:
    ArrayLike = Any  # type: ignore[misc]  # assign multiple types in both branches
    DTypeLike = Any  # type: ignore[misc]  # assign multiple types in both branches

from . import layout, utils as storage_utils


def _error_on_invalid_backend(backend):
    if backend not in layout.REGISTRY:
        raise RuntimeError(f"Storage preset '{backend}' is not registered.")


def empty(
    shape: Sequence[int],
    dtype: DTypeLike = np.float64,
    *,
    backend: str,
    aligned_index: Optional[Sequence[int]] = None,
    dimensions: Optional[Sequence[str]] = None,
) -> Union[np.ndarray, "cp.ndarray"]:
    """Allocate an array of uninitialized (undefined) values with performance-optimal strides and alignment.

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
    _error_on_invalid_backend(backend)
    storage_info = layout.from_name(backend)
    assert storage_info is not None
    if storage_info["device"] == "gpu":
        allocate_f = storage_utils.allocate_gpu
    else:
        allocate_f = storage_utils.allocate_cpu

    aligned_index, shape, dtype, dimensions = storage_utils.normalize_storage_spec(
        aligned_index, shape, dtype, dimensions
    )

    _error_on_invalid_backend(backend)

    alignment = storage_info["alignment"]
    layout_map = storage_info["layout_map"](dimensions)

    dtype = np.dtype(dtype)
    _, res = allocate_f(shape, layout_map, dtype, alignment * dtype.itemsize, aligned_index)
    return res


def ones(
    shape: Sequence[int],
    dtype: DTypeLike = np.float64,
    *,
    backend: str,
    aligned_index: Optional[Sequence[int]] = None,
    dimensions: Optional[Sequence[str]] = None,
) -> Union[np.ndarray, "cp.ndarray"]:
    """Allocate an array with values initialized to 1.0 with performance-optimal strides and alignment.

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
            Initialized to 1.0, padded and aligned to provide optimal performance for the given `backend` and
            `aligned_index`

    Raises
    -------
        TypeError
            If arguments of an unexpected type are specified.
        ValueError
            If illegal or inconsistent arguments are specified.
    """
    storage = empty(
        shape=shape,
        dtype=dtype,
        backend=backend,
        aligned_index=aligned_index,
        dimensions=dimensions,
    )
    storage[...] = storage.dtype.type(1)
    return storage


def full(
    shape: Sequence[int],
    fill_value: numbers.Number,
    dtype: DTypeLike = np.float64,
    *,
    backend: str,
    aligned_index: Optional[Sequence[int]] = None,
    dimensions: Optional[Sequence[str]] = None,
) -> Union[np.ndarray, "cp.ndarray"]:
    """Allocate an array with values initialized to `fill_value` with performance-optimal strides and alignment.

    Parameters
    ----------
        shape : `Sequence` of `int`
            The shape of the resulting `ndarray`
        fill_value: `numbers.Number`
            The value to which the array elements are initialized.
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
            Initialized to `fill_value`, padded and aligned to provide optimal performance for the given `backend` and
            `aligned_index`

    Raises
    -------
        TypeError
            If arguments of an unexpected type are specified.
        ValueError
            If illegal or inconsistent arguments are specified.
    """
    storage = empty(
        shape=shape,
        dtype=dtype,
        backend=backend,
        aligned_index=aligned_index,
        dimensions=dimensions,
    )
    storage[...] = storage.dtype.type(fill_value)
    return storage


def zeros(
    shape: Sequence[int],
    dtype: DTypeLike = np.float64,
    *,
    backend: str,
    aligned_index: Optional[Sequence[int]] = None,
    dimensions: Optional[Sequence[str]] = None,
) -> Union[np.ndarray, "cp.ndarray"]:
    """Allocate an array with values initialized to 0.0 with performance-optimal strides and alignment.

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
            Initialized to 0.0, padded and aligned to provide optimal performance for the given `backend` and
            `aligned_index`

    Raises
    -------
        TypeError
            If arguments of an unexpected type are specified.
        ValueError
            If illegal or inconsistent arguments are specified.
    """
    storage = empty(
        shape=shape,
        dtype=dtype,
        backend=backend,
        aligned_index=aligned_index,
        dimensions=dimensions,
    )
    storage[...] = storage.dtype.type(0)
    return storage


def from_array(
    data: ArrayLike,
    dtype: DTypeLike = np.float64,
    *,
    backend: str,
    aligned_index: Optional[Sequence[int]] = None,
    dimensions: Optional[Sequence[str]] = None,
) -> Union[np.ndarray, "cp.ndarray"]:
    """Allocate an array with values initialized to those of `data` with performance-optimal strides and alignment.

    This copies the values from `data` to the resulting buffer.

    Parameters
    ----------
        data : `ArrayLike`
            Source data to be copied, from which also the `shape` of the result is derived.
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
            Copy of `data`, padded and aligned to provide optimal performance for the given `backend` and
            `aligned_index`

    Raises
    -------
        TypeError
            If arguments of an unexpected type are specified.
        ValueError
            If illegal or inconsistent arguments are specified.
    """
    is_cupy_array = cp is not None and isinstance(data, cp.ndarray)
    asarray = storage_utils.as_cupy if is_cupy_array else storage_utils.as_numpy
    shape = asarray(data).shape
    if dtype is None:
        dtype = asarray(data).dtype
    dtype = np.dtype(dtype)
    if dtype.shape:
        if dtype.shape and not shape[-dtype.ndim :] == dtype.shape:
            raise ValueError(f"Incompatible data shape {shape} with dtype of shape {dtype.shape}.")
        shape = shape[: -dtype.ndim]
    storage = empty(
        shape=shape,
        dtype=dtype,
        backend=backend,
        aligned_index=aligned_index,
        dimensions=dimensions,
    )

    if cp is not None and isinstance(storage, cp.ndarray):
        storage[...] = storage_utils.as_cupy(data)
    else:
        storage[...] = storage_utils.as_numpy(data)

    return storage


if dace is not None:

    def dace_descriptor(
        shape: Sequence[int],
        dtype: DTypeLike = np.float64,
        *,
        backend: str,
        aligned_index: Optional[Sequence[int]] = None,
        dimensions: Optional[Sequence[str]] = None,
    ) -> dace.data.Array:
        """Return a DaCe data descriptor which describes performance-optimal strides and alignment.

        Parameters
        ----------
            shape : `Sequence` of `int`
                The shape of the resulting data descriptor.
            dtype :  DTypeLike, optional
                The dtype encoded in the resulting data descriptor.

        Keyword Arguments
        -----------------
            backend : `str`
                The target backend for which the described allocation is optimized.
            aligned_index: `Sequence` of `int`, optional
                Indicate the index of the resulting array that most commonly corresponds to the origin of the compute
                domain. If not passed, it is aligned at the data origin.
            dimensions: `Sequence` of `str`, optional
                Indicate the semantic meaning of the dimensions in the data descriptor. Only used for determining
                optimal strides, the information is not stored.

        Returns
        -------
            DaCe data descriptor
                With strides that encode padding and aligned to provide optimal performance for the given `backend` and
                `aligned_index`

        Raises
        -------
            TypeError
                If arguments of an unexpected type are specified.
            ValueError
                If illegal or inconsistent arguments are specified.
        """
        aligned_index, shape, dtype, dimensions = storage_utils.normalize_storage_spec(
            aligned_index, shape, dtype, dimensions
        )
        itemsize = dtype.itemsize
        storage_info = layout.from_name(backend)
        assert storage_info is not None
        layout_map = storage_info["layout_map"](dimensions)

        order_idx = storage_utils.idx_from_order([i for i in layout_map if i is not None])
        padded_shape = storage_utils.compute_padded_shape(
            shape, storage_info["alignment"], order_idx
        )

        strides = storage_utils.strides_from_padded_shape(padded_shape, order_idx, itemsize)

        storage = (
            dace.StorageType.GPU_Global
            if storage_info["device"] == "gpu"
            else dace.StorageType.CPU_Heap
        )
        start_offset = int(np.array([aligned_index]) @ np.array([strides]).T) // itemsize

        total_size = int(int(np.array([shape]) @ np.array([strides]).T) // itemsize)

        start_offset = start_offset % storage_info["alignment"]
        return dace.data.Array(
            shape=shape,
            strides=[s // itemsize for s in strides],
            dtype=dace.typeclass(str(dtype)),
            storage=storage,
            total_size=total_size,
            start_offset=start_offset,
        )
