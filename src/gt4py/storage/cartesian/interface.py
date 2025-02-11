# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numbers
from typing import Optional, Sequence, Union

import numpy as np

from gt4py.storage import allocators
from gt4py.storage.cartesian import layout, utils as storage_utils


try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import dace
except ImportError:
    dace = None

from numpy.typing import ArrayLike, DTypeLike


# Helper functions
def _error_on_invalid_preset(backend):
    if backend not in layout.REGISTRY:
        raise RuntimeError(f"Storage preset '{backend}' is not registered.")


# Public interface
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
    _error_on_invalid_preset(backend)
    storage_info = layout.from_name(backend)
    assert storage_info is not None
    if storage_info["device"] == layout.StorageDevice.GPU:
        allocate_f = storage_utils.allocate_gpu
    else:
        allocate_f = storage_utils.allocate_cpu

    aligned_index, shape, dtype, dimensions = storage_utils.normalize_storage_spec(
        aligned_index, shape, dtype, dimensions
    )

    _error_on_invalid_preset(backend)

    alignment = storage_info["alignment"]
    layout_map = storage_info["layout_map"](dimensions)
    assert allocators.is_valid_layout_map(layout_map)

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
    shape = storage_utils.asarray(data).shape
    if dtype is None:
        dtype = storage_utils.asarray(data).dtype
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

    layout_info = layout.from_name(backend)
    assert layout_info is not None
    storage[...] = storage_utils.asarray(data, device=layout_info["device"])

    return storage
