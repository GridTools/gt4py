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

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

import gt4py.storage.utils
from gt4py import backend as gt_backend


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
    ArrayLike = Any
    DTypeLike = Any

from . import utils as storage_utils


def _error_on_invalid_backend(backend):
    if backend not in gt_backend.REGISTRY:
        raise RuntimeError(f"Backend '{backend}' is not registered.")


def empty(
    shape: Sequence[int],
    dtype: DTypeLike = np.float64,
    *,
    backend: str,
    aligned_index: Tuple[int, ...],
    dimensions: Optional[Tuple[str, ...]] = None,
) -> Union[np.ndarray, "cp.ndarray"]:
    _error_on_invalid_backend(backend)
    if gt_backend.from_name(backend).storage_info["device"] == "gpu":
        allocate_f = storage_utils.allocate_gpu
    else:
        allocate_f = storage_utils.allocate_cpu

    aligned_index, shape, dtype, dimensions = storage_utils.normalize_storage_spec(
        aligned_index, shape, dtype, dimensions
    )

    _error_on_invalid_backend(backend)

    alignment = gt_backend.from_name(backend).storage_info["alignment"]
    layout_map = gt_backend.from_name(backend).storage_info["layout_map"](dimensions)

    dtype = np.dtype(dtype)
    _, res = allocate_f(aligned_index, shape, layout_map, dtype, alignment * dtype.itemsize)
    return res


def ones(
    shape: Sequence[int],
    dtype: DTypeLike = np.float64,
    *,
    backend: str,
    aligned_index: Tuple[int, ...],
    dimensions: Optional[Tuple[str, ...]] = None,
) -> Union[np.ndarray, "cp.ndarray"]:
    storage = empty(
        shape=shape,
        dtype=dtype,
        backend=backend,
        aligned_index=aligned_index,
        dimensions=dimensions,
    )
    storage[...] = 1
    return storage


def zeros(
    shape: Sequence[int],
    dtype: DTypeLike = np.float64,
    *,
    backend: str,
    aligned_index: Tuple[int, ...],
    dimensions: Optional[Tuple[str, ...]] = None,
) -> Union[np.ndarray, "cp.ndarray"]:
    storage = empty(
        shape=shape,
        dtype=dtype,
        backend=backend,
        aligned_index=aligned_index,
        dimensions=dimensions,
    )
    storage[...] = 0
    return storage


def from_array(
    data: ArrayLike,
    dtype: DTypeLike = np.float64,
    *,
    backend: str,
    aligned_index: Tuple[int, ...],
    dimensions: Optional[Tuple[str, ...]] = None,
) -> Union[np.ndarray, "cp.ndarray"]:
    is_cupy_array = cp is not None and isinstance(data, cp.ndarray)
    asarray = gt4py.storage.utils.as_cupy if is_cupy_array else gt4py.storage.utils.as_numpy
    shape = asarray(data).shape
    if dtype is None:
        dtype = asarray(data).dtype
    storage = empty(
        shape=shape,
        dtype=dtype,
        backend=backend,
        aligned_index=aligned_index,
        dimensions=dimensions,
    )

    if cp is not None and isinstance(storage, cp.ndarray):
        storage[...] = gt4py.storage.utils.as_cupy(data)
    else:
        storage[...] = gt4py.storage.utils.as_numpy(data)

    return storage


if dace is not None:

    def dace_descriptor(
        shape: Sequence[int],
        dtype: DTypeLike = np.float64,
        *,
        backend: str,
        aligned_index: Tuple[int, ...],
        dimensions: Optional[Tuple[str, ...]] = None,
    ) -> dace.data.Array:
        aligned_index, shape, dtype, dimensions = storage_utils.normalize_storage_spec(
            aligned_index, shape, dtype, dimensions
        )
        itemsize = dtype.itemsize
        layout_map = gt_backend.from_name(backend).storage_info["layout_map"](dimensions)

        order_idx = storage_utils.idx_from_order([i for i in layout_map if i is not None])
        padded_shape = storage_utils.compute_padded_shape(
            shape, gt_backend.from_name(backend).storage_info["alignment"], order_idx
        )

        strides = storage_utils.strides_from_padded_shape(padded_shape, order_idx, itemsize)

        storage = (
            dace.StorageType.GPU_Global
            if gt_backend.from_name(backend).storage_info["device"] == "gpu"
            else dace.StorageType.CPU_Heap
        )
        start_offset = int(np.array([aligned_index]) @ np.array([strides]).T) // itemsize

        total_size = int(int(np.array([shape]) @ np.array([strides]).T) // itemsize)

        start_offset = start_offset % gt_backend.from_name(backend).storage_info["alignment"]
        return dace.data.Array(
            shape=shape,
            strides=[s // itemsize for s in strides],
            dtype=dace.typeclass(str(dtype)),
            storage=storage,
            total_size=total_size,
            start_offset=start_offset,
        )
