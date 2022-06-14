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

import itertools
from typing import Any, Dict, Tuple

import numpy as np


try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import dace
except ImportError:
    dace = None

from gt4py import backend as gt_backend
from gtc import utils as gtc_utils

from . import utils as storage_utils


def _error_on_invalid_backend(backend):
    if backend not in gt_backend.REGISTRY:
        raise RuntimeError(f"Backend '{backend}' is not registered.")


def empty(backend, default_origin, shape, dtype, mask=None):
    _error_on_invalid_backend(backend)
    if gt_backend.from_name(backend).storage_info["device"] == "gpu":
        allocate_f = storage_utils.allocate_gpu
    else:
        allocate_f = storage_utils.allocate_cpu

    default_origin, shape, dtype, mask = storage_utils.normalize_storage_spec(
        default_origin, shape, dtype, mask
    )

    _error_on_invalid_backend(backend)

    alignment = gt_backend.from_name(backend).storage_info["alignment"]
    layout_map = gt_backend.from_name(backend).storage_info["layout_map"](mask)

    dtype = np.dtype(dtype)
    _, res = allocate_f(
        default_origin, shape, layout_map, dtype, alignment * dtype.itemsize
    )
    return res




def ones(backend, default_origin, shape, dtype, mask=None):
    storage = empty(
        shape=shape,
        dtype=dtype,
        backend=backend,
        default_origin=default_origin,
        mask=mask,
    )
    storage[...] = 1
    return storage


def zeros(backend, default_origin, shape, dtype, mask=None):
    storage = empty(
        shape=shape,
        dtype=dtype,
        backend=backend,
        default_origin=default_origin,
        mask=mask,
    )
    storage[...] = 0
    return storage


def from_array(
    data, backend, default_origin, shape=None, dtype=None, mask=None
):
    is_cupy_array = cp is not None and isinstance(data, cp.ndarray)
    xp = cp if is_cupy_array else np
    if shape is None:
        shape = xp.asarray(data).shape
    if dtype is None:
        dtype = xp.asarray(data).dtype
    storage = empty(
        shape=shape,
        dtype=dtype,
        backend=backend,
        default_origin=default_origin,
        mask=mask,
    )
    if is_cupy_array:
        if isinstance(storage, cp.ndarray):
            storage[...] = data
        else:
            storage[...] = cp.asnumpy(data)
    else:
        storage[...] = data

    return storage

def dace_descriptor(
    backend, default_origin, shape, dtype, mask=None
):
    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize
    items_per_alignment = int(

    order_idx = idx_from_order([i for i in layout_map if i is not None])
    padded_shape = compute_padded_shape(shape, gt_backend.from_name(backend).storage_info["alignment"], order_idx)

    strides = strides_from_padded_shape(padded_shape, order_idx, itemsize)

    storage = dace.StorageType.GPU_Global if gt_backend.from_name(backend).storage_info["device"] == "gpu" else dace.StorageType.CPU_Heap
    start_offset = int(np.array([default_origin]) @ np.array([strides]).T) // itemsize


    total_size = int(int(np.array([shape]) @ np.array([strides]).T) // itemsize)

    start_offset = (
        start_offset % gt_backend.from_name(backend).storage_info["alignment"]
    )
    return dace.data.Array(
        shape=shape,
        strides=[s // itemsize for s in strides],
        dtype=dace.typeclass(str(dtype)),
        storage=storage,
        total_size=total_size,
        start_offset=start_offset,
    )

