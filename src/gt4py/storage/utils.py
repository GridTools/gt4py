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

import collections
import math
import numbers
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np

import gt4py.utils as gt_util


try:
    import cupy as cp
    from cupy.lib.stride_tricks import as_strided
except ImportError:
    cp = None

import gt4py.ir as gt_ir
import gt4py.utils as gt_utils
from gt4py.storage.default_parameters import get_default_parameters
from gt4py.storage.definitions import (
    CudaManagedGPUStorage,
    ExplicitlyManagedGPUStorage,
    Storage,
    SyncState,
)


def idx_from_order(order):
    return list(np.argsort(order))


def check_mask(mask):
    if not gt_util.is_iterable_of(mask, bool) and mask is not None:
        raise TypeError("Mask must be an iterable of booleans.")


def normalize_shape(shape, mask=None):
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

    return tuple(new_shape)


def is_cuda_managed(data):
    if hasattr(data, "__cuda_array_interface__"):
        attrs = cp.cuda.runtime.pointerGetAttributes(data.__cuda_array_interface__["data"][0])
        return attrs.devicePointer == attrs.hostPointer
    else:
        try:
            tmp_array = np.asarray(data)
        except Exception:
            return False
        else:
            try:
                attrs = cp.cuda.runtime.pointerGetAttributes(tmp_array.ctypes.data)
                return attrs.devicePointer == attrs.hostPointer and attrs.devicePointer != 0
            except Exception:
                return False


def normalize_halo(halo) -> Optional[Tuple[Tuple[int, int], ...]]:
    if halo is None:
        return None

    if not isinstance(halo, Sequence) or not all(
        (
            isinstance(h, numbers.Integral)
            or (
                isinstance(h, Sequence)
                and len(h) == 2
                and isinstance(h[0], numbers.Integral)
                and isinstance(h[1], numbers.Integral)
            )
        )
        for h in halo
    ):
        raise TypeError("halo must be a tuple of ints or pairs of ints.")
    halo = tuple(tuple(h) if isinstance(h, Sequence) else (h, h) for h in halo)

    if any(h[0] < 0 or h[1] < 0 for h in halo):
        raise ValueError("halo ({}) contains negative value.".format(halo))

    return tuple(halo)


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


def get_ptr(array):
    if isinstance(array, np.ndarray):
        return array.ctypes.data
    else:
        return array.data.ptr


def _reshape(array, strides, padded_shape):
    if isinstance(array, np.ndarray):
        field = np.reshape(array, padded_shape)
        if field.ndim > 0:
            field.strides = strides
    else:
        field = cp.reshape(array, padded_shape)
        if field.ndim > 0:
            field = as_strided(field, strides=strides)
    return field


def allocate(default_origin, shape, layout_map, dtype, alignment_bytes, allocate_f):
    dtype = np.dtype(dtype)
    assert (
        alignment_bytes % dtype.itemsize
    ) == 0, "Alignment must be a multiple of byte-width of dtype."
    itemsize = dtype.itemsize
    items_per_alignment = int(alignment_bytes / itemsize)

    if len(shape) == 0 or np.product(shape) == 0:
        array, raw_buffer = allocate_f(shape, dtype=dtype)
        return raw_buffer, array[...]

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

    allocation_mismatch = int((get_ptr(array) % alignment_bytes) / itemsize)

    alignment_offset = (halo_offset - allocation_mismatch) % items_per_alignment
    field = _reshape(
        array[alignment_offset : alignment_offset + padded_size],  # noqa: E203
        strides,
        padded_shape,
    )
    field = field[tuple(slice(0, s, None) for s in shape)]
    return raw_buffer, field


def has_cpu_buffer(data):
    if data is None or isinstance(data, numbers.Number):
        return False
    elif hasattr(data, "__gt_data_interface__"):
        return "cpu" in data.__gt_data_interface__
    else:
        try:
            np.asarray(data)
        except Exception:
            try:
                tmp_array = cp.asarray(data)
                attrs = cp.cuda.runtime.pointerGetAttributes(tmp_array.data.ptr)
                return attrs.hostPointer == attrs.devicePointer
            except Exception:
                return False
        else:
            return True


def has_gpu_buffer(data):
    if data is None or isinstance(data, numbers.Number):
        return False
    elif hasattr(data, "__gt_data_interface__"):
        return "gpu" in data.__gt_data_interface__
    else:
        try:
            cp.asarray(SimpleNamespace(__cuda_array_interface__=data.__cuda_array_interface__))
        except Exception:
            try:
                cp.asarray(SimpleNamespace(__cuda_array_interface__=data.__array_interface__))
            except Exception:
                return False
            else:
                return True
        else:
            return True


def as_np_array(data):
    if data is None:
        return None
    if not has_cpu_buffer(data):
        return None

    if hasattr(data, "__gt_data_interface__"):
        return np.asarray(SimpleNamespace(__array_interface__=data.__gt_data_interface__["cpu"]))
    else:
        try:
            return np.asarray(data)
        except Exception:
            return np.asarray(SimpleNamespace(__array_interface__=data.__cuda_array_interface__))


def as_cp_array(data):
    if data is None:
        return None
    if not has_gpu_buffer(data):
        return None

    if hasattr(data, "__gt_data_interface__"):
        return cp.asarray(
            SimpleNamespace(__cuda_array_interface__=data.__gt_data_interface__["gpu"])
        )
    else:
        try:
            # cp.asarray(data) also accepts cpu buffers
            return cp.asarray(
                SimpleNamespace(__cuda_array_interface__=data.__cuda_array_interface__)
            )
        except Exception:
            # in case of managed storage, this doesn't raise
            return cp.asarray(SimpleNamespace(__cuda_array_interface__=data.__array_interface__))


def get_buffers(data, device_data):
    data_cpu = as_np_array(data)
    data_gpu = as_cp_array(data)
    device_data_cpu = as_np_array(device_data)
    device_data_gpu = as_cp_array(device_data)

    res_cpu = None
    res_gpu = None

    if device_data_gpu is not None:
        res_gpu = device_data_gpu
    else:
        res_gpu = data_gpu
    if data_cpu is not None:
        res_cpu = data_cpu
    else:
        res_cpu = device_data_cpu

    return res_cpu, res_gpu


def normalize_data_and_device_data(data, device_data, device, managed, copy):
    if isinstance(data, numbers.Number):
        if device_data is not None:
            raise ValueError("If data is a scalar, device_data must not be provided")
        return data, None

    cpu_buffer, gpu_buffer = get_buffers(data, device_data)

    return cpu_buffer, gpu_buffer


def _is_contiguous(data):
    all(
        (st % data.itemsize) == 0 and (st // data.itemsize) == sh
        for st, sh in zip(data.strides, data.shape)
    )


def parameter_lookup_and_normalize(
    *,
    aligned_index: Optional[Sequence[int]],
    alignment_size: Optional[int],
    data: Any,
    dims: Optional[Sequence[str]],
    copy: bool,
    defaults: Optional[str],
    device: Optional[str],
    device_data: Any = None,
    dtype: Any,
    halo: Optional[Sequence[Union[int, Tuple[int, int]]]],
    layout: Optional[Union[Callable, Sequence[int]]],
    managed: Optional[Union[bool, str]],
    shape: Optional[Sequence[int]] = None,
    sync_state: Optional[SyncState] = None,
    template: Any,
):
    if data is not None:

        if not isinstance(data, Storage) and not hasattr(data, "__gt_data_interface__"):
            try:
                np.asarray(data)
            except Exception:
                try:
                    cp.asarray(data)
                except Exception:
                    raise TypeError("'data' not understood as array")

    if device_data is not None:
        if not isinstance(device_data, Storage) and not hasattr(data, "__gt_data_interface__"):
            try:
                cp.asarray(device_data)
            except Exception:
                raise TypeError("'device_data' not understood as gpu array")

    if template is not None:
        if not isinstance(template, Storage) and not hasattr(template, "__gt_data_interface__"):
            try:
                tmp_array = np.asarray(template)
                assert "O" not in tmp_array.dtype.str
            except Exception:
                if not hasattr(template, "__cuda_array_interface__"):
                    raise TypeError("'template' not understood as array")
    if shape is not None:
        if not gt_utils.is_iterable_of(shape, numbers.Integral):
            raise TypeError("'shape' must be an iterable of integers")
        elif not all(map(lambda x: x >= 0, shape)):
            raise ValueError("shape contains negative values")
        shape = tuple(int(s) for s in shape)

    if defaults is not None:
        from .default_parameters import REGISTRY as default_parameter_registry

        if not isinstance(defaults, str):
            raise TypeError("'defaults' must be a string")
        elif defaults not in default_parameter_registry:
            raise ValueError(f"'defaults' must be in {list(default_parameter_registry.keys())}")

    if dtype is not None:
        if isinstance(dtype, gt_ir.DataType):
            dtype = dtype.dtype
        else:
            try:
                dtype = np.dtype(dtype)
            except Exception:
                raise TypeError("'dtype' not understood ")

    if dims is not None:
        if not gt_utils.is_iterable_of(
            dims, iterable_class=collections.abc.Sequence, item_class=(str, int)
        ) or not len(set(str(a) for a in dims)) == len(dims):
            raise TypeError("'axes' must be a sequence of unique characters or integers")
        for a in dims:
            if not str(a).isdecimal() and a not in ["I", "J", "K"]:
                raise ValueError(
                    f"'axes' must only contain integers or " f"characters in {['I', 'J', 'K']}"
                )
        dims = list(str(d) for d in dims)
    else:
        dims = ["I", "J", "K"]

    if alignment_size is not None:
        if not isinstance(alignment_size, int):
            raise TypeError("'alignment_size' must be an integer")
        if not alignment_size > 0:
            raise ValueError("'alignment_size' must be positive")
    if device not in [None, "cpu", "gpu"]:
        raise TypeError(f"device {device} not supported")

    if layout is not None:
        if not gt_utils.is_iterable_of(
            layout, numbers.Integral, iterable_class=collections.abc.Sequence
        ) and not callable(layout):
            raise TypeError(
                "'layout_map' must either be a sequence of integers"
                " or a callable returning such a sequence when given 'dims'"
            )

    if not isinstance(copy, bool):
        raise TypeError("'copy' must be a boolean")

    if managed is not None:
        if not isinstance(managed, str) and managed is not False:
            raise TypeError("'managed' must be a string or 'False'")
        elif managed not in ["cuda", "gt4py", False]:
            raise ValueError('\'managed\' must be in ["cuda", "gt4py", False]')
    if sync_state is not None:
        if not isinstance(sync_state, SyncState):
            raise TypeError("'sync_state' must be an instance of SyncState.")

    default_parameters = get_default_parameters(defaults) if defaults is not None else None
    if default_parameters is not None:
        if device is None:
            device = default_parameters.device
        if alignment_size is None:
            alignment_size = default_parameters.alignment_size
        if layout is None:
            layout = default_parameters.layout

    if device is None and template is not None:
        if has_gpu_buffer(template):
            device = "gpu"
        elif has_cpu_buffer(template):
            device = "cpu"

    if device is None and default_parameters is not None and default_parameters.device is not None:
        device = default_parameters.device

    if template is None:
        if data is not None and not isinstance(data, numbers.Number):
            template = data
        else:
            template = device_data

    data, device_data = normalize_data_and_device_data(data, device_data, device, managed, copy)

    if device is None:
        if device_data is not None:
            device = "gpu"
        else:
            device = "cpu"

    if template is None:
        if data is not None and not isinstance(data, numbers.Number):
            template = data
        elif device_data is not None:
            template = device_data
        else:
            template = None
    if device is None and template is not None:
        if isinstance(template, Storage):
            device = template.device
        else:
            if hasattr(data, "__cuda_array_interface__") or hasattr(
                device_data, "__cuda_array_interface__"
            ):
                device = "gpu"

    if template is not None:
        if dtype is None:
            dtype = template.dtype
        if managed is None:
            if isinstance(template, CudaManagedGPUStorage):
                managed = "cuda"
            elif isinstance(template, ExplicitlyManagedGPUStorage):
                managed = "gt4py"
            elif isinstance(template, Storage):
                managed = False
            elif cp is not None and is_cuda_managed(template):
                managed = "cuda"

    if shape is None:
        if template is not None:
            shape = template.shape
        elif data is not None and not isinstance(data, numbers.Number):
            shape = data.shape
        elif device_data is not None:
            shape = device_data.shape

    if isinstance(template, Storage):
        ndim = template.ndim
    elif isinstance(data, Storage):
        ndim = device_data.ndim
    elif isinstance(device_data, Storage):
        ndim = device_data.ndim
    elif shape is not None:
        ndim = len(shape)
    else:
        raise TypeError("not enough information to determine the number of dimensions")

    if template is not None and layout is None:
        if layout is None:
            layout = layout_from_strides(
                template.strides if template.strides is not None else tuple(range(ndim))
            )

    normalized_halo = normalize_halo(halo)
    if normalized_halo is None:
        normalized_halo = ((0, 0),) * ndim
    if aligned_index is None:
        aligned_index = tuple(int(h[0]) for h in normalized_halo)
    aligned_index = tuple(int(a) for a in aligned_index)

    if layout is not None:
        if not gt_utils.is_iterable_of(
            layout, iterable_class=collections.abc.Sequence, item_class=numbers.Integral
        ):
            assert callable(layout)
            layout = layout(dims)
            if not gt_utils.is_iterable_of(
                layout, iterable_class=collections.abc.Sequence, item_class=numbers.Integral
            ):
                raise TypeError(
                    f"'layout_map' did not return an iterable of integers for ndim={ndim}"
                )
        assert isinstance(layout, collections.abc.Sequence)
        tmp_layout: Sequence[int] = layout
        if (
            not all(item >= 0 for item in tmp_layout)
            or not all(item < len(tmp_layout) for item in tmp_layout)
            or not len(set(tmp_layout)) == len(tmp_layout)
        ):
            raise ValueError(
                "elements of layout map must be a permutation of (0, ..., len(layout_map))"
            )
        normalized_layout = tuple(int(p) for p in tmp_layout)
    else:
        normalized_layout = tuple(range(ndim))
    assert isinstance(ndim, int) and ndim >= 0

    if alignment_size is None:
        alignment_size = 1
    if aligned_index is None:
        aligned_index = ndim * (0,)
    if dtype is None:
        dtype = np.dtype("float64")
    if managed is None:
        managed = False

    if managed == "gt4py":
        if sync_state is None:
            if hasattr(template, "sync_state"):
                sync_state = template.sync_state
            else:
                sync_state = SyncState()

        if copy:
            state = sync_state.state
            sync_state = SyncState()
            sync_state.state = state

    assert gt_utils.is_iterable_of(shape, item_class=int, iterable_class=tuple)

    return dict(
        aligned_index=aligned_index,
        alignment_size=alignment_size,
        data=data,
        copy=copy,
        device=device,
        device_data=device_data,
        dtype=dtype,
        halo=normalized_halo,
        layout=normalized_layout,
        managed=managed,
        shape=shape,
        sync_state=sync_state,
    )


def allocate_cpu(aligned_index, shape, layout_map, dtype, alignment_bytes):
    def allocate_f(shape, dtype):
        raw_buffer = np.empty(shape, dtype)
        return raw_buffer, raw_buffer

    return allocate(aligned_index, shape, layout_map, dtype, alignment_bytes, allocate_f)


def allocate_gpu_cuda_managed(aligned_index, shape, layout_map, dtype, alignment_bytes):
    def allocate_f(shape, dtype):
        allocator = cp.cuda.get_allocator()
        cp.cuda.set_allocator(cp.cuda.malloc_managed)
        device_buffer = cp.empty(shape, dtype)
        array = _cpu_view(device_buffer)
        cp.cuda.set_allocator(allocator)
        return array, device_buffer

    return allocate(aligned_index, shape, layout_map, dtype, alignment_bytes, allocate_f)


def allocate_gpu_only(aligned_index, shape, layout_map, dtype, alignment_bytes):
    def allocate_f(shape, dtype):
        raw_buffer = cp.empty(shape, dtype)
        return raw_buffer, raw_buffer

    return allocate(aligned_index, shape, layout_map, dtype, alignment_bytes, allocate_f)


def allocate_gpu_gt4py_managed(aligned_index, shape, layout_map, dtype, alignment_bytes):
    cpu_buffers = allocate_cpu(aligned_index, shape, layout_map, dtype, alignment_bytes)
    gpu_buffers = allocate_gpu_only(aligned_index, shape, layout_map, dtype, alignment_bytes)
    return (*cpu_buffers, *gpu_buffers)


def _gpu_view(cpu_array):
    if 0 in cpu_array.shape:
        res = cp.asarray(cpu_array)
        return as_strided(res, shape=cpu_array.shape, strides=cpu_array.strides)
    array_interface = cpu_array.__array_interface__
    array_interface["version"] = 2
    array_interface["strides"] = cpu_array.strides
    array_interface.pop("offset", None)
    return cp.asarray(SimpleNamespace(__cuda_array_interface__=array_interface))


def _cpu_view(gpu_array):
    if 0 in gpu_array.shape:
        return gpu_array.get()
    array_interface = gpu_array.__cuda_array_interface__
    array_interface["version"] = 3
    return np.asarray(SimpleNamespace(__array_interface__=array_interface))


def is_compatible_layout(strides, shape, layout_map):
    if len(shape) < len(layout_map):
        return False

    layout_map = tuple(int(lm) for lm, sh in zip(layout_map, shape) if sh > 1)
    if strides is None:
        return tuple(int(lm) for lm in sorted(layout_map)) == layout_map
    else:
        strides = tuple(int(st) for st, sh in zip(strides, shape) if sh > 1)
        if len(strides) < len(layout_map):
            return False

    stride = 0
    for dim in reversed(np.argsort(layout_map)):
        if strides[dim] < stride:
            return False
        stride = strides[dim]
    return True


def layout_from_strides(strides):
    res = np.argsort(np.argsort(strides))
    res = -res + res.max()
    return tuple(int(r) for r in res)
