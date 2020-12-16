# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
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

import numpy as np


try:
    import cupy as cp
except ImportError:
    cp = None

import gt4py
from gt4py import backend as gt_backend

from . import utils as storage_utils


def empty(backend, default_origin, shape, dtype, mask=None, *, managed_memory=False):

    return Storage(
        shape=shape,
        dtype=dtype,
        backend=backend,
        default_origin=default_origin,
        mask=mask,
        managed_memory=managed_memory,
    )


def ones(backend, default_origin, shape, dtype, mask=None, *, managed_memory=False):
    storage = empty(
        shape=shape,
        dtype=dtype,
        backend=backend,
        default_origin=default_origin,
        mask=mask,
        managed_memory=managed_memory,
    )
    storage[...] = 1
    return storage


def zeros(backend, default_origin, shape, dtype, mask=None, *, managed_memory=False):
    storage = empty(
        shape=shape,
        dtype=dtype,
        backend=backend,
        default_origin=default_origin,
        mask=mask,
        managed_memory=managed_memory,
    )
    storage[...] = 0
    return storage


def as_storage(*args, **kwargs):
    return storage(*args, copy=False, **kwargs)


def storage(
    data=None,
    backend=None,
    default_origin=None,
    shape=None,
    dtype=None,
    mask=None,
    *,
    copy=True,
    managed_memory=False,
):
    if not copy:
        assert isinstance(data, Storage)
        assert default_origin is None
        default_origin = data.default_origin
        assert shape is None
        shape = data.shape
        assert dtype is None
        dtype = data.dtype
        assert mask is None
        mask = data.mask
        assert not managed_memory
        managed_memory = isinstance(data, gt4py.storage.definitions.CudaManagedGPUStorage)

        return Storage(
            data=data,
            copy=False,
            shape=shape,
            dtype=dtype,
            backend=backend,
            default_origin=default_origin,
            mask=mask,
            managed_memory=managed_memory,
        )

    else:
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
            managed_memory=managed_memory,
        )
        if is_cupy_array:
            if isinstance(storage, CudaManagedGPUStorage) or isinstance(
                storage, ExplicitlyManagedGPUStorage
            ):
                tmp = storage_utils.gpu_view(storage)
                tmp[...] = data
            else:
                storage[...] = cp.asnumpy(data)
        else:
            storage[...] = data

        return storage


class Storage:
    """
    Storage class based on a numpy (CPU) or cupy (GPU) array.

    Takes care of proper memory alignment, with additional information that is required by the
    backends.
    """

    __array_subok__ = True

    def __new__(
        cls,
        shape,
        dtype,
        backend,
        default_origin,
        mask=None,
        managed_memory=False,
        data=None,
        copy=True,
    ):
        """
        Storage constructor, not intended to be called directly by user.

        Parameters
        ----------
        shape: tuple of ints
            the shape of the storage

        dtype: data type compatible with numpy dtypes
            supported are the floating point and integer dtypes of numpy

        backend: string, backend identifier
            Currently possible: debug, numpy, gtx86, gtmc, gtcuda

        default_origin: tuple of ints
            determines the point to which the storage memory address is aligned.
            for performance, this should be the coordinates of the most common origin
            at call time.
            when calling a stencil and no origin is specified, the default_origin is used.

        mask: list of booleans
            False entries indicate that the corresponding dimension is masked, i.e. the storage
            has reduced dimension and reading and writing from offsets along this axis acces the
            same element.
        """
        if copy:
            if mask is None:
                mask = [True] * len(shape)
            default_origin = storage_utils.normalize_default_origin(default_origin, mask)
            shape = storage_utils.normalize_shape(shape, mask)

            if backend not in gt_backend.REGISTRY:
                ValueError("Backend must be in {}.".format(gt_backend.REGISTRY))

            alignment = gt_backend.from_name(backend).storage_info["alignment"]
            layout_map = gt_backend.from_name(backend).storage_info["layout_map"](mask)

            if gt_backend.from_name(backend).storage_info["device"] == "gpu":
                if managed_memory:
                    storage_t = CudaManagedGPUStorage
                else:
                    storage_t = ExplicitlyManagedGPUStorage
            else:
                storage_t = CPUStorage
            obj = storage_t.__new__(
                storage_t,
                backend,
                np.dtype(dtype),
                default_origin,
                shape,
                alignment,
                layout_map,
                data,
                copy,
            )
        else:
            assert isinstance(data, gt4py.storage.Storage)
            storage_t = type(data)
            obj = storage_t.__new__(
                storage_t, None, np.dtype(dtype), default_origin, shape, None, None, data, copy
            )

        obj._backend = backend
        obj._mask = mask
        obj._shape = shape
        obj.default_origin = default_origin
        obj._dtype = np.dtype(dtype)
        obj._check_data()

        return obj

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def backend(self):
        return self._backend

    @property
    def mask(self):
        return self._mask

    @property
    def layout_map(self):
        return gt_backend.from_name(self.backend).storage_info["layout_map"](self.mask)

    @property
    def dtype(self):
        return self._dtype

    @property
    def itemsize(self):
        return self._dtype.itemsize

    @property
    def size(self):
        return int(np.prod(self.shape))

    def __deepcopy__(self, memo={}):
        return self.copy()

    def __setitem__(self, key, value):
        self._forward_setitem(key, value)

    def __getitem__(self, key):

        if key is Ellipsis:
            return gt4py.storage.as_storage(self)

        if not isinstance(key, tuple):
            key = (key,)

        if any(isinstance(k, slice) for k in key):
            res = gt4py.storage.as_storage(self)
            res._forward_getitem_slice(key)
            res._shape = res._field.shape
            return res
        else:
            return self._forward_getitem_scalar(key)

    def copy(self):
        return storage(
            data=self,
        )

    def synchronize(self):
        pass

    def host_to_device(self, force=False):
        pass

    def device_to_host(self, force=False):
        pass

    def __iconcat__(self, other):
        raise NotImplementedError("Concatenation of Storages is not supported")


class CudaManagedGPUStorage(Storage):
    @property
    def __cuda_array_interface__(self):
        array_interface = self.__array_interface__
        array_interface["version"] = 2
        array_interface["strides"] = self.strides
        array_interface.pop("offset", None)
        return array_interface

    def copy(self):
        res = super().copy()
        res.gpu_view[...] = self.gpu_view
        cp.cuda.Device(0).synchronize()
        return res

    def __setitem__(self, key, value):
        if hasattr(value, "__cuda_array_interface__"):
            gpu_view = storage_utils.gpu_view(self)
            gpu_view[key] = cp.asarray(value.data)
            cp.cuda.Device(0).synchronize()
            return value
        else:
            return super().__setitem__(key, value)


class CPUStorage(Storage):
    @property
    def _ptr(self):
        return self._ndarray.ctypes.data

    def __new__(cls, backend, dtype, default_origin, shape, alignment, layout_map, data, copy):
        if copy:
            (raw_buffer, field) = storage_utils.allocate_cpu(
                default_origin, shape, layout_map, dtype, alignment * dtype.itemsize
            )
        else:
            assert isinstance(data, CPUStorage)
            raw_buffer, field = data._raw_buffer, data._field
        self = super(Storage, cls).__new__(
            CPUStorage,
        )
        self._raw_buffer = raw_buffer
        self._field = field
        self.default_origin = default_origin
        return self

    def _check_data(self):
        # check that memory of field is within raw_buffer and that field is a view of raw_buffer
        if (
            not self._field.ctypes.data >= self._raw_buffer.ctypes.data
            and self._field.ctypes.data + self.itemsize * (self.size - 1)
            <= self._raw_buffer[-1:].ctypes.data
            and self._field.base is not self._raw_buffer
        ):
            raise Exception("The buffers are in an inconsistent state.")

    @property
    def data(self):
        return self._field.data

    @property
    def strides(self):
        return self._field.strides

    def copy(self):
        res = super().copy()
        res[...] = self
        return res

    def _forward_setitem(self, key, value):
        self._field.__setitem__(key, value)

    def _forward_getitem_slice(self, key):
        self._field = self._field.__getitem__(key)

    def _forward_getitem_scalar(self, key):
        return self._field.__getitem__(key)

    @property
    def __array_interface__(self):
        return self._field.__array_interface__

    def transpose(self, *axes):
        res: ExplicitlyManagedGPUStorage = as_storage(self)
        res._field = res._field.transpose(*axes)
        return res


class ExplicitlyManagedGPUStorage(Storage):
    class SyncState:
        SYNC_CLEAN = 0
        SYNC_HOST_DIRTY = 1
        SYNC_DEVICE_DIRTY = 2

        def __init__(self):
            self.state = self.SYNC_CLEAN

    def __init__(self, *args, **kwargs):
        self._sync_state = self.SyncState()

    def copy(self):
        res = super().copy()
        res._sync_state = self.SyncState()
        res._sync_state.state = self._sync_state.state
        res[...] = self
        res._device_field[...] = self._device_field
        return res

    def __new__(cls, backend, dtype, default_origin, shape, alignment, layout_map, data, copy):
        if copy:
            (
                raw_buffer,
                field,
                device_raw_buffer,
                device_field,
            ) = storage_utils.allocate_gpu_unmanaged(
                default_origin, shape, layout_map, dtype, alignment * dtype.itemsize
            )
        else:
            assert isinstance(data, ExplicitlyManagedGPUStorage)
            (raw_buffer, field, device_raw_buffer, device_field,) = (
                data._raw_buffer,
                data._field,
                data._device_raw_buffer,
                data._device_field,
            )
        self = super(Storage, cls).__new__(
            ExplicitlyManagedGPUStorage,
        )
        self._raw_buffer = raw_buffer
        self._field = field
        self._device_raw_buffer = device_raw_buffer
        self._device_field = device_field
        self.default_origin = default_origin
        return self

    @property
    def data(self):
        return self._field.data

    @property
    def strides(self):
        return self._field.strides

    @property
    def device_data(self):
        return self._device_field.data

    def synchronize(self):
        if self._is_host_modified:
            self.host_to_device()
        elif self._is_device_modified:
            self.device_to_host()
        self._set_clean()

    def host_to_device(self, force=False):
        if force or self._is_host_modified:
            self._set_clean()
            self._device_raw_buffer.set(self._raw_buffer)

    def device_to_host(self, force=False):
        if force or self._is_device_modified:
            self._set_clean()
            self._device_raw_buffer.get(out=self._raw_buffer)

    @property
    def _is_clean(self):
        return self._sync_state.state == self.SyncState.SYNC_CLEAN

    @property
    def _is_host_modified(self):
        return self._sync_state.state == self.SyncState.SYNC_HOST_DIRTY

    @property
    def _is_device_modified(self):
        return self._sync_state.state == self.SyncState.SYNC_DEVICE_DIRTY

    def _set_clean(self):
        self._sync_state.state = self.SyncState.SYNC_CLEAN

    def _set_host_modified(self):
        self._sync_state.state = self.SyncState.SYNC_HOST_DIRTY

    def _set_device_modified(self):
        self._sync_state.state = self.SyncState.SYNC_DEVICE_DIRTY

    def _forward_getitem_slice(self, key):
        self._field = self._field.__getitem__(key)
        self._device_field = self._device_field.__getitem__(key)

    def _forward_getitem_scalar(self, key):
        if self._is_device_modified:
            return self.dtype.type(self._device_field.__getitem__(key))
        else:
            return self._field.__getitem__(key)

    def _forward_setitem(self, key, value):
        if isinstance(value, ExplicitlyManagedGPUStorage):
            if (self._is_clean or self._is_device_modified) and (
                value._is_clean or value._is_device_modified
            ):
                self._set_device_modified()
                self._device_field[key] = value._device_field
                return value
            elif (self._is_clean or self._is_host_modified) and (
                value._is_clean or value._is_host_modified
            ):
                self._set_host_modified()
                return self._field.__setitem__(key, value)
            else:
                if self._is_host_modified:
                    self.host_to_device()
                else:
                    value.host_to_device()
                self._set_device_modified()
                self._device_field.__setitem__(key, value._device_field)
                return value
        elif hasattr(value, "__cuda_array_interface__"):
            if self._is_host_modified:
                self.host_to_device()
            self._set_device_modified()
            return self._device_field.__setitem__(key, value)
        else:
            if self._is_device_modified:
                self.device_to_host()
            self._set_host_modified()
            return self._field.__setitem__(key, value)

    @property
    def sync_state(self):
        return self._sync_state.state

    @property
    def _ptr(self):
        return self.data.ptr

    def _check_data(self):

        # check that memory of field is within raw_buffer
        if (
            not self._field.ctypes.data >= self._raw_buffer.ctypes.data
            and self._field.ctypes.data + self.itemsize * (self.size - 1)
            <= self._raw_buffer[-1:].ctypes.data
            and self._field.base is not self._raw_buffer
        ):
            raise Exception("The buffers are in an inconsistent state.")

        # check that memory of field is within raw_buffer
        if (
            not self._device_field.data.ptr >= self._device_raw_buffer.data.ptr
            and self._device_field.data.ptr + self.itemsize * (self.size - 1)
            <= self._device_raw_buffer[-1:].data.ptr
        ):
            raise Exception("The buffers are in an inconsistent state.")

    def transpose(self, *axes):
        res: ExplicitlyManagedGPUStorage = as_storage(self)
        res._field = res._field.transpose(*axes)
        res._device_field = cp.lib.stride_tricks.as_strided(
            res._device_raw_buffer, shape=res._field.shape, strides=res._field.strides
        )
        return res

    @property
    def __cuda_array_interface__(self):
        self.host_to_device()
        res = self._device_field.__cuda_array_interface__
        res["strides"] = self.strides
        return res

    @property
    def __array_interface__(self):
        self.device_to_host()
        return self._field.__array_interface__
