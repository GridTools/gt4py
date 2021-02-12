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

import numpy as np


try:
    import cupy as cp
except ImportError:
    cp = None

from gt4py import backend as gt_backend

from . import utils as storage_utils


def empty(backend, default_origin, shape, dtype, mask=None, *, managed_memory=False):
    if gt_backend.from_name(backend).storage_info["device"] == "gpu":
        if managed_memory:
            storage_t = GPUStorage
        else:
            storage_t = ExplicitlySyncedGPUStorage
    else:
        storage_t = CPUStorage

    return storage_t(
        shape=shape, dtype=dtype, backend=backend, default_origin=default_origin, mask=mask
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


def from_array(
    data, backend, default_origin, shape=None, dtype=None, mask=None, *, managed_memory=False
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
        managed_memory=managed_memory,
    )
    if is_cupy_array:
        if isinstance(storage, GPUStorage) or isinstance(storage, ExplicitlySyncedGPUStorage):
            tmp = storage_utils.gpu_view(storage)
            tmp[...] = data
        else:
            storage[...] = cp.asnumpy(data)
    else:
        storage[...] = data

    return storage


class Storage(np.ndarray):
    """
    Storage class based on a numpy (CPU) or cupy (GPU) array, taking care of proper memory alignment, with additional
    information that is required by the backends.
    """

    __array_subok__ = True

    def __new__(cls, shape, dtype, backend, default_origin, mask=None):
        """
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
            has reduced dimension and reading and writing from offsets along this axis acces the same element.
        """

        if mask is None:
            mask = [True] * len(shape)
        default_origin = tuple(storage_utils.normalize_default_origin(default_origin, mask))
        shape = tuple(storage_utils.normalize_shape(shape, mask))

        if not backend in gt_backend.REGISTRY:
            ValueError("Backend must be in {}.".format(gt_backend.REGISTRY))

        alignment = gt_backend.from_name(backend).storage_info["alignment"]
        layout_map = gt_backend.from_name(backend).storage_info["layout_map"](mask)

        obj = cls._construct(backend, np.dtype(dtype), default_origin, shape, alignment, layout_map)
        obj._backend = backend
        obj.is_stencil_view = True
        obj._mask = mask
        obj._check_data()

        return obj

    @property
    def backend(self):
        """The backend identifier string of the storage."""
        return self._backend

    @property
    def mask(self):
        """
        Iterable of booleans.

        Dimensions where the corresponding entry is `False` are ignored.
        """
        return self._mask

    @property
    def layout_map(self):
        return gt_backend.from_name(self.backend).storage_info["layout_map"](self.mask)

    def transpose(self, *axes):
        res = super().transpose(*axes)
        if res._is_consistent(self):
            res.is_stencil_view = self.is_stencil_view
        else:
            res.is_stencil_view = False
        return res

    def __deepcopy__(self, memo={}):
        return self.copy()

    def copy(self):
        res = empty(
            shape=self.shape,
            dtype=self.dtype,
            backend=self.backend,
            default_origin=self.default_origin,
            mask=self.mask,
            managed_memory=not isinstance(self, ExplicitlySyncedGPUStorage),
        )
        res.is_stencil_view = self.is_stencil_view
        return res

    def __array_finalize__(self, obj):
        if obj is None:
            # constructor called previously
            return
        else:
            if self.base is None:
                # case np.array
                raise RuntimeError(
                    "Copying storages is only possible through Storage.copy() or deepcopy."
                )
            else:
                if not isinstance(obj, Storage) and not isinstance(obj, _ViewableNdarray):
                    raise RuntimeError(
                        "Meta information can not be inferred when creating Storage views from other classes than Storage."
                    )
                self.__dict__ = {**obj.__dict__, **self.__dict__}
                self.is_stencil_view = False
                if not hasattr(obj, "default_origin"):
                    self.is_stencil_view = True
                elif self._is_consistent(obj):
                    self.is_stencil_view = obj.is_stencil_view
                self._finalize_view(obj)

    def _is_consistent(self, obj):
        if not self.shape == obj.shape:
            return False
        # check strides
        stride = 0
        if len(self.strides) < len(self.layout_map):
            return False
        for dim in reversed(np.argsort(self.layout_map)):
            if self.strides[dim] < stride:
                return False
            stride = self.strides[dim]

        # check alignment
        if (
            self.ctypes.data + np.sum([o * s for o, s in zip(self.default_origin, self.strides)])
        ) % gt_backend.from_name(self.backend).storage_info["alignment"]:
            return False

        return True

    def _finalize_view(self, obj):
        pass

    def synchronize(self):
        pass

    def host_to_device(self, force=False):
        pass

    def device_to_host(self, force=False):
        pass

    def __iconcat__(self, other):
        raise NotImplementedError("Concatenation of Storages is not supported")


class GPUStorage(Storage):
    @classmethod
    def _construct(cls, backend, dtype, default_origin, shape, alignment, layout_map):

        (raw_buffer, field) = storage_utils.allocate_gpu(
            default_origin, shape, layout_map, dtype, alignment * dtype.itemsize
        )
        obj = field.view(_ViewableNdarray)
        obj = obj.view(GPUStorage)
        obj._raw_buffer = raw_buffer
        obj.default_origin = default_origin
        return obj

    @property
    def __cuda_array_interface__(self):
        array_interface = self.__array_interface__
        array_interface["version"] = 2
        array_interface["strides"] = self.strides
        array_interface.pop("offset", None)
        return array_interface

    def _check_data(self):
        # check that memory of field is within raw_buffer
        if (
            not self.ctypes.data >= self._raw_buffer.data.ptr
            and self.ctypes.data + self.itemsize * (self.size - 1) <= self._raw_buffer[-1:].data.ptr
        ):
            raise Exception("The buffers are in an inconsistent state.")

    def copy(self):
        res = super().copy()
        res.gpu_view[...] = self.gpu_view
        cp.cuda.Device(0).synchronize()
        return res

    @property
    def gpu_view(self):
        return storage_utils.gpu_view(self)

    def __setitem__(self, key, value):
        if hasattr(value, "__cuda_array_interface__"):
            gpu_view = storage_utils.gpu_view(self)
            gpu_view[key] = cp.asarray(value.data)
            cp.cuda.Device(0).synchronize()
            return value
        else:
            return super().__setitem__(key, value)

    @property
    def _is_clean(self):
        return True

    @property
    def _is_host_modified(self):
        return False

    @property
    def _is_device_modified(self):
        return False

    def _set_clean(self):
        pass

    def _set_host_modified(self):
        pass

    def _set_device_modified(self):
        pass


class CPUStorage(Storage):
    @property
    def _ptr(self):
        return self._ndarray.ctypes.data

    @classmethod
    def _construct(cls, backend, dtype, default_origin, shape, alignment, layout_map):
        (raw_buffer, field) = storage_utils.allocate_cpu(
            default_origin, shape, layout_map, dtype, alignment * dtype.itemsize
        )
        obj = field.view(_ViewableNdarray)
        obj = obj.view(CPUStorage)
        obj._raw_buffer = raw_buffer
        obj.default_origin = default_origin
        return obj

    def _check_data(self):
        # check that memory of field is within raw_buffer and that field is a view of raw_buffer
        if (
            not self.ctypes.data >= self._raw_buffer.ctypes.data
            and self.ctypes.data + self.itemsize * (self.size - 1)
            <= self._raw_buffer[-1:].ctypes.data
            and self.base is not self._raw_buffer
        ):
            raise Exception("The buffers are in an inconsistent state.")

    @property
    def data(self):
        return self.view(np.ndarray)

    def copy(self):
        res = super().copy()
        res[...] = self
        return res


class ExplicitlySyncedGPUStorage(Storage):
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

    @classmethod
    def _construct(cls, backend, dtype, default_origin, shape, alignment, layout_map):

        (
            raw_buffer,
            field,
            device_raw_buffer,
            device_field,
        ) = storage_utils.allocate_gpu_unmanaged(
            default_origin, shape, layout_map, dtype, alignment * dtype.itemsize
        )
        obj = field.view(_ViewableNdarray)
        obj = obj.view(ExplicitlySyncedGPUStorage)
        obj._raw_buffer = raw_buffer
        obj._device_field = device_field
        obj._device_raw_buffer = device_raw_buffer
        obj.default_origin = default_origin

        return obj

    @property
    def data(self):
        return self._device_field

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

    def __getitem__(self, item):
        if self._is_device_modified:
            self.device_to_host()
        return super().__getitem__(item)

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

    def __setitem__(self, key, value):
        if isinstance(value, ExplicitlySyncedGPUStorage):
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
                return self.view(_ViewableNdarray).view(np.ndarray).__setitem__(key, value)
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
            return super().__setitem__(key, value)

    # @property
    # def sync_state(self):
    #     return self._sync_state.state

    @property
    def _ptr(self):
        return self.data.ptr

    def _check_data(self):

        # check that memory of field is within raw_buffer
        if (
            not self.ctypes.data >= self._raw_buffer.ctypes.data
            and self.ctypes.data + self.itemsize * (self.size - 1)
            <= self._raw_buffer[-1:].ctypes.data
            and self.base is not self._raw_buffer
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
        res = super().transpose(*axes)
        res._device_field = cp.lib.stride_tricks.as_strided(
            res._device_raw_buffer, shape=res.shape, strides=res.strides
        )
        return res

    def _finalize_view(self, base):

        if self.shape != base.shape or self.strides != base.strides:
            offset = (base.ctypes.data - self.ctypes.data) + (
                self._device_field.data.ptr - self._device_raw_buffer.data.ptr
            )
            assert not offset % self.dtype.itemsize
            offset = int(offset / self.dtype.itemsize)
            raw_with_offset = self._device_raw_buffer[offset:]
            self._device_field = cp.lib.stride_tricks.as_strided(
                raw_with_offset, shape=self.shape, strides=self.strides
            )

    @property
    def __cuda_array_interface__(self):
        self.host_to_device()
        res = self._device_field.__cuda_array_interface__
        res["strides"] = self.strides
        return res

    def _call_inplace(self, fname, other):
        if isinstance(other, ExplicitlySyncedGPUStorage):
            if (self._is_clean or self._is_device_modified) and (
                other._is_clean or other._is_device_modified
            ):
                self._set_device_modified()
                getattr(self._device_field, fname)(other)
                return self
            elif (self._is_clean or self._is_host_modified) and (
                other._is_clean or other._is_host_modified
            ):
                self._set_host_modified()
                return getattr(super(), fname)(other)
            else:
                if self._is_host_modified:
                    self.host_to_device()
                else:
                    other.host_to_device()
                self._set_device_modified()
                getattr(self._device_field, fname)(other)
                return self
        elif hasattr(other, "__cuda_array_interface__"):
            if self._is_host_modified:
                self.host_to_device()
            self._set_device_modified()
            getattr(self._device_field, fname)(other)
            return self
        else:
            if self._is_device_modified:
                self.device_to_host()
            self._set_host_modified()
            return getattr(super(), fname)(other)

    def __iadd__(self, other):
        return self._call_inplace("__iadd__", other)

    def __iand__(self, other):
        return self._call_inplace("__iand__", other)

    def __ifloordiv__(self, other):
        return self._call_inplace("__ifloordiv__", other)

    def __ilshift__(self, other):
        return self._call_inplace("__ilshift__", other)

    def __imod__(self, other):
        return self._call_inplace("__imod__", other)

    def __imul__(self, other):
        return self._call_inplace("__imul__", other)

    def __ior__(self, other):
        return self._call_inplace("__ior__", other)

    def __ipow__(self, other):
        return self._call_inplace("__ipow__", other)

    def __irshift__(self, other):
        return self._call_inplace("__irshift__", other)

    def __isub__(self, other):
        return self._call_inplace("__isub__", other)

    def __itruediv__(self, other):
        return self._call_inplace("__itruediv__", other)

    def __ixor__(self, other):
        return self._call_inplace("__ixor__", other)


class _ViewableNdarray(np.ndarray):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_stencil_view = True

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(*args, **kwargs)
        obj.is_stencil_view = True
        return obj

    def __array_finalize__(self, obj):
        if hasattr(obj, "is_stencil_view"):
            self.is_stencil_view = obj.is_stencil_view
        else:
            self.is_stencil_view = True
