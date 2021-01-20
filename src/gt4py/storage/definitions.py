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

from numbers import Integral
from typing import Any, Optional, Sequence, Tuple, Type, Union

import numpy as np


try:
    import cupy as cp
except ImportError:
    cp = None

import gt4py
from gt4py import storage as gt_store


class SyncState:
    SYNC_CLEAN = 0
    SYNC_HOST_DIRTY = 1
    SYNC_DEVICE_DIRTY = 2

    def __init__(self):
        self.state = self.SYNC_CLEAN


class Storage:
    """
    Storage class based on a numpy (CPU) or cupy (GPU) array.

    Takes care of proper memory alignment, with additional information that is required by the
    backends.
    """

    __array_subok__ = True

    def __new__(
        cls,
        *,
        aligned_index: Optional[Sequence[int]],
        alignment_size: Optional[int],
        data: Any,
        copy: bool,
        defaults: Optional[str],
        device: Optional[str],
        device_data: Any = None,
        dims: Optional[Sequence[str]],
        dtype: Any,
        halo: Optional[Sequence[Union[int, Tuple[int, int]]]],
        layout: Optional[Sequence[int]],
        managed: Optional[Union[bool, str]],
        shape: Optional[Sequence[int]] = None,
        sync_state: SyncState = None,
        template: Any,
    ):
        params = gt_store.utils.parameter_lookup_and_normalize(
            aligned_index=aligned_index,
            alignment_size=alignment_size,
            copy=copy,
            data=data,
            defaults=defaults,
            device=device,
            device_data=device_data,
            dims=dims,
            dtype=dtype,
            halo=halo,
            layout=layout,
            managed=managed,
            shape=shape,
            sync_state=sync_state,
            template=template,
        )

        # 7) determine storage type
        storage_t: Type[Storage]
        if params["device"] == "gpu":
            if params["managed"] == "cuda":
                storage_t = CudaManagedGPUStorage
            elif params["managed"] == "gt4py":
                storage_t = ExplicitlyManagedGPUStorage
            else:
                storage_t = GPUStorage
        else:
            storage_t = CPUStorage

        self = storage_t._new(**params)
        self._shape = params["shape"]
        self._dtype = params["dtype"]
        self._halo = params["halo"]
        return self

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def mask(self):
        return self._mask

    @property
    def dtype(self):
        return self._dtype

    @property
    def itemsize(self):
        return self._dtype.itemsize

    @property
    def size(self):
        return int(np.prod(self.shape))

    def transpose(self, *axes):
        res: Storage = gt_store.as_storage(self)
        res._transpose(*axes)
        if len(axes) == 1 or gt4py.utils.is_iterable_of(axes[0], item_class=Integral):
            axes = axes[0]
        res._shape = tuple(self._shape[i] for i in axes)
        res._halo = tuple(self._halo[i] for i in axes)

        return res

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
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
            if hasattr(res, "_field"):
                res._shape = res._field.shape
            else:
                res._shape = res._device_field.shape
            return res
        else:
            return self._forward_getitem_scalar(key)

    @classmethod
    def _new(cls, *args, **kwargs):
        raise NotImplementedError("_new of base class not implemented")

    def copy(self):
        return gt4py.storage.storage(data=self)

    def synchronize(self):
        pass

    def host_to_device(self, force=False):
        pass

    def device_to_host(self, force=False):
        pass

    def _set_device_modified(self):
        pass

    def _set_host_modified(self):
        pass

    def _set_clean(self):
        pass

    def __iconcat__(self, other):
        raise NotImplementedError("Concatenation of Storages is not supported")

    @property
    def halo(self):
        return self._halo

    @halo.setter
    def halo(self, value):
        if value is None:
            raise TypeError("Halo can not be None.")
        self._halo = gt_store.utils.normalize_halo(value)

    def to_numpy(self):
        if self.device == "cpu":
            return np.array(self, copy=True)
        else:
            return cp.asnumpy(self)

    def to_cupy(self):
        if cp is None:
            raise BufferError("Not possible to convert to CuPy, since CuPy could not be imported.")
        return cp.array(self, copy=True)

    @property
    def __gt_data_interface__(self):
        res = dict()

        if hasattr(self, "_field"):
            array_interface = self._field.__array_interface__
            array_interface["acquire"] = self.device_to_host
            array_interface["touch"] = self._set_host_modified
            res["cpu"] = array_interface

        if hasattr(self, "_device_field"):
            cuda_array_interface = self._device_field.__cuda_array_interface__
            cuda_array_interface["acquire"] = self.host_to_device
            cuda_array_interface["touch"] = self._set_device_modified
            res["gpu"] = cuda_array_interface
        elif hasattr(self, "__cuda_array_interface__"):
            cuda_array_interface = self.__cuda_array_interface__
            cuda_array_interface["acquire"] = self.host_to_device
            cuda_array_interface["touch"] = self._set_device_modified
            res["gpu"] = cuda_array_interface

        for v in res.values():
            v["halo"] = self._halo
        return res


class CudaManagedGPUStorage(Storage):
    device = "gpu"

    @classmethod
    def _new(
        cls,
        *,
        shape,
        dtype,
        data=None,
        device_data=None,
        aligned_index=None,
        alignment_size=None,
        layout=None,
        copy=False,
        halo,
        **kwargs,
    ):

        if data is None or copy:
            device_memory, field = gt_store.utils.allocate_gpu_cuda_managed(
                aligned_index,
                shape,
                layout,
                dtype,
                alignment_size * np.dtype(dtype).itemsize,
            )
        else:
            assert isinstance(data, np.ndarray)
            assert isinstance(device_data, cp.ndarray)
            field = data
            device_memory = device_data

        self = super(Storage, cls).__new__(
            CudaManagedGPUStorage,
        )
        self._field = field
        self._device_memory = device_memory

        if copy:
            if device_data is not None:
                gpu_view = cp.asarray(self)
                gpu_view[...] = device_data
                cp.cuda.Device(0).synchronize()
            else:
                self._field[...] = data

        return self

    @property
    def __array_interface__(self):
        array_interface = self._field.__array_interface__
        return array_interface

    @property
    def __cuda_array_interface__(self):
        array_interface = self._field.__array_interface__
        array_interface["version"] = 2
        array_interface["strides"] = self.strides
        array_interface.pop("offset", None)
        return array_interface

    @property
    def strides(self):
        return self._field.strides

    def _forward_setitem(self, key, value):
        if hasattr(value, "__cuda_array_interface__"):
            gpu_view = cp.asarray(self)
            res = gpu_view.__setitem__(key, value)
            cp.cuda.Device(0).synchronize()
            return res
        else:
            return self._field.__setitem__(key, value)

    def _forward_getitem_slice(self, key):
        self._field = self._field.__getitem__(key)

    def _transpose(self, *axes):
        self._field = self._field.transpose(*axes)

    def as_numpy(self):
        raise np.asarray(self)

    def as_cupy(self):
        raise cp.asarray(self)


class CPUStorage(Storage):
    device = "cpu"

    @property
    def _ptr(self):
        return self._ndarray.ctypes.data

    @classmethod
    def _new(
        cls,
        *,
        shape,
        dtype,
        data=None,
        aligned_index=None,
        alignment_size=None,
        layout=None,
        copy=False,
        halo,
        **kwargs,
    ):
        self = super(Storage, cls).__new__(
            CPUStorage,
        )

        self._field = data
        if data is None or copy:
            _, self._field = gt_store.utils.allocate_cpu(
                aligned_index,
                shape,
                layout,
                dtype,
                alignment_size * np.dtype(dtype).itemsize,
            )
        else:
            self._field = np.asarray(data)

        if copy:
            self._field[...] = data
        return self

    @property
    def data(self):
        return self._field.data

    @property
    def strides(self):
        return self._field.strides

    def _forward_setitem(self, key, value):
        self._field.__setitem__(key, value)

    def _forward_getitem_slice(self, key):
        self._field = self._field.__getitem__(key)

    def _forward_getitem_scalar(self, key):
        return self._field.__getitem__(key)

    @property
    def __array_interface__(self):
        return self._field.__array_interface__

    def _transpose(self, *axes):
        self._field = self._field.transpose(*axes)

    def as_numpy(self):
        raise np.asarray(self)


class GPUStorage(Storage):
    device = "gpu"

    @property
    def _ptr(self):
        return self._device_field.data.ptr

    @classmethod
    def _new(
        cls,
        *,
        shape,
        dtype,
        data=None,
        device_data=None,
        aligned_index=None,
        alignment_size=None,
        layout=None,
        copy=False,
        halo,
        **kwargs,
    ):
        self = super(Storage, cls).__new__(
            GPUStorage,
        )

        if device_data is None or copy:
            _, self._device_field = gt_store.utils.allocate_gpu_only(
                aligned_index,
                shape,
                layout,
                dtype,
                alignment_size * np.dtype(dtype).itemsize,
            )
        else:
            self._device_field = device_data

        if copy:
            if device_data is not None:
                self._device_field[...] = device_data
            elif isinstance(data, np.ndarray) and gt4py.storage.utils._is_contiguous(data):
                self._device_field.set(data)
            else:
                self._device_field[...] = cp.asarray(data)

        return self

    @property
    def device_data(self):
        return self._device_field.data

    @property
    def strides(self):
        return self._device_field.strides

    def _forward_setitem(self, key, value):
        self._device_field.__setitem__(key, value)

    def _forward_getitem_slice(self, key):
        self._device_field = self._device_field.__getitem__(key)

    def _forward_getitem_scalar(self, key):
        return self._device_field.__getitem__(key)

    @property
    def __cuda_array_interface__(self):
        return self._device_field.__cuda_array_interface__

    def _transpose(self, *axes):
        self._device_field = self._device_field.transpose(*axes)

    def as_cupy(self):
        return cp.asarray(self)


class ExplicitlyManagedGPUStorage(Storage):
    device = "gpu"

    @classmethod
    def _new(
        cls,
        *,
        shape,
        dtype,
        data=None,
        aligned_index=None,
        alignment_size=None,
        layout=None,
        copy=False,
        device_data=None,
        halo,
        sync_state,
        **kwargs,
    ):
        if data is None or copy:
            _, field = gt_store.utils.allocate_cpu(
                aligned_index,
                shape,
                layout,
                dtype,
                alignment_size * np.dtype(dtype).itemsize,
            )
        else:
            assert isinstance(data, np.ndarray)
            field = data
        if device_data is None or copy:
            _, device_field = gt_store.utils.allocate_gpu_only(
                aligned_index,
                shape,
                layout,
                dtype,
                alignment_size * np.dtype(dtype).itemsize,
            )
        else:
            assert isinstance(device_data, cp.ndarray)
            device_field = device_data
        self = super(Storage, cls).__new__(
            ExplicitlyManagedGPUStorage,
        )
        self._field = field
        self._device_field = device_field

        self._sync_state = sync_state

        if copy:
            if sync_state is None:
                if device_data is not None:
                    self._device_field[...] = device_data
                elif isinstance(data, np.ndarray) and gt4py.storage.utils._is_contiguous(data):
                    self._device_field.set(data)
                else:
                    self._device_field[...] = cp.asarray(data)
                self._set_device_modified()
            else:
                if device_data is not None and sync_state.state != SyncState.SYNC_HOST_DIRTY:
                    self._device_field[...] = device_data
                    self._set_device_modified()
                else:
                    self._field[...] = data
                    self._set_host_modified()
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
            if gt4py.storage.utils._is_contiguous(self._device_field):
                self._device_field.set(self._field)
            else:
                self._device_field[...] = cp.asarray(self._field)

    def device_to_host(self, force=False):
        if force or self._is_device_modified:
            self._set_clean()
            if gt4py.storage.utils._is_contiguous(self._field):
                self._device_field.get(out=self._field)
            else:
                self._field[...] = cp.asnumpy(self._device_field)

    @property
    def _is_clean(self):
        return self._sync_state.state == SyncState.SYNC_CLEAN

    @property
    def _is_host_modified(self):
        return self._sync_state.state == SyncState.SYNC_HOST_DIRTY

    @property
    def _is_device_modified(self):
        return self._sync_state.state == SyncState.SYNC_DEVICE_DIRTY

    def _set_clean(self):
        self._sync_state.state = SyncState.SYNC_CLEAN

    def _set_host_modified(self):
        self._sync_state.state = SyncState.SYNC_HOST_DIRTY

    def _set_device_modified(self):
        self._sync_state.state = SyncState.SYNC_DEVICE_DIRTY

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
        return self._sync_state

    @property
    def _ptr(self):
        return self.data.ptr

    def _transpose(self, *axes):
        self._field = self._field.transpose(*axes)
        self._device_field = self._device_field.transpose(*axes)

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

    def as_numpy(self):
        raise np.asarray(self)

    def as_cupy(self):
        raise cp.asarray(self)
