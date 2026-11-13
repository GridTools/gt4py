# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`common.BufferInfo` of any array that exports DLPack, read from the capsule with `ctypes`."""

from __future__ import annotations

import ctypes
from typing import Any, Final

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common


class _DLDevice(ctypes.Structure):
    _fields_ = (("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32))


class _DLDataType(ctypes.Structure):
    _fields_ = (("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16))


class _DLTensor(ctypes.Structure):
    _fields_ = (
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    )


class _DLPackVersion(ctypes.Structure):
    _fields_ = (("major", ctypes.c_uint32), ("minor", ctypes.c_uint32))


class _DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = (
        ("version", _DLPackVersion),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", _DLTensor),
    )


_VERSIONED: Final = b"dltensor_versioned"
_UNVERSIONED: Final = b"dltensor"
_DL_TENSOR_OFFSET: Final[int] = _DLManagedTensorVersioned.dl_tensor.offset  # type: ignore[attr-defined] # ctypes field descriptors are untyped

_capsule_is_valid = ctypes.pythonapi.PyCapsule_IsValid
_capsule_is_valid.restype = ctypes.c_int
_capsule_is_valid.argtypes = (ctypes.py_object, ctypes.c_char_p)
_capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_pointer.restype = ctypes.c_void_p
_capsule_pointer.argtypes = (ctypes.py_object, ctypes.c_char_p)

_DLPACK_TYPE_CODES: Final = {"i": 0, "u": 1, "f": 2, "b": 6}
_DTYPES: Final[dict[tuple[int, int, int], core_defs.DType]] = {
    (_DLPACK_TYPE_CODES[np.dtype(t).kind], 8 * np.dtype(t).itemsize, 1): core_defs.dtype(t)
    for t in core_defs.SCALAR_TYPES
    if issubclass(t, np.generic)
}
_DEVICE_TYPES: Final = {t.value: t for t in core_defs.DeviceType}


def device(device_type: int, device_id: int) -> core_defs.Device:
    """The `Device` of a DLPack device pair; device types gt4py does not name are kept as integers."""
    return core_defs.Device(_DEVICE_TYPES.get(int(device_type), int(device_type)), int(device_id))  # type: ignore[arg-type] # unnamed DLPack device types are kept as plain integers


def c_contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Element strides of a C-contiguous array of `shape`."""
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    return tuple(reversed(strides))


def buffer_info(array: Any) -> common.BufferInfo:
    """Describe the buffer of `array` from its DLPack capsule, without consuming the capsule.

    The capsule is requested without stream synchronization: the result describes where the
    data lives, not when it is ready.

    Raises:
        TypeError: `array` does not export DLPack, or its dtype is not a gt4py scalar type.
        BufferError: `array` cannot be exported without a copy, or the capsule has an
            unsupported DLPack major version.
    """
    try:
        device_type, _ = array.__dlpack_device__()
    except AttributeError:
        raise TypeError(f"'{type(array).__name__}' does not export DLPack.") from None
    stream = None if device_type == core_defs.DeviceType.CPU else -1
    try:
        capsule = array.__dlpack__(stream=stream, max_version=(1, 0), copy=False)
    except TypeError:
        capsule = array.__dlpack__(stream=stream)

    # Not renaming the capsule leaves it unconsumed, so its destructor calls the producer's deleter.
    if _capsule_is_valid(capsule, _VERSIONED):
        address = _capsule_pointer(capsule, _VERSIONED)
        version = _DLPackVersion.from_address(address)
        if version.major > 1:
            raise BufferError(f"Unsupported DLPack version {version.major}.{version.minor}.")
        tensor = _DLTensor.from_address(address + _DL_TENSOR_OFFSET)
    else:
        tensor = _DLTensor.from_address(_capsule_pointer(capsule, _UNVERSIONED))

    dl_dtype = tensor.dtype
    try:
        dtype = _DTYPES[dl_dtype.code, dl_dtype.bits, dl_dtype.lanes]
    except KeyError:
        raise TypeError(
            f"DLPack dtype (code={dl_dtype.code}, bits={dl_dtype.bits}, lanes={dl_dtype.lanes})"
            f" of '{type(array).__name__}' is not a gt4py scalar type."
        ) from None
    ndim = tensor.ndim
    shape = tuple(tensor.shape[:ndim])
    return common.BufferInfo(
        data_ptr=(tensor.data or 0) + tensor.byte_offset,
        shape=shape,
        elem_strides=tuple(tensor.strides[:ndim]) if tensor.strides else c_contiguous_strides(shape),
        dtype=dtype,
        device=device(tensor.device.device_type, tensor.device.device_id),
    )
