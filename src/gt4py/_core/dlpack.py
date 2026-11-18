# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The `DLTensor` of any array that exports DLPack, read from its capsule with `ctypes`.

The binary layout follows `dlpack.h` of DLPack v1.4
(https://github.com/dmlc/dlpack/blob/v1.4/include/dlpack/dlpack.h, documented at
https://dmlc.github.io/dlpack/latest/c_api.html); the exchange protocol is the Python
specification of DLPack (https://dmlc.github.io/dlpack/latest/python_spec.html) and the Array
API `__dlpack__` method
(https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html).

Supported capsules:
- `"dltensor_versioned"` (`DLManagedTensorVersioned`, DLPack >= 1.0) of ABI major version 1,
  any minor version. It is requested with `max_version=(1, 0)`.
- `"dltensor"` (legacy `DLManagedTensor`, DLPack < 1.0), which producers without `max_version`
  support, and JAX, return.

Rejected:
- `BufferError`: versioned capsules of major version 2 or later, and exports that need a copy.
- `TypeError`: arrays without `__dlpack__`/`__dlpack_device__`.
"""

from __future__ import annotations

import ctypes
import enum
from typing import Any, Final


class DeviceType(enum.IntEnum):
    """The `DLDeviceType` values this module needs to name."""

    CPU = 1
    CUDA = 2
    ROCM = 10


class DataTypeCode(enum.IntEnum):
    """`DLDataTypeCode` values up to `kDLBool`."""

    INT = 0
    UINT = 1
    FLOAT = 2
    OPAQUE_HANDLE = 3
    BFLOAT = 4
    COMPLEX = 5
    BOOL = 6


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
_DL_TENSOR_OFFSET: Final[int] = _DLManagedTensorVersioned.dl_tensor.offset

# Own prototypes instead of setting `argtypes` on the `ctypes.pythonapi` functions shared
# with every other library in the process.
_capsule_is_valid = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_IsValid", ctypes.pythonapi)
)
_capsule_pointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi)
)

_NO_SYNC_STREAM_DEVICE_TYPES: Final = (DeviceType.CUDA, DeviceType.ROCM)


def tensor_fields(array: Any) -> dict[str, Any]:
    """The `DLTensor` fields of the DLPack export of `array`, read without consuming the capsule.

    The keys are the `DLTensor` field names, with the values as exported: `data` (address,
    0 for NULL), `device` as `(device_type, device_id)`, `ndim`, `dtype` as `(code, bits,
    lanes)`, `shape`, `strides` in elements (`None` for NULL, which DLPack before 1.2 allows
    for C-contiguous tensors) and `byte_offset`.

    On CUDA and ROCm the capsule is requested without stream synchronization (`stream=-1`), the
    only devices for which the Array API standard defines that request; all other devices get
    `stream=None`, which the standard requires on CPU and on devices without streams. The result
    describes where the data lives, not when it is ready.

    Raises:
        TypeError: `array` does not export DLPack.
        BufferError: `array` cannot be exported without a copy, or the capsule has an
            unsupported DLPack major version.
    """
    try:
        device_type, _ = array.__dlpack_device__()
    except AttributeError:
        raise TypeError(f"'{type(array).__name__}' does not export DLPack.") from None
    stream = -1 if device_type in _NO_SYNC_STREAM_DEVICE_TYPES else None
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

    ndim = tensor.ndim
    return {
        "data": tensor.data or 0,
        "device": (tensor.device.device_type, tensor.device.device_id),
        "ndim": ndim,
        "dtype": (tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes),
        "shape": tuple(tensor.shape[:ndim]),
        "strides": tuple(tensor.strides[:ndim]) if tensor.strides else None,
        "byte_offset": tensor.byte_offset,
    }
