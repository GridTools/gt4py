# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`BufferInfo`: buffer metadata of any array that exports DLPack, read from its capsule with `ctypes`.

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
- `BufferError`: versioned capsules of major version 2 or later, exports that need a copy, and
  strides of `2**63` bytes or more, which no buffer can have.
- `TypeError`: arrays without `__dlpack__`/`__dlpack_device__`, and DLPack dtypes that are not
  gt4py scalar types (bool, 8 to 64 bit signed and unsigned integers, float32, float64; one lane).

NULL strides, allowed before DLPack 1.2, are read as C-contiguous.
"""

from __future__ import annotations

import ctypes
import dataclasses
import functools
from typing import Any, Final

from gt4py._core import definitions as core_defs


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

_DLPACK_TYPE_CODES: Final = {
    core_defs.DTypeKind.INT: 0,
    core_defs.DTypeKind.UINT: 1,
    core_defs.DTypeKind.FLOAT: 2,
    core_defs.DTypeKind.BOOL: 6,
}
_DTYPES: Final[dict[tuple[int, int, int], core_defs.DType]] = {
    (_DLPACK_TYPE_CODES[dt.kind], dt.bit_width, dt.lanes): dt
    for dt in (core_defs.dtype(t) for t in core_defs.SCALAR_TYPES)
}
_DEVICE_TYPES: Final = {t.value: t for t in core_defs.DeviceType}
_NO_SYNC_STREAM_DEVICE_TYPES: Final = (core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM)


def device(device_type: int, device_id: int) -> core_defs.Device:
    """The `Device` of a DLPack device pair; device types gt4py does not name are kept as integers."""
    return core_defs.Device(_DEVICE_TYPES.get(int(device_type), int(device_type)), int(device_id))


def c_contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Element strides of a C-contiguous array of `shape`."""
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    return tuple(reversed(strides))


@dataclasses.dataclass(frozen=True)
class BufferInfo:
    """Holds information about a buffer in memory.

    Equal `BufferInfo`s (and equal `hash_key`s) describe buffers that behave the same apart
    from their contents.

    Args:
        data_ptr: Address of the element at index `(0, ..., 0)`.
        shape: Number of elements along each dimension.
        elem_strides: Distance between consecutive elements along each dimension, in elements.
        dtype: Data type of the elements.
        device: Device the buffer is allocated on.
    """

    data_ptr: int
    shape: tuple[int, ...]
    elem_strides: tuple[int, ...]
    dtype: core_defs.DType
    device: core_defs.Device

    @functools.cached_property
    def hash_key(self) -> int:
        return hash((self.data_ptr, self.shape, self.elem_strides, self.dtype, self.device))

    def __hash__(self) -> int:
        return self.hash_key

    @classmethod
    def from_dlpack(cls, array: Any) -> BufferInfo:
        """Describe the buffer of `array` from its DLPack capsule, without consuming the capsule.

        On CUDA and ROCm the capsule is requested without stream synchronization (`stream=-1`), the
        only devices for which the Array API standard defines that request; all other devices get
        `stream=None`, which the standard requires on CPU and on devices without streams. The result
        describes where the data lives, not when it is ready.

        Raises:
            TypeError: `array` does not export DLPack, or its dtype is not a gt4py scalar type.
            BufferError: `array` cannot be exported without a copy, the capsule has an
                unsupported DLPack major version, or it reports a stride of `2**63` bytes or more.
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
        strides = tuple(tensor.strides[:ndim]) if tensor.strides else c_contiguous_strides(shape)
        itemsize = dtype.byte_size
        if any(abs(stride) * itemsize >= 2**63 for stride in strides):
            raise BufferError(
                f"'{type(array).__name__}' exports DLPack element strides {strides} that no buffer"
                f" of {itemsize}-byte elements can have: the exporter reports an impossible stride"
                " (e.g. CuPy before cupy#10228 exports negative strides divided as unsigned integers)."
            )
        return cls(
            data_ptr=(tensor.data or 0) + tensor.byte_offset,
            shape=shape,
            elem_strides=strides,
            dtype=dtype,
            device=device(tensor.device.device_type, tensor.device.device_id),
        )
