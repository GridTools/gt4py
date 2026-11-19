# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
from collections.abc import Hashable
from typing import Any, Callable, Final, Protocol, Self, TypeGuard, cast

import array_api_compat
import numpy as np

from gt4py._core import definitions as core_defs, dlpack as core_dlpack, types as core_types


try:
    import cupy
except ImportError:
    cupy = None

try:
    import torch
except ImportError:
    torch = None


class _ArrayDType(Protocol):
    # The only requirement for array API dtypes is equality comparability.
    def __eq__(self, other: Any) -> bool: ...


class ArrayNamespace(Hashable, Protocol):
    """
    Currently only a subset of the Array API standard namespace with functions relevant for array creation.

    TODO(havogt): replace by ... or make it more complete and put next to NDArrayObject definition.
    """

    def empty(
        self, shape: tuple[int, ...], *, dtype: Any | None = None, device: Any | None = None
    ) -> core_defs.NDArrayObject: ...

    def zeros(
        self, shape: tuple[int, ...], *, dtype: Any | None = None, device: Any | None = None
    ) -> core_defs.NDArrayObject: ...

    def ones(
        self, shape: tuple[int, ...], *, dtype: Any | None = None, device: Any | None = None
    ) -> core_defs.NDArrayObject: ...

    def full(
        self,
        shape: tuple[int, ...],
        fill_value: Any,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
    ) -> core_defs.NDArrayObject: ...

    def asarray(
        self,
        obj: Any,
        /,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
        copy: bool | None = None,
    ) -> core_defs.NDArrayObject: ...

    bool: _ArrayDType
    int8: _ArrayDType
    int16: _ArrayDType
    int32: _ArrayDType
    int64: _ArrayDType
    uint8: _ArrayDType
    uint16: _ArrayDType
    uint32: _ArrayDType
    uint64: _ArrayDType
    float32: _ArrayDType
    float64: _ArrayDType


def is_array_namespace(obj: Any) -> TypeGuard[ArrayNamespace]:
    """
    Check whether `obj` (structurally) is a namespace of the array API.

    See description in 'ArrayNamespace'.
    """

    return (
        hasattr(obj, "empty")
        and hasattr(obj, "zeros")
        and hasattr(obj, "ones")
        and hasattr(obj, "full")
        and hasattr(obj, "asarray")
        and (hasattr(obj, "bool") or hasattr(obj, "bool_"))  # for NumPy 1.x compatibility
        and hasattr(obj, "int8")
        and hasattr(obj, "int16")
        and hasattr(obj, "int32")
        and hasattr(obj, "int64")
        and hasattr(obj, "uint8")
        and hasattr(obj, "uint16")
        and hasattr(obj, "uint32")
        and hasattr(obj, "uint64")
        and hasattr(obj, "float32")
        and hasattr(obj, "float64")
    )


def array_namespace(array: core_defs.NDArrayObject) -> ArrayNamespace:
    return cast(ArrayNamespace, array_api_compat.array_namespace(array))


_SCALAR_TYPES: Final = frozenset(core_defs.SCALAR_TYPES)


@functools.cache
def _canonicalize_numpy_dtype(dtype: np.dtype) -> np.dtype:
    # dtypes named by C type map to distinct scalar types, e.g. `np.longlong` for int64 on Linux
    return np.dtype(f"{dtype.kind}{dtype.itemsize}")


@functools.cache
def _dtype_tables(
    array_ns: ArrayNamespace,
) -> tuple[dict[type, _ArrayDType], dict[_ArrayDType, type]]:
    to_namespace = {
        scalar_type: getattr(array_ns, name)
        for scalar_type, name in core_types.type_to_name.items()
        if hasattr(array_ns, name)
    }
    return to_namespace, {dtype: scalar_type for scalar_type, dtype in to_namespace.items()}


def namespace_dtype(array_ns: ArrayNamespace, scalar_type: type[core_defs.Scalar]) -> _ArrayDType:
    """The dtype object of `array_ns` for the gt4py scalar type `scalar_type`."""
    key: type = (
        _canonicalize_numpy_dtype(np.dtype(scalar_type)).type
        if issubclass(scalar_type, np.generic)
        else scalar_type
    )
    try:
        return _dtype_tables(array_ns)[0][key]
    except KeyError:
        raise TypeError(f"Array namespace '{array_ns}' has no dtype for '{scalar_type}'.") from None


def scalar_type(array_ns: ArrayNamespace, dtype: _ArrayDType) -> type[core_defs.Scalar]:
    """The gt4py scalar type of the arrays of `array_ns` with dtype `dtype`."""
    result = (
        _canonicalize_numpy_dtype(dtype).type
        if isinstance(dtype, np.dtype)
        else _dtype_tables(array_ns)[1].get(dtype)
    )
    if result not in _SCALAR_TYPES:
        raise TypeError(
            f"Dtype '{dtype}' of array namespace '{array_ns}' is not a gt4py scalar type."
        )
    return result


def dtype_of(array: core_defs.NDArrayObject) -> core_defs.DType:
    """The gt4py dtype of `array`."""
    return core_defs.dtype(scalar_type(array_namespace(array), array.dtype))


def _numpy_device_translator(device: core_defs.Device | None) -> Any:
    if device is None:
        return None
    if device.device_type == core_defs.DeviceType.CPU:
        return None  # or Literal['cpu']
    raise ValueError(f"NumPy does not support device type {device.device_type}.")


# Currently private as we only support a concrete set of array namespaces.
_device_translation_registry: dict[ArrayNamespace, Callable[[core_defs.Device | None], Any]] = {
    cast(ArrayNamespace, np): _numpy_device_translator
}
"""
Registry for mappings from array namespaces to device_translators.

Device translators are functions mapping GT4Py 'Device' objects to the corresponding device objects for the given array namespace.
It's responsibility of the translator to check if 'Device' is valid for the given array namespace and raise a 'ValueError' if not.
"""

if cupy is not None:

    def _cupy_device_translator(device: core_defs.Device | None) -> Any:
        if device is None:
            return None
        if device.device_type != core_defs.CUPY_DEVICE_TYPE:
            raise ValueError(
                f"CuPy only supports GPU devices, got device type {device.device_type}."
            )
        return cupy.cuda.Device(device.device_id)

    _device_translation_registry[cupy] = _cupy_device_translator


if torch is not None:

    def _torch_device_translator(device: core_defs.Device | None) -> Any:
        if device is None:
            return None
        if device.device_type == core_defs.DeviceType.CPU:
            return torch.device("cpu")
        # ROCm builds of torch use the 'cuda' device type
        if device.device_type in (core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM):
            return torch.device("cuda", device.device_id)
        raise ValueError(f"torch does not support device type {device.device_type}.")

    _device_translation_registry[torch] = _torch_device_translator


def get_device_translator(array_ns: ArrayNamespace) -> Callable[[core_defs.Device | None], Any]:
    """
    Returns a mapping from a GT4Py 'Device' to the corresponding device object for the given array namespace.
    """
    try:
        return _device_translation_registry[array_ns]
    except KeyError:
        raise ValueError(
            f"No device translator registered for array namespace {array_ns}."
        ) from None


_DLPACK_TYPE_CODES: Final = {
    core_defs.DTypeKind.INT: core_dlpack.DataTypeCode.INT,
    core_defs.DTypeKind.UINT: core_dlpack.DataTypeCode.UINT,
    core_defs.DTypeKind.FLOAT: core_dlpack.DataTypeCode.FLOAT,
    core_defs.DTypeKind.BOOL: core_dlpack.DataTypeCode.BOOL,
}
_DTYPES_BY_DLPACK_DTYPE: Final[dict[tuple[int, int, int], core_defs.DType]] = {
    (_DLPACK_TYPE_CODES[dt.kind], dt.bit_width, dt.lanes): dt
    for dt in (core_defs.dtype(t) for t in core_defs.SCALAR_TYPES)
}
_DEVICE_TYPES: Final = {t.value: t for t in core_defs.DeviceType}


def device_from_dlpack(device_type: int, device_id: int) -> core_defs.Device:
    """The `Device` of a DLPack device pair; device types gt4py does not name are kept as integers."""
    return core_defs.Device(_DEVICE_TYPES.get(int(device_type), int(device_type)), int(device_id))


def _c_contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
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
    def from_dlpack(cls, array: Any) -> Self:
        """Describe the buffer of `array` from its DLPack export.

        Raises:
            TypeError: `array` does not export DLPack, or its dtype is not a gt4py scalar type.
            BufferError: `array` cannot be exported without a copy, or its DLPack export is
                not supported (see `gt4py._core.dlpack.tensor_fields`), or it reports a stride
                of `2**63` bytes or more.
        """
        tensor = core_dlpack.tensor_fields(array)
        code, bits, lanes = tensor["dtype"]
        try:
            dtype = _DTYPES_BY_DLPACK_DTYPE[code, bits, lanes]
        except KeyError:
            raise TypeError(
                f"DLPack dtype (code={code}, bits={bits}, lanes={lanes})"
                f" of '{type(array).__name__}' is not a gt4py scalar type."
            ) from None
        shape = tensor["shape"]
        strides = tensor["strides"]
        if strides is None:
            strides = _c_contiguous_strides(shape)
        itemsize = (bits * lanes + 7) // 8
        if any(abs(stride) * itemsize >= 2**63 for stride in strides):
            raise BufferError(
                f"'{type(array).__name__}' exports DLPack element strides {strides} that no buffer"
                f" of {itemsize}-byte elements can have: the exporter reports an impossible stride"
                " (e.g. CuPy before cupy#10228 exports negative strides divided as unsigned"
                " integers)."
            )
        return cls(
            data_ptr=tensor["data"] + tensor["byte_offset"],
            shape=shape,
            elem_strides=strides,
            dtype=dtype,
            device=device_from_dlpack(*tensor["device"]),
        )
