# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Hashable
from typing import Any, Callable, Protocol, TypeGuard, cast

import array_api_compat
import numpy as np

from gt4py._core import definitions as core_defs


try:
    import cupy
except ImportError:
    cupy = None


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


def _numpy_device_translator(device: core_defs.Device | None) -> Any:
    if device is None:
        return None
    if device.device_type == core_defs.DeviceType.CPU:
        return None  # or Literal['cpu']
    raise ValueError(f"NumPy does not support device type {device.device_type}.")


# Currently private as we only support a concrete set of array namespaces.
_device_translation_registry: dict[ArrayNamespace, Callable[[core_defs.Device], Any]] = {
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


def get_device_translator(array_ns: ArrayNamespace) -> Callable[[core_defs.Device], Any]:
    """
    Returns a mapping from a GT4Py 'Device' to the corresponding device object for the given array namespace.
    """
    try:
        return _device_translation_registry[array_ns]
    except KeyError:
        raise ValueError(
            f"No device translator registered for array namespace {array_ns}."
        ) from None
