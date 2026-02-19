# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from collections.abc import Hashable
from typing import Any, Callable, Protocol, TypeGuard, cast

import numpy as np

from gt4py._core import definitions as core_defs


try:
    import cupy
except ImportError:
    cupy = None


class ArrayNamespace(Hashable, Protocol):
    """
    Currently only a subset of the Array API standard namespace with functions relevant for array creation.

    TODO(havogt): replace by ... or make it more complete and put next to NDArrayObject definition.

    See also 'ArrayNamespace'.
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

    bool: type
    int8: type
    int16: type
    int32: type
    int64: type
    uint8: type
    uint16: type
    uint32: type
    uint64: type
    float32: type
    float64: type


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
        and hasattr(obj, "bool")
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
    """
    Get the namespace of the array.

    This is defined in https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__array_namespace__.html,
    however not implemented in CuPy < 14.
    """
    # TODO(havogt): this function can be replaced by https://data-apis.org/array-api-compat/
    if hasattr(array, "__array_namespace__"):
        return array.__array_namespace__()
    else:
        if isinstance(array, np.ndarray):
            return np
        if cupy is not None and isinstance(array, cupy.ndarray):
            return cupy
        raise TypeError(f"Could not determine array namespace of {array} of type {type(array)}")


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
            # TODO test this code path
            raise ValueError(
                f"CuPy only supports GPU devices, got device type {device.device_type}."
            )
        return cupy.cuda.Device(device.device_id)

    _device_translation_registry[cupy] = _cupy_device_translator


@functools.cache
def get_device_translator(array_ns: ArrayNamespace) -> Callable[[core_defs.Device], Any]:
    """
    Returns a mapping from a GT4Py 'Device' to the corresponding device object for the given array namespace.
    """
    if array_ns not in _device_translation_registry:
        raise ValueError(f"No device translator registered for array namespace {array_ns}.")
    return _device_translation_registry[array_ns]
