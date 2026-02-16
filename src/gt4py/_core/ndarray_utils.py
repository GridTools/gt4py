# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Protocol, TypeGuard

import numpy as np

from gt4py._core import definitions as core_defs


try:
    import cupy
except ImportError:
    cupy = None


class ArrayCreationNamespace(Protocol):
    """
    Subset of the Array API standard namespace with functions relevant for array creation.

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

    @property
    def bool(self) -> Any: ...

    @property
    def int8(self) -> Any: ...

    @property
    def int16(self) -> Any: ...

    @property
    def int32(self) -> Any: ...

    @property
    def int64(self) -> Any: ...

    @property
    def uint8(self) -> Any: ...

    @property
    def uint16(self) -> Any: ...

    @property
    def uint32(self) -> Any: ...

    @property
    def uint64(self) -> Any: ...

    @property
    def float32(self) -> Any: ...

    @property
    def float64(self) -> Any: ...


def is_array_api_creation_namespace(obj: Any) -> TypeGuard[ArrayCreationNamespace]:
    """Check whether `obj` (structurally) supports the subset of the array API relevant for array creation."""

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


class ArrayNamespace(ArrayCreationNamespace, Protocol):
    """
    An array namespace is any module that follows the Array API standard
    (https://data-apis.org/array-api/latest/).
    """

    # TODO(havogt): replace by https://github.com/data-apis/array-api-typing once fully supported
    ...


def array_namespace(array: core_defs.NDArrayObject) -> ArrayNamespace:
    """
    Get the namespace of the array.

    This is defined in https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__array_namespace__.html,
    however not implemented in CuPy < 14.
    """
    if hasattr(array, "__array_namespace__"):
        return array.__array_namespace__()
    else:
        if isinstance(array, np.ndarray):
            return np
        if cupy is not None and isinstance(array, cupy.ndarray):
            return cupy
        raise TypeError(f"Could not determine array namespace of {array} of type {type(array)}")
