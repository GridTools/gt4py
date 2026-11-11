# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Array operations of the embedded fields, per array namespace.

`ArrayOps` implements everything `NdArrayField` does on its buffer against the
Array API standard, through the namespace `array_api_compat` returns for the
buffer. Any standard-conforming array library works without registration.

A library that needs different behaviour, or has a faster path outside the
standard, subclasses `ArrayOps`, overrides only those methods and is registered
with `register`. The registry is keyed by a predicate on the compat namespace and
holds a factory, so registering a library does not import it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any, ClassVar, Final

import array_api_compat
import array_api_compat.numpy as compat_numpy
import numpy as np

from gt4py._core import definitions as core_defs, ndarray_utils as core_ndarray_utils
from gt4py.next import common


# TODO(havogt): drop once gt4py requires NumPy >= 2.1, the first version whose `from_dlpack`
#   takes `device` and `copy`; NumPy 2.0's `from_dlpack` cannot copy a non-CPU array to the host.
_NUMPY_FROM_DLPACK_TAKES_DEVICE: Final = np.lib.NumpyVersion(np.__version__) >= "2.1.0"

#: gt4py builtin and operator names whose Array API name differs.
STANDARD_NAMES: Final[Mapping[str, str]] = {
    "neg": "negative",
    "pos": "positive",
    "sub": "subtract",
    "mul": "multiply",
    "div": "divide",
    "floordiv": "floor_divide",
    "power": "pow",
    "pow": "pow",
    "mod": "remainder",
    "invert": "logical_not",
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "arcsinh": "asinh",
    "arccosh": "acosh",
    "arctanh": "atanh",
}


class ArrayOps:
    """
    Operations on the buffers of embedded fields for one array namespace.

    `xp` is the Array API namespace of the buffers. Field code uses `xp` directly
    for standard functions and the methods of this class wherever a library may
    need to differ; those methods are the override points.
    """

    #: Replacements of whole `fbuiltins` implementations for fields of this namespace.
    builtin_overrides: ClassVar[Mapping[Any, Callable[..., Any]]] = {}

    xp: ModuleType

    def __init__(self, xp: ModuleType) -> None:
        self.xp = xp
        self._elementwise_function_cache: dict[str, Callable[..., Any]] = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.xp.__name__})"

    @property
    def array_ns(self) -> ModuleType:
        """The library's own module, as exposed by `NdArrayField.array_ns`."""
        return self.xp

    def dtype(self, scalar_type: type) -> Any:
        return core_ndarray_utils.namespace_dtype(self.xp, core_defs.dtype(scalar_type).scalar_type)

    def scalar_type(self, dtype: Any) -> type[core_defs.Scalar]:
        return core_ndarray_utils.scalar_type(self.xp, dtype)

    def device(self, array: core_defs.NDArrayObject) -> Any:
        return array_api_compat.device(array)

    def asarray(
        self,
        data: Any,
        *,
        dtype: Any = None,
        like: core_defs.NDArrayObject | None = None,
    ) -> core_defs.NDArrayObject:
        """`data` as an array of this namespace, on the device of `like` if given."""
        return self.xp.asarray(
            data,
            dtype=None if dtype is None else self.dtype(dtype),
            device=None if like is None else self.device(like),
        )

    def operand(self, value: Any, like: core_defs.NDArrayObject) -> Any:
        """
        `value` as an operand of an elementwise function next to the array `like`.

        Python scalars stay Python scalars, so they keep the weak typing of the
        standard. NumPy scalars become 0-d arrays of their dtype, which promotes like
        NumPy does with NumPy scalars.
        """
        if isinstance(value, np.generic):
            return self.asarray(value, dtype=type(value), like=like)
        return value

    def elementwise_function(self, name: str) -> Callable[..., Any]:
        """The elementwise function for a gt4py builtin or operator name."""
        try:
            return self._elementwise_function_cache[name]
        except KeyError:
            func = self._elementwise_function_cache[name] = self._elementwise_function(name)
            return func

    def _elementwise_function(self, name: str) -> Callable[..., Any]:
        if name == "where":
            return self.where
        func = getattr(self.xp, STANDARD_NAMES.get(name, name), None)
        if func is None:
            raise NotImplementedError(
                f"'{name}' is not implemented for array namespace '{self.xp.__name__}'."
            )
        return func

    def where(self, condition: Any, x: Any, y: Any) -> core_defs.NDArrayObject:
        """
        `xp.where` for the operands of the `where` builtin.

        The standard requires an array `condition`, but the builtin also receives a
        scalar one (e.g. from `broadcast(True)`), which some libraries reject (CuPy,
        torch). It becomes a 0-d array on the device of the other operands. `x` and `y`
        are converted like the operands of any other elementwise function; the standard
        also requires one of them to be an array, so if both are scalars, `x` becomes a
        0-d array with the default dtype for its kind, as NumPy promotes it.
        """
        if isinstance(condition, bool | np.bool_):
            like = next(a for a in (x, y) if array_api_compat.is_array_api_obj(a))
            condition = self.asarray(condition, dtype=bool, like=like)
        x, y = self.operand(x, condition), self.operand(y, condition)
        if not (array_api_compat.is_array_api_obj(x) or array_api_compat.is_array_api_obj(y)):
            x = self.asarray(x, like=condition)
        return self.xp.where(condition, x, y)

    def setitem(
        self, array: core_defs.NDArrayObject, key: common.RelativeIndexSequence, value: Any
    ) -> core_defs.NDArrayObject:
        """Write `value` into `array[key]`; returns the array holding the result."""
        if not isinstance(value, bool | int | float):
            # the standard leaves assigning another dtype or device unspecified
            value = self.xp.asarray(value, device=self.device(array))
            if value.dtype != array.dtype:
                value = self.xp.astype(value, array.dtype)
        array[key] = value  # type: ignore[index] # `NDArrayObject` typing is not complete
        return array

    def to_numpy(self, array: core_defs.NDArrayObject) -> np.ndarray:
        if _NUMPY_FROM_DLPACK_TAKES_DEVICE:
            return np.from_dlpack(array, device="cpu", copy=None)  # type: ignore[arg-type] # `NDArrayObject` typing is not complete
        return np.from_dlpack(array)  # type: ignore[arg-type] # `NDArrayObject` typing is not complete

    def buffer_info(self, array: core_defs.NDArrayObject) -> common.BufferInfo:
        return common.BufferInfo.from_ndarray(array)


_registry: list[tuple[Callable[[ModuleType], bool], Callable[[ModuleType], ArrayOps]]] = []
_ops_by_namespace: dict[ModuleType, ArrayOps] = {}
_ops_by_array_type: dict[type, ArrayOps] = {}


def register(
    is_namespace: Callable[[ModuleType], bool], factory: Callable[[ModuleType], ArrayOps]
) -> None:
    """
    Use `factory(xp)` for the buffers whose compat namespace `xp` satisfies `is_namespace`.

    Later registrations take precedence.
    """
    _registry.insert(0, (is_namespace, factory))
    _ops_by_namespace.clear()
    _ops_by_array_type.clear()


def for_namespace(xp: ModuleType) -> ArrayOps:
    try:
        return _ops_by_namespace[xp]
    except KeyError:
        factory = next((f for is_ns, f in _registry if is_ns(xp)), ArrayOps)
        ops = _ops_by_namespace[xp] = factory(xp)
        return ops


def of(array: Any) -> ArrayOps:
    """The `ArrayOps` for the namespace of `array`; raises `TypeError` if it is not an array."""
    try:
        return _ops_by_array_type[type(array)]
    except KeyError:
        ops = _ops_by_array_type[type(array)] = for_namespace(
            array_api_compat.array_namespace(array)
        )
        return ops


def default() -> ArrayOps:
    """The `ArrayOps` used for data that is not an array, e.g. nested lists."""
    return for_namespace(compat_numpy)
