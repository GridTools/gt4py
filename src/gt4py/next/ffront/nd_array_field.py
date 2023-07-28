# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from types import ModuleType
from typing import Any, ClassVar, Optional, ParamSpec, TypeVar, overload

import numpy as np
from numpy import typing as npt

from gt4py.next import common


try:
    import cupy as cp
except ImportError:
    cp: Optional[ModuleType] = None  # type:ignore[no-redef]

try:
    from jax import numpy as jnp
except ImportError:
    jnp: Optional[ModuleType] = None  # type:ignore[no-redef]


from gt4py._core import definitions
from gt4py._core.definitions import ScalarT
from gt4py.next.common import DimsT, DomainT
from gt4py.next.ffront import fbuiltins


def _make_unary_array_field_intrinsic_func(builtin_name: str, array_builtin_name: str) -> Callable:
    def _builtin_unary_op(a: _BaseNdArrayField) -> common.Field:
        xp = a.__class__.array_ns
        op = getattr(xp, array_builtin_name)
        new_data = op(a.ndarray)

        return a.__class__.from_array(new_data, domain=a.domain)

    _builtin_unary_op.__name__ = builtin_name
    return _builtin_unary_op


def _make_binary_array_field_intrinsic_func(builtin_name: str, array_builtin_name: str) -> Callable:
    def _builtin_binary_op(a: _BaseNdArrayField, b: common.Field) -> common.Field:
        xp = a.__class__.array_ns
        op = getattr(xp, array_builtin_name)
        if hasattr(b, "__gt_builtin_func__"):  # isinstance(b, common.Field):
            if not a.domain == b.domain:
                raise NotImplementedError(
                    f"support for different domain not implemented: {a.domain}, {b.domain}"
                )
            new_data = op(a.ndarray, xp.asarray(b.ndarray))
        else:
            assert isinstance(b, definitions.SCALAR_TYPES)
            new_data = op(a.ndarray, b)

        return a.__class__.from_array(new_data, domain=a.domain)

    _builtin_binary_op.__name__ = builtin_name
    return _builtin_binary_op


Value = Any  # TODO
P = ParamSpec("P")
R = TypeVar("R", Value, tuple[Value, ...])


@dataclasses.dataclass(frozen=True)
class _BaseNdArrayField(common.FieldABC[DimsT, ScalarT]):
    """
    Shared field implementation for NumPy-like fields.

    Builtin function implementations are registered in a dictionary.
    Note: Currently, all concrete NdArray-implementations share
    the same implementation, dispatching is handled inside of the registered
    function via its namespace.
    """

    _domain: DomainT
    _ndarray: definitions.NDArrayObject
    _value_type: type[ScalarT]

    array_ns: ClassVar[ModuleType]

    _builtin_func_map: ClassVar[dict[fbuiltins.BuiltInFunction, Callable]] = {}

    @classmethod
    def __gt_builtin_func__(cls, func: fbuiltins.BuiltInFunction[R, P], /) -> Callable[P, R]:
        return cls._builtin_func_map.get(func, NotImplemented)

    @overload
    @classmethod
    def register_builtin_func(
        cls, op: fbuiltins.BuiltInFunction[R, P], op_func: None
    ) -> functools.partial[Callable[P, R]]:
        ...

    @overload
    @classmethod
    def register_builtin_func(
        cls, op: fbuiltins.BuiltInFunction[R, P], op_func: Callable[P, R]
    ) -> Callable[P, R]:
        ...

    @classmethod
    def register_builtin_func(
        cls, op: fbuiltins.BuiltInFunction[R, P], op_func: Optional[Callable[P, R]] = None
    ) -> Callable[P, R] | functools.partial[Callable[P, R]]:
        assert op not in cls._builtin_func_map
        if op_func is None:  # when used as a decorator
            return functools.partial(cls.register_builtin_func, op)  # type: ignore[arg-type]
        return cls._builtin_func_map.setdefault(op, op_func)

    @property
    def domain(self) -> DomainT:
        return self._domain

    @property
    def ndarray(self) -> definitions.NDArrayObject:
        return self._ndarray

    @property
    def value_type(self) -> type[definitions.ScalarT]:
        return self._value_type

    @classmethod
    def from_array(
        cls,
        data: npt.ArrayLike,
        /,
        *,
        domain: DomainT,
        value_type: Optional[type] = None,
    ) -> _BaseNdArrayField:
        xp = cls.array_ns
        dtype = None
        if value_type is not None:
            dtype = xp.dtype(value_type)
        array = xp.asarray(data, dtype=dtype)

        value_type = array.dtype.type  # TODO add support for Dimensions as value_type

        assert issubclass(array.dtype.type, definitions.SCALAR_TYPES)

        assert all(isinstance(d, common.Dimension) for d, r in domain), domain
        assert len(domain) == array.ndim
        assert all(len(nr[1]) == s for nr, s in zip(domain, array.shape))

        assert value_type is not None  # for mypy
        return cls(domain, array, value_type)

    def remap(self: _BaseNdArrayField, connectivity) -> _BaseNdArrayField:
        raise NotImplementedError()

    def restrict(self: _BaseNdArrayField, domain) -> _BaseNdArrayField:
        raise NotImplementedError()

    __call__ = None  # type: ignore[assignment]  # TODO: remap

    __getitem__ = None  # type: ignore[assignment]  # TODO: restrict

    __abs__ = _make_unary_array_field_intrinsic_func("abs", "abs")

    __neg__ = _make_unary_array_field_intrinsic_func("neg", "negative")

    __add__ = __radd__ = _make_binary_array_field_intrinsic_func("add", "add")

    __sub__ = __rsub__ = _make_binary_array_field_intrinsic_func("sub", "subtract")

    __mul__ = __rmul__ = _make_binary_array_field_intrinsic_func("mul", "multiply")

    __truediv__ = __rtruediv__ = _make_binary_array_field_intrinsic_func("div", "divide")

    __floordiv__ = __rfloordiv__ = _make_binary_array_field_intrinsic_func(
        "floordiv", "floor_divide"
    )

    __pow__ = _make_binary_array_field_intrinsic_func("pow", "power")


# -- Specialized implementations for intrinsic operations on array fields --

_BaseNdArrayField.register_builtin_func(fbuiltins.abs, _BaseNdArrayField.__abs__)  # type: ignore[attr-defined]
_BaseNdArrayField.register_builtin_func(fbuiltins.power, _BaseNdArrayField.__pow__)  # type: ignore[attr-defined]
# TODO gamma

for name in (
    fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES
    + fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES
    + fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES
):
    if name in ["abs", "power", "gamma"]:
        continue
    _BaseNdArrayField.register_builtin_func(
        getattr(fbuiltins, name), _make_unary_array_field_intrinsic_func(name, name)
    )

_BaseNdArrayField.register_builtin_func(
    fbuiltins.minimum, _make_binary_array_field_intrinsic_func("minimum", "minimum")  # type: ignore[attr-defined]
)
_BaseNdArrayField.register_builtin_func(
    fbuiltins.maximum, _make_binary_array_field_intrinsic_func("maximum", "maximum")  # type: ignore[attr-defined]
)
_BaseNdArrayField.register_builtin_func(
    fbuiltins.fmod, _make_binary_array_field_intrinsic_func("fmod", "fmod")  # type: ignore[attr-defined]
)

# -- Concrete array implementations --
# NumPy
_nd_array_implementations = [np]


@dataclasses.dataclass(frozen=True)
class NumPyArrayField(_BaseNdArrayField):
    array_ns: ClassVar[ModuleType] = np


common.field.register(np.ndarray, NumPyArrayField.from_array)


# CuPy
if cp:
    _nd_array_implementations.append(cp)

    @dataclasses.dataclass(frozen=True)
    class CuPyArrayField(_BaseNdArrayField):
        array_ns: ClassVar[ModuleType] = cp

    common.field.register(cp.ndarray, CuPyArrayField.from_array)

# JAX
if jnp:
    _nd_array_implementations.append(jnp)

    @dataclasses.dataclass(frozen=True)
    class JaxArrayField(_BaseNdArrayField):
        array_ns: ClassVar[ModuleType] = jnp

    common.field.register(jnp.ndarray, JaxArrayField.from_array)
