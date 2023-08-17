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
from typing import ClassVar, Optional, ParamSpec, TypeAlias, TypeVar, overload

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
from gt4py.next.common import DimsT, Domain
from gt4py.next.ffront import fbuiltins


def _make_builtin(builtin_name: str, array_builtin_name: str) -> Callable:
    def _builtin_op(*fields: common.Field) -> common.Field:
        first = fields[0]
        assert isinstance(first, _BaseNdArrayField)
        xp = first.__class__.array_ns
        op = getattr(xp, array_builtin_name)

        others_transformed = []
        if len(fields) > 1:
            for other in fields[1:]:
                if hasattr(other, "__gt_builtin_func__"):  # isinstance(b, common.Field):
                    if not first.domain == other.domain:
                        raise NotImplementedError(
                            f"support for different domain not implemented: {first.domain}, {other.domain}"
                        )
                    others_transformed.append(xp.asarray(other.ndarray))
                else:
                    assert isinstance(other, definitions.SCALAR_TYPES)
                    others_transformed.append(other)

        new_data = op(first.ndarray, *others_transformed)
        return first.__class__.from_array(new_data, domain=first.domain)

    _builtin_op.__name__ = builtin_name
    return _builtin_op


_Value: TypeAlias = common.Field | ScalarT
_P = ParamSpec("_P")
_R = TypeVar("_R", _Value, tuple[_Value, ...])


@dataclasses.dataclass(frozen=True)
class _BaseNdArrayField(common.FieldABC[DimsT, ScalarT]):
    """
    Shared field implementation for NumPy-like fields.

    Builtin function implementations are registered in a dictionary.
    Note: Currently, all concrete NdArray-implementations share
    the same implementation, dispatching is handled inside of the registered
    function via its namespace.
    """

    _domain: Domain
    _ndarray: definitions.NDArrayObject
    _value_type: type[ScalarT]

    array_ns: ClassVar[
        ModuleType
    ]  # TODO(havogt) after storage PR is merged, update to the NDArrayNamespace protocol

    _builtin_func_map: ClassVar[dict[fbuiltins.BuiltInFunction, Callable]] = {}

    @classmethod
    def __gt_builtin_func__(cls, func: fbuiltins.BuiltInFunction[_R, _P], /) -> Callable[_P, _R]:
        return cls._builtin_func_map.get(func, NotImplemented)

    @overload
    @classmethod
    def register_builtin_func(
        cls, op: fbuiltins.BuiltInFunction[_R, _P], op_func: None
    ) -> functools.partial[Callable[_P, _R]]:
        ...

    @overload
    @classmethod
    def register_builtin_func(
        cls, op: fbuiltins.BuiltInFunction[_R, _P], op_func: Callable[_P, _R]
    ) -> Callable[_P, _R]:
        ...

    @classmethod
    def register_builtin_func(
        cls, op: fbuiltins.BuiltInFunction[_R, _P], op_func: Optional[Callable[_P, _R]] = None
    ) -> Callable[_P, _R] | functools.partial[Callable[_P, _R]]:
        assert op not in cls._builtin_func_map
        if op_func is None:  # when used as a decorator
            return functools.partial(cls.register_builtin_func, op)  # type: ignore[arg-type]
        return cls._builtin_func_map.setdefault(op, op_func)

    @property
    def domain(self) -> Domain:
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
        domain: Domain,
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

    __abs__ = _make_builtin("abs", "abs")

    __neg__ = _make_builtin("neg", "negative")

    __add__ = __radd__ = _make_builtin("add", "add")

    __sub__ = __rsub__ = _make_builtin("sub", "subtract")

    __mul__ = __rmul__ = _make_builtin("mul", "multiply")

    __truediv__ = __rtruediv__ = _make_builtin("div", "divide")

    __floordiv__ = __rfloordiv__ = _make_builtin("floordiv", "floor_divide")

    __pow__ = _make_builtin("pow", "power")


# -- Specialized implementations for builtin operations on array fields --

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
    _BaseNdArrayField.register_builtin_func(getattr(fbuiltins, name), _make_builtin(name, name))

_BaseNdArrayField.register_builtin_func(
    fbuiltins.minimum, _make_builtin("minimum", "minimum")  # type: ignore[attr-defined]
)
_BaseNdArrayField.register_builtin_func(
    fbuiltins.maximum, _make_builtin("maximum", "maximum")  # type: ignore[attr-defined]
)
_BaseNdArrayField.register_builtin_func(
    fbuiltins.fmod, _make_builtin("fmod", "fmod")  # type: ignore[attr-defined]
)
_BaseNdArrayField.register_builtin_func(fbuiltins.where, _make_builtin("where", "where"))

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
