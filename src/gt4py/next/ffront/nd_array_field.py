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

import abc
import dataclasses
import functools
import math
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Generic,
    NamedTuple,
    Optional,
    TypeVar,
    final,
)

import numpy as np
from numpy import typing as npt
from typing_extensions import TypeVarTuple, Unpack

from gt4py.next import common


try:
    import cupy as cp
except ImportError:
    cp: Final = None

try:
    from jax import numpy as jnp
except ImportError:
    jnp: Final = None


from gt4py._core import definitions
from gt4py._core.definitions import ScalarT
from gt4py.next.common import DomainT
from gt4py.next.ffront import fbuiltins


class NonProtocolABCMeta(typing._ProtocolMeta):
    __instancecheck__ = abc.ABCMeta.__instancecheck__


class NonProtocolABC(metaclass=NonProtocolABCMeta):
    pass


def _make_unary_array_field_intrinsic_func(builtin_name: str, array_builtin_name: str) -> Callable:
    def _builtin_unary_op(a: _BaseNdArrayField) -> definitions.Field:
        xp = a.__class__.array_ns
        op = getattr(xp, array_builtin_name)
        new_data = op(a.ndarray)

        return a.__class__.from_array(new_data, domain=a.domain)

    _builtin_unary_op.__name__ = builtin_name
    return _builtin_unary_op


def _make_binary_array_field_intrinsic_func(builtin_name: str, array_builtin_name: str) -> Callable:
    def _builtin_binary_op(a: _BaseNdArrayField, b: definitions.Field) -> definitions.Field:
        xp = a.__class__.array_ns
        op = getattr(xp, array_builtin_name)
        if hasattr(b, "__gt_op_func__"):  # isinstance(b, definitions.Field):
            if not a.domain == b.domain:
                raise NotImplementedError(
                    f"support for different domain not implemented: {a.domain}, {b.domain}"
                )
            new_data = op(a.ndarray, xp.asarray(b.ndarray))
        else:
            assert isinstance(b, definitions.Scalar)
            new_data = op(a.ndarray, b)

        return a.__class__.from_array(new_data, domain=a.domain)

    _builtin_binary_op.__name__ = builtin_name
    return _builtin_binary_op


@dataclasses.dataclass(frozen=True)
class _BaseNdArrayField(NonProtocolABC, common.Field[common.DimsT, ScalarT]):
    _domain: DomainT
    _ndarray: definitions.NDArrayObject
    _value_type: ScalarT

    _ops_mapping: ClassVar[dict[fbuiltins.BuiltInFunction, Callable]] = {}
    array_ns: ClassVar[definitions.NDArrayObject]

    @classmethod
    def __gt_op_func__(cls, op: definitions.IntrinsicOp[R, P]) -> Callable[P, R]:
        return cls._ops_mapping.get(op, NotImplemented)

    @classmethod
    def register_gt_op_func(
        cls, op: definitions.IntrinsicOp[R, P], op_func: Optional[Callable[P, R]] = None
    ) -> Callable[P, R]:
        assert op not in cls._ops_mapping
        if op_func is None:
            return functools.partial(cls.register_gt_op_func, op)
        return cls._ops_mapping.setdefault(op, op_func)

    @property
    def domain(self) -> DomainT:
        return self._domain

    @property
    def ndarray(self) -> definitions.NDArrayObject:
        return self._ndarray

    @property
    def value_type(self) -> definitions.ScalarT:
        return self._value_type

    @classmethod
    def from_array(
        cls,
        data: npt.ArrayLike,
        /,
        *,
        domain: Optional[DomainT] = None,
        value_type: Optional[type] = None,
    ) -> _BaseNdArrayField:
        xp = cls.array_ns
        dtype = None
        if value_type is not None:
            dtype = xp.dtype(value_type)
        array = xp.asarray(data, dtype=dtype)
        if value_type is not None and isinstance(value_type, definitions.DimensionType):
            if not (
                xp.min(array) >= value_type.extent.start and xp.max(array) < value_type.extent.stop
            ):
                raise ValueError(f"Impossible to interpret data to {value_type}")
        else:
            value_type = array.dtype.type

        assert issubclass(array.dtype.type, definitions.Scalar)

        assert all(isinstance(d, common.Dimension) for d, r in domain), domain
        assert len(domain) == array.ndim
        assert all(len(nr[1]) == s for nr, s in zip(domain, array.shape))

        return cls(domain, array, value_type)

    def remap(
        self: _BaseNdArrayField, connectivity: definitions.ConnectivityField
    ) -> definitions.Field:
        raise NotImplementedError()

    def restrict(self: _BaseNdArrayField, domain: "DomainLike") -> _BaseNdArrayField:
        raise NotImplementedError()

    # __call__ = remap
    __call__ = None

    # __getitem__ = restrict
    __getitem__ = None

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

_BaseNdArrayField.register_gt_op_func(fbuiltins.abs, _BaseNdArrayField.__abs__)
_BaseNdArrayField.register_gt_op_func(fbuiltins.power, _BaseNdArrayField.__pow__)
# TODO gamma

for name in (
    fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES
    + fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES
    + fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES
):
    if name in ["abs", "power", "gamma"]:
        continue
    _BaseNdArrayField.register_gt_op_func(
        getattr(fbuiltins, name), _make_unary_array_field_intrinsic_func(name, name)
    )

_BaseNdArrayField.register_gt_op_func(
    fbuiltins.minimum, _make_binary_array_field_intrinsic_func("minimum", "minimum")
)
_BaseNdArrayField.register_gt_op_func(
    fbuiltins.maximum, _make_binary_array_field_intrinsic_func("maximum", "maximum")
)
_BaseNdArrayField.register_gt_op_func(
    fbuiltins.fmod, _make_binary_array_field_intrinsic_func("fmod", "fmod")
)

# # Domain-related
# _BaseNdArrayField.register_gt_op_func(fbuiltins.remap, _BaseNdArrayField.remap)
# _BaseNdArrayField.register_gt_op_func(fbuiltins.restrict, _BaseNdArrayField.restrict)


# -- Concrete array implementations --
# NumPy
_nd_array_implementations = [np]


@dataclasses.dataclass(frozen=True)
class NumPyArrayField(_BaseNdArrayField):
    array_ns: ClassVar[definitions.NDArrayObject] = np


common.field.register(np.ndarray, NumPyArrayField.from_array)
# common.field.register(memoryview, NumPyArrayField.from_array)
# common.field.register(Iterable, NumPyArrayField.from_array)


# CuPy
if cp:
    _nd_array_implementations.append(cp)

    @dataclasses.dataclass(frozen=True)
    class CuPyArrayField(_BaseNdArrayField):
        array_ns: ClassVar[definitions.NDArrayObject] = cp

    definitions.field.register(cp.ndarray, CuPyArrayField.from_array)

# JAX
if jnp:
    _nd_array_implementations.append(jnp)

    @dataclasses.dataclass(frozen=True)
    class JaxArrayField(_BaseNdArrayField):
        array_ns: ClassVar[definitions.NDArrayObject] = jnp

    common.field.register(jnp.ndarray, JaxArrayField.from_array)
