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
            assert a.domain == b.domain
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

    # def __post_init__(self) -> None:
    #     if self._extended_domain is None:
    #         assert self._extended_ndarray is None
    #         object.__setattr__(self, "_extended_domain", self._domain)
    #         object.__setattr__(self, "_extended_ndarray", self._ndarray)
    #         object.__setattr__(self, "_ndarray", self._extended_ndarray.view())

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

        # assert all(isinstance(d, definitions.DimensionType) for d in domain)
        # assert len(domain) == array.ndim
        # assert all(len(d) == s for d, s in zip(domain, array.shape))

        # return cls(definitions.Domain(*domain), array, value_type)
        return cls(domain, array, value_type)

    def remap(
        self: _BaseNdArrayField, connectivity: definitions.ConnectivityField
    ) -> definitions.Field:
        raise NotImplementedError()

    #     assert isinstance(connectivity.value_type, definitions.DimensionType)

    #     # TODO: if connectivity is a regular connectivity use shift-like implementation
    #     # if is_structured(connectivity):
    #     #     return _shift_structured(field, connectivity)

    #     # restrict index field
    #     dim_idx = self.domain.tag_index(connectivity.value_type.tag)
    #     if dim_idx is None:
    #         raise ValueError(f"Incompatible index field.")
    #     current_dim: definitions.DimensionType = self.domain[dim_idx]

    #     new_dims = connectivity.inverse_image(current_dim)
    #     restricted_domain = definitions.Domain(*new_dims)
    #     restricted_connectivity = (
    #         connectivity.restrict(restricted_domain)
    #         if restricted_domain != connectivity.domain
    #         else connectivity
    #     )

    #     # perform contramap
    #     xp = self.array_ns
    #     dim_idx = self.domain.tag_index(restricted_connectivity.value_type.tag)
    #     new_domain = self._domain.replace_at(dim_idx, restricted_connectivity.domain)
    #     new_idx_array = (
    #         xp.asarray(restricted_connectivity.ndarray) - self.domain[dim_idx].extent.start
    #     )
    #     new_data = xp.take(self._ndarray, new_idx_array, axis=dim_idx)

    #     return self.__class__.from_array(new_data, domain=new_domain, value_type=self.value_type)

    def restrict(self: _BaseNdArrayField, domain: "DomainLike") -> _BaseNdArrayField:
        raise NotImplementedError()

    #     new_domain = None
    #     if isinstance(domain, definitions.Domain):
    #         new_domain = domain
    #     else:
    #         if not isinstance(domain, Sequence):
    #             domain = (domain,)
    #         if all(isinstance(d, definitions.DimensionType) for d in domain):
    #             new_domain = definitions.Domain(*domain)
    #         elif all(isinstance(d, definitions.DimensionIndex) for d in domain):
    #             new_domain = definitions.Domain(*(d.__class__ & [d, d] for d in domain))

    #     if new_domain is None:
    #         raise IndexError(f"Invalid slice definition {domain}")
    #     if not new_domain in self._domain:
    #         raise IndexError(f"Invalid subdomain {new_domain=}")

    #     view_slice = tuple(
    #         slice(start := d.point_index(new_d.extent.start), start + len(new_d))
    #         for d, new_d in zip(self._extended_domain, new_domain)
    #     )
    #     view_array = self._extended_ndarray[view_slice]

    #     return dataclasses.replace(self, _domain=new_domain, _ndarray=view_array)

    # __call__ = remap
    __call__ = None

    # __getitem__ = restrict
    __getitem__ = None

    __floordiv__ = None
    __rfloordiv__ = None

    __abs__ = _make_unary_array_field_intrinsic_func("abs", "abs")

    __neg__ = _make_unary_array_field_intrinsic_func("neg", "negative")

    __add__ = __radd__ = _make_binary_array_field_intrinsic_func("add", "add")

    __sub__ = __rsub__ = _make_binary_array_field_intrinsic_func("sub", "subtract")

    __mul__ = __rmul__ = _make_binary_array_field_intrinsic_func("mul", "multiply")

    __truediv__ = __rtruediv__ = _make_binary_array_field_intrinsic_func("div", "divide")

    __pow__ = _make_binary_array_field_intrinsic_func("pow", "power")


# -- Specialized implementations for intrinsic operations on array fields --

# for name in ["abs", "neg", "add", "sub", "mul", "pow"]:  # div -> truediv
#     _BaseNdArrayField.register_gt_op_func(
#         getattr(fbuiltins, name), getattr(_BaseNdArrayField, f"__{name}__")
#     )

# _BaseNdArrayField.register_gt_op_func(fbuiltins.div, _BaseNdArrayField.__truediv__)


_BaseNdArrayField.register_gt_op_func(
    fbuiltins.isfinite, _make_unary_array_field_intrinsic_func("isfinite", "isfinite")
)
_BaseNdArrayField.register_gt_op_func(
    fbuiltins.isinf, _make_unary_array_field_intrinsic_func("isinf", "isinf")
)
_BaseNdArrayField.register_gt_op_func(
    fbuiltins.isnan, _make_unary_array_field_intrinsic_func("isnan", "isnan")
)
_BaseNdArrayField.register_gt_op_func(
    fbuiltins.sin, _make_unary_array_field_intrinsic_func("sin", "sin")
)
_BaseNdArrayField.register_gt_op_func(
    fbuiltins.cos, _make_unary_array_field_intrinsic_func("cos", "cos")
)
_BaseNdArrayField.register_gt_op_func(
    fbuiltins.tan, _make_unary_array_field_intrinsic_func("tan", "tan")
)
_BaseNdArrayField.register_gt_op_func(
    fbuiltins.arcsin, _make_unary_array_field_intrinsic_func("arcsin", "arcsin")
)
_BaseNdArrayField.register_gt_op_func(
    fbuiltins.arccos, _make_unary_array_field_intrinsic_func("arccos", "arccos")
)
_BaseNdArrayField.register_gt_op_func(
    fbuiltins.arctan, _make_unary_array_field_intrinsic_func("arctan", "arctan")
)

# sinh, cosh, ...

_BaseNdArrayField.register_gt_op_func(
    fbuiltins.floor, _make_unary_array_field_intrinsic_func("floor", "floor")
)
_BaseNdArrayField.register_gt_op_func(
    fbuiltins.ceil, _make_unary_array_field_intrinsic_func("ceil", "ceil")
)
_BaseNdArrayField.register_gt_op_func(
    fbuiltins.trunc, _make_unary_array_field_intrinsic_func("trunc", "trunc")
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
@dataclasses.dataclass(frozen=True)
class NumPyArrayField(_BaseNdArrayField):
    array_ns: ClassVar[definitions.NDArrayObject] = np


common.field.register(np.ndarray, NumPyArrayField.from_array)
# common.field.register(memoryview, NumPyArrayField.from_array)
# common.field.register(Iterable, NumPyArrayField.from_array)


# @dataclasses.dataclass(frozen=True)
# class NumPyArrayConnectivityField(_BaseNdArrayConnectivityField):
#     array_ns: ClassVar[definitions.NDArrayObject] = np

#     __hash__ = _BaseNdArrayConnectivityField.__hash__


# definitions.connectivity.register(np.ndarray, NumPyArrayConnectivityField.from_array)
# definitions.connectivity.register(memoryview, NumPyArrayConnectivityField.from_array)
# definitions.connectivity.register(Iterable, NumPyArrayConnectivityField.from_array)

if jnp:

    @dataclasses.dataclass(frozen=True)
    class JaxArrayField(_BaseNdArrayField):
        array_ns: ClassVar[definitions.NDArrayObject] = jnp

    common.field.register(jnp.ndarray, JaxArrayField.from_array)

    # @dataclasses.dataclass(frozen=True)
    # class JaxArrayConnectivityField(_BaseNdArrayConnectivityField):
    #     array_ns: ClassVar[definitions.NDArrayObject] = jnp

    #     __hash__ = _BaseNdArrayConnectivityField.__hash__

    # definitions.field.register(jnp.ndarray, JaxArrayConnectivityField.from_array)
