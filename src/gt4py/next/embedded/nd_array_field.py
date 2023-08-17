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

from gt4py._core import definitions as core_defs
from gt4py.next import common


try:
    import cupy as cp
except ImportError:
    cp: Optional[ModuleType] = None  # type:ignore[no-redef]

try:
    from jax import numpy as jnp
except ImportError:
    jnp: Optional[ModuleType] = None  # type:ignore[no-redef]

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
                domain_intersection = a.domain & b.domain
                a_slices = _get_slices_from_domain_slice(a.domain, domain_intersection)
                b_slices = _get_slices_from_domain_slice(b.domain, domain_intersection)
                new_data = op(a.ndarray[a_slices], b.ndarray[b_slices])
                return a.__class__.from_array(new_data, domain=domain_intersection)
            new_data = op(a.ndarray, xp.asarray(b.ndarray))
        else:
            # assert isinstance(b, core_defs.SCALAR_TYPES) # TODO reenable this assert (if b is an array it should be wrapped into a field)
            new_data = op(a.ndarray, b)

        return a.__class__.from_array(new_data, domain=a.domain)

    _builtin_binary_op.__name__ = builtin_name
    return _builtin_binary_op


_Value: TypeAlias = common.Field | core_defs.ScalarT
_P = ParamSpec("_P")
_R = TypeVar("_R", _Value, tuple[_Value, ...])


@dataclasses.dataclass(frozen=True)
class _BaseNdArrayField(common.FieldABC[common.DimsT, core_defs.ScalarT]):
    """
    Shared field implementation for NumPy-like fields.

    Builtin function implementations are registered in a dictionary.
    Note: Currently, all concrete NdArray-implementations share
    the same implementation, dispatching is handled inside of the registered
    function via its namespace.
    """

    _domain: common.Domain
    _ndarray: core_defs.NDArrayObject
    _value_type: type[core_defs.ScalarT]

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
    def domain(self) -> common.Domain:
        return self._domain

    @property
    def __gt_dims__(self) -> common.DimsT:
        return self._domain.dims

    @property
    def __gt_origin__(self) -> tuple[int, ...]:
        return tuple(-r.start for _, r in self._domain)

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        return self._ndarray

    def __array__(self) -> np.ndarray:
        return np.asarray(self._ndarray)

    @property
    def value_type(self) -> type[core_defs.ScalarT]:
        return self._value_type

    @property
    def dtype(self):
        return self.array_ns.dtype(self._value_type)

    @classmethod
    def from_array(
        cls,
        data: npt.ArrayLike,
        /,
        *,
        domain: common.Domain,
        value_type: Optional[type] = None,
    ) -> _BaseNdArrayField:
        xp = cls.array_ns
        dtype = None
        if value_type is not None:
            dtype = xp.dtype(value_type)
        array = xp.asarray(data, dtype=dtype)

        value_type = array.dtype.type  # TODO add support for Dimensions as value_type

        assert issubclass(array.dtype.type, core_defs.SCALAR_TYPES)

        assert all(isinstance(d, common.Dimension) for d in domain.dims), domain
        assert len(domain) == array.ndim
        assert all(len(r) == s for r, s in zip(domain.ranges, array.shape))

        assert value_type is not None  # for mypy
        return cls(domain, array, value_type)

    def remap(self: _BaseNdArrayField, connectivity) -> _BaseNdArrayField:
        raise NotImplementedError()

    def restrict(
        self: _BaseNdArrayField, domain_slice: common.Domain | common.DomainSlice | common.Position
    ) -> _BaseNdArrayField | _Value:
        if common.is_domain_slice(domain_slice):
            return self._getitem_domain_slice(domain_slice)
        else:
            return self.ndarray[domain_slice]  # TODO should return field

    def __setitem__(self, domain, value):
        slices = _get_slices_from_domain_slice(self.domain, domain)
        self.ndarray[slices] = value

    def _getitem_domain_slice(self, index: common.DomainSlice) -> common.Field:
        slices = _get_slices_from_domain_slice(self.domain, index)

        dims = []
        ranges = []
        for k, v in index:
            if not common.is_int_index(v):
                dims.append(k)
                ranges.append(v)

        new = self.ndarray[slices]
        if len(dims) == 0:
            return new  # scalar
        else:
            new_domain = common.Domain(tuple(dims), tuple(ranges))
            return self.__class__(new_domain, new, self.value_type)

    __call__ = None  # type: ignore[assignment]  # TODO: remap

    __getitem__ = restrict

    __abs__ = _make_unary_array_field_intrinsic_func("abs", "abs")

    __neg__ = _make_unary_array_field_intrinsic_func("neg", "negative")

    def __pos__(self):
        return self

    __add__ = __radd__ = _make_binary_array_field_intrinsic_func("add", "add")

    __sub__ = __rsub__ = _make_binary_array_field_intrinsic_func("sub", "subtract")

    __mul__ = __rmul__ = _make_binary_array_field_intrinsic_func("mul", "multiply")

    __truediv__ = __rtruediv__ = _make_binary_array_field_intrinsic_func("div", "divide")

    __floordiv__ = __rfloordiv__ = _make_binary_array_field_intrinsic_func(
        "floordiv", "floor_divide"
    )

    __pow__ = _make_binary_array_field_intrinsic_func("pow", "power")

    __mod__ = __rmod__ = _make_binary_array_field_intrinsic_func("mod", "mod")

    __and__ = __rand__ = _make_binary_array_field_intrinsic_func("bitwise_and", "bitwise_and")
    __or__ = __ror__ = _make_binary_array_field_intrinsic_func("bitwise_or", "bitwise_or")
    __xor__ = __rxor__ = _make_binary_array_field_intrinsic_func("bitwise_xor", "bitwise_xor")

    __invert__ = _make_unary_array_field_intrinsic_func("invert", "invert")


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


def _get_slices_from_domain_slice(
    domain: common.Domain, domain_slice: common.DomainSlice
) -> tuple[slice | int | None, ...]:
    """Generate slices for sub-array extraction based on named ranges or named indices within a Domain.

    This function generates a tuple of slices that can be used to extract sub-arrays from a field. The provided
    named ranges or indices specify the dimensions and ranges of the sub-arrays to be extracted.

    Args:
        domain (common.Domain): The Domain object representing the original field.
        domain_slice (DomainSlice): A sequence of dimension names and associated ranges.

    Returns:
        tuple[slice | int | None, ...]: A tuple of slices representing the sub-array extraction along each dimension
                                       specified in the Domain. If a dimension is not included in the named indices
                                       or ranges, a None is used to indicate expansion along that axis.
    """
    slice_indices: list[slice | int | None] = []

    for new_dim, new_rng in domain_slice:
        pos_new = next(index for index, (dim, _) in enumerate(domain_slice) if dim == new_dim)

        if new_dim in domain.dims:
            pos_old = domain.dims.index(new_dim)
            slice_indices.append(_compute_slice(new_rng, domain, pos_old))
        else:
            slice_indices.insert(pos_new, None)  # None is equal to np.newaxis

    return tuple(slice_indices)


def _compute_slice(rng: common.DomainRange, domain: common.Domain, pos: int) -> slice | int:
    """Compute a slice or integer based on the provided range, domain, and position.

    Args:
        rng (DomainRange): The range to be computed as a slice or integer.
        domain (common.Domain): The domain containing dimension information.
        pos (int): The position of the dimension in the domain.

    Returns:
        slice | int: Slice if `new_rng` is a UnitRange, otherwise an integer.

    Raises:
        ValueError: If `new_rng` is not an integer or a UnitRange.
    """
    if isinstance(rng, common.UnitRange):
        return slice(
            rng.start - domain.ranges[pos].start,
            rng.stop - domain.ranges[pos].start,
        )
    elif common.is_int_index(rng):
        return rng - domain.ranges[pos].start
    else:
        raise ValueError(f"Can only use integer or UnitRange ranges, provided type: {type(rng)}")
