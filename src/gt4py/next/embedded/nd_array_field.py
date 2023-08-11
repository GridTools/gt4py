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
from collections.abc import Callable, Sequence
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

from gt4py._core import definitions
from gt4py._core.definitions import ScalarT
from gt4py.next.common import Dimension, DimsT, Domain, DomainRange, DomainSlice, UnitRange
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
            assert isinstance(b, definitions.SCALAR_TYPES)
            new_data = op(a.ndarray, b)

        return a.__class__.from_array(new_data, domain=a.domain)

    _builtin_binary_op.__name__ = builtin_name
    return _builtin_binary_op


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

    @overload
    def __getitem__(self, index: DomainSlice) -> common.Field:
        """Absolute slicing with dimension names."""
        ...

    @overload
    def __getitem__(self, index: tuple[slice | int, ...]) -> common.Field:
        """Relative slicing with ordered dimension access."""
        ...

    @overload
    def __getitem__(self, index: tuple[int, ...]) -> core_defs.DType[core_defs.ScalarT]:
        ...

    @overload
    def __getitem__(
        self, index: Sequence[common.NamedIndex]
    ) -> common.Field | core_defs.DType[core_defs.ScalarT]:
        # Value in case len(i) == len(self.domain)
        ...

    def __getitem__(
        self, index: DomainSlice | Sequence[common.NamedIndex] | tuple[int, ...]
    ) -> common.Field | core_defs.DType[core_defs.ScalarT]:
        if isinstance(index, Sequence):
            if all(
                isinstance(idx, tuple)
                and isinstance(idx[0], Dimension)
                and isinstance(idx[1], UnitRange)
                or isinstance(idx[1], int)
                for idx in index
            ):
                return self._getitem_domain_slice(index)
            return self._getitem_named_index_sequence(index)
        elif isinstance(index, tuple) and all(isinstance(idx, (slice, int)) for idx in index):
            return self._getitem_relative_slice(index)
        elif isinstance(index, tuple) and all(isinstance(idx, int) for idx in index):
            return self._getitem_tuple_int(index)
        else:
            raise IndexError("Unsupported index type")

    def _getitem_domain_slice(self, index: DomainSlice) -> common.Field:
        slices = _get_slices_from_domain_slice(self.domain, index)
        new = self.ndarray[slices]
        return common.field(new, domain=index)

    def _getitem_relative_slice(self, index: tuple[slice | int, ...]) -> common.Field:
        # TODO: implement relative slicing
        ...

    def _getitem_tuple_int(self, index: tuple[int, ...]) -> core_defs.DType[core_defs.ScalarT]:
        # TODO: implement tuple of integers indexing
        ...


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
    domain: Domain, domain_slice: DomainSlice
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


def _compute_slice(rng: DomainRange, domain: Domain, pos: int) -> slice | int:
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
    if isinstance(rng, UnitRange):
        return slice(
            rng.start - domain.ranges[pos].start,
            rng.stop - domain.ranges[pos].start,
        )
    elif isinstance(rng, int):
        return rng
    else:
        raise ValueError(f"Can only use integer or UnitRange ranges, provided type: {type(rng)}")
