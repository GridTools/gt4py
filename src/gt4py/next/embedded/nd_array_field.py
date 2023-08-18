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
from gt4py.next.common import (
    Dimension,
    DimsT,
    Domain,
    DomainRange,
    DomainSlice,
    FieldSlice,
    NamedIndex,
    NamedRange,
    UnitRange,
    contains_only_one_value,
    is_domain_slice,
)
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
                a_broadcasted = _broadcast(a, domain_intersection.dims)
                b_broadcasted = _broadcast(b, domain_intersection.dims)
                a_slices = _get_slices_from_domain_slice(a_broadcasted.domain, domain_intersection)
                b_slices = _get_slices_from_domain_slice(b_broadcasted.domain, domain_intersection)
                new_data = op(a_broadcasted.ndarray[a_slices], b_broadcasted.ndarray[b_slices])
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
        assert all(
            len(nr[1]) == s or (s == 1 and nr[1] == UnitRange.infinity)
            for nr, s in zip(domain, array.shape)
        )

        assert value_type is not None  # for mypy
        return cls(domain, array, value_type)

    def remap(self: _BaseNdArrayField, connectivity) -> _BaseNdArrayField:
        raise NotImplementedError()

    __call__ = None  # type: ignore[assignment]  # TODO: remap

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

    @overload
    def __getitem__(self, index: DomainSlice) -> common.Field | core_defs.ScalarT:
        """Absolute slicing with dimension names."""
        ...

    @overload
    def __getitem__(
        self, index: tuple[slice | int, ...] | slice | int
    ) -> common.Field | core_defs.ScalarT:
        """Relative slicing with ordered dimension access."""
        ...

    def __getitem__(self, index: FieldSlice) -> common.Field | core_defs.ScalarT:
        if not isinstance(index, tuple) and not is_domain_slice(index):
            index = (index,)

        if is_domain_slice(index):
            return self._getitem_absolute_slice(index)

        if all(isinstance(idx, (slice, int)) for idx in index):
            return self._getitem_relative_slice(index)

        raise IndexError(f"Unsupported index type: {index}")

    restrict = __getitem__

    def _getitem_absolute_slice(self, index: DomainSlice) -> common.Field | core_defs.ScalarT:
        slices = _get_slices_from_domain_slice(self.domain, index)
        new_ranges = []
        new_dims = []
        new = self.ndarray[slices]

        for i, dim in enumerate(self.domain.dims):
            if (pos := _find_index_of_dim(dim, index)) is not None:
                index_or_range = index[pos][1]
                if isinstance(index_or_range, UnitRange):
                    new_ranges.append(index_or_range)
                    new_dims.append(dim)
            else:
                # dimension not mentioned in slice
                new_ranges.append(self.domain.ranges[i])
                new_dims.append(dim)

        new_domain = Domain(dims=tuple(new_dims), ranges=tuple(new_ranges))

        return new if new.ndim == 0 else common.field(new, domain=new_domain)

    def _getitem_relative_slice(
        self, index: tuple[slice | int, ...]
    ) -> common.Field | core_defs.ScalarT:
        new = self.ndarray[index]
        new_dims = []
        new_ranges = []

        dim_diff = len(new.shape) - len(index)

        if dim_diff > 0:
            new_index = tuple([*index, Ellipsis])
        elif dim_diff == 0:
            new_index = index
        else:
            new_index = tuple(idx for idx, value in enumerate(index) if value == slice(None))

        for i, elem in enumerate(new_index):
            if isinstance(elem, slice):
                new_dims.append(self.domain.dims[i])
                new_ranges.append(_slice_range(self.domain.ranges[i], elem))
            elif isinstance(elem, int):
                new_dims.append(self.domain.dims[elem])
                new_ranges.append(self.domain.ranges[elem])
            elif isinstance(elem, type(Ellipsis)):
                curr_len = len(new_ranges)
                rest_slice = slice(curr_len, len(new.shape))
                new_dims.extend(self.domain.dims[rest_slice])
                new_ranges.extend(self.domain.ranges[rest_slice])

        new_domain = Domain(dims=tuple(new_dims), ranges=tuple(new_ranges))

        if contains_only_one_value(new):
            return new[0]
        else:
            return common.field(new, domain=new_domain)


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


def _find_index_of_dim(dim: Dimension, domain_slice: Domain | Sequence[NamedRange | NamedIndex]) -> Optional[int]:
    for i, (d, _) in enumerate(domain_slice):
        if dim == d:
            return i
    return None


def _broadcast(field: common.Field, new_dimensions: tuple[common.Dimension, ...]) -> common.Field:
    domain_slice = []
    new_domain_dims = []
    new_domain_ranges = []
    for dim in new_dimensions:
        if (pos := _find_index_of_dim(dim, field.domain)) is not None:
            domain_slice.append(slice(None))
            new_domain_dims.append(dim)
            new_domain_ranges.append(field.domain[pos][1])
        else:
            domain_slice.append(np.newaxis)
            new_domain_dims.append(dim)
            new_domain_ranges.append(UnitRange(common.Infinity.negative(), common.Infinity.positive()))
    return common.field(
        field.ndarray[tuple(domain_slice)],
        domain=Domain(tuple(new_domain_dims), tuple(new_domain_ranges)),
    )


_BaseNdArrayField.register_builtin_func(fbuiltins.broadcast, _broadcast)


def _get_slices_from_domain_slice(
    domain: Domain, domain_slice: Domain | Sequence[NamedRange | NamedIndex | None]  # None refers to np.newaxis
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

    for pos_old, (dim, _) in enumerate(domain):
        if domain_slice is None:
            slice_indices.append(slice(None))
            continue
        if (pos := _find_index_of_dim(dim, domain_slice)) is not None:
            index_or_range = domain_slice[pos][1]
            slice_indices.append(_compute_slice(index_or_range, domain, pos_old))
        else:
            slice_indices.append(slice(None))
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
        if domain.ranges[pos] == UnitRange.infinity:
            return slice(None)
        else:
            return slice(
                rng.start - domain.ranges[pos].start,
                rng.stop - domain.ranges[pos].start,
            )
    elif isinstance(rng, int):
        return rng - domain.ranges[pos].start
    else:
        raise ValueError(f"Can only use integer or UnitRange ranges, provided type: {type(rng)}")


def _slice_range(input_range: UnitRange, slice_obj: slice) -> UnitRange:
    # handle slice(None) case
    if slice_obj == slice(None):
        return UnitRange(input_range.start, input_range.stop)

    start = (
        input_range.start if slice_obj.start is None or slice_obj.start >= 0 else input_range.stop
    ) + (slice_obj.start or 0)
    stop = (
        input_range.start if slice_obj.stop is None or slice_obj.stop >= 0 else input_range.stop
    ) + (slice_obj.stop or len(input_range))

    return UnitRange(start, stop)
