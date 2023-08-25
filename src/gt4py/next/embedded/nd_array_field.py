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
import itertools
from collections.abc import Callable, Sequence
from types import EllipsisType, ModuleType
from typing import Any, ClassVar, Optional, ParamSpec, TypeAlias, TypeVar, cast, overload

import numpy as np
from numpy import typing as npt

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.ffront import fbuiltins


try:
    import cupy as cp
except ImportError:
    cp: Optional[ModuleType] = None  # type:ignore[no-redef]

try:
    from jax import numpy as jnp
except ImportError:
    jnp: Optional[ModuleType] = None  # type:ignore[no-redef]


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
        if hasattr(b, "__gt_builtin_func__"):  # common.is_field(b):
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
            assert isinstance(b, core_defs.SCALAR_TYPES)
            new_data = op(a.ndarray, b)

        return a.__class__.from_array(new_data, domain=a.domain)

    _builtin_binary_op.__name__ = builtin_name
    return _builtin_binary_op


_Value: TypeAlias = common.Field | core_defs.ScalarT
_P = ParamSpec("_P")
_R = TypeVar("_R", _Value, tuple[_Value, ...])


@dataclasses.dataclass(frozen=True)
class _BaseNdArrayField(common.MutableField[common.DimsT, core_defs.ScalarT]):
    """
    Shared field implementation for NumPy-like fields.

    Builtin function implementations are registered in a dictionary.
    Note: Currently, all concrete NdArray-implementations share
    the same implementation, dispatching is handled inside of the registered
    function via its namespace.
    """

    _domain: common.Domain
    _ndarray: core_defs.NDArrayObject

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
    def __gt_dims__(self) -> tuple[common.Dimension, ...]:
        return self._domain.dims

    @property
    def __gt_origin__(self) -> tuple[int, ...]:
        return tuple(-r.start for _, r in self._domain)

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        return self._ndarray

    def __array__(self, dtype: npt.DTypeLike = None) -> np.ndarray:
        return np.asarray(self._ndarray, dtype)

    @property
    def dtype(self):
        return core_defs.dtype(self._ndarray.dtype.type)

    @classmethod
    def from_array(
        cls,
        data: npt.ArrayLike
        | core_defs.NDArrayObject,  # TODO: NDArrayObject should be part of ArrayLike
        /,
        *,
        domain: common.Domain,
        dtype_like: Optional[core_defs.DType] = None,  # TODO define DTypeLike
    ) -> _BaseNdArrayField:
        xp = cls.array_ns

        xp_dtype = None if dtype_like is None else xp.dtype(core_defs.dtype(dtype_like).scalar_type)
        array = xp.asarray(data, dtype=xp_dtype)

        if dtype_like is not None:
            assert array.dtype.type == core_defs.dtype(dtype_like).scalar_type

        assert issubclass(array.dtype.type, core_defs.SCALAR_TYPES)

        assert all(isinstance(d, common.Dimension) for d in domain.dims), domain
        assert len(domain) == array.ndim
        assert all(
            len(r) == s or (s == 1 and r == common.UnitRange.infinity())
            for r, s in zip(domain.ranges, array.shape)
        )

        return cls(domain, array)

    def remap(self: _BaseNdArrayField, connectivity) -> _BaseNdArrayField:
        raise NotImplementedError()

    def __getitem__(self, index: common.FieldSlice) -> common.Field | core_defs.ScalarT:
        index = _tuplize_field_slice(index)

        if common.is_domain_slice(index):
            return self._getitem_absolute_slice(index)

        assert isinstance(index, tuple)
        if all(
            isinstance(idx, slice) or common.is_int_index(idx) or idx is Ellipsis for idx in index
        ):
            return self._getitem_relative_slice(index)

        raise IndexError(f"Unsupported index type: {index}")

    restrict = (
        __getitem__  # type:ignore[assignment] # TODO(havogt) I don't see the problem that mypy has
    )

    __call__ = None  # type: ignore[assignment]  # TODO: remap

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

    def _getitem_absolute_slice(
        self, index: common.DomainSlice
    ) -> common.Field | core_defs.ScalarT:
        slices = _get_slices_from_domain_slice(self.domain, index)
        new_ranges = []
        new_dims = []
        new = self.ndarray[slices]

        for i, dim in enumerate(self.domain.dims):
            if (pos := _find_index_of_dim(dim, index)) is not None:
                index_or_range = index[pos][1]
                if isinstance(index_or_range, common.UnitRange):
                    new_ranges.append(index_or_range)
                    new_dims.append(dim)
            else:
                # dimension not mentioned in slice
                new_ranges.append(self.domain.ranges[i])
                new_dims.append(dim)

        new_domain = common.Domain(dims=tuple(new_dims), ranges=tuple(new_ranges))

        if len(new_domain) == 0:
            assert core_defs.is_scalar_type(new)
            return new  # type: ignore[return-value] # I don't think we can express that we return `ScalarT` here
        else:
            return self.__class__.from_array(new, domain=new_domain)

    def _getitem_relative_slice(
        self, indices: tuple[slice | common.IntIndex | EllipsisType, ...]
    ) -> common.Field | core_defs.ScalarT:
        new = self.ndarray[indices]
        new_dims = []
        new_ranges = []

        for (dim, rng), idx in itertools.zip_longest(  # type: ignore[misc] # "slice" object is not iterable, not sure which slice...
            self.domain, _expand_ellipsis(indices, len(self.domain)), fillvalue=slice(None)
        ):
            if isinstance(idx, slice):
                new_dims.append(dim)
                new_ranges.append(_slice_range(rng, idx))
            else:
                assert common.is_int_index(idx)  # not in new_domain

        new_domain = common.Domain(dims=tuple(new_dims), ranges=tuple(new_ranges))

        if len(new_domain) == 0:
            assert core_defs.is_scalar_type(new), new
            return new  # type: ignore[return-value] # I don't think we can express that we return `ScalarT` here
        else:
            return self.__class__.from_array(new, domain=new_domain)


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


def _np_cp_setitem(
    self,
    index: common.FieldSlice,
    value: common.Field | core_defs.NDArrayObject | core_defs.ScalarT,
) -> None:
    index = _tuplize_field_slice(index)
    if common.is_field(value):
        # TODO(havogt): in case of `is_field(value)` we should additionally check that `value.domain == self[slice].domain`
        value = value.ndarray

    if common.is_domain_slice(index):
        slices = _get_slices_from_domain_slice(self.domain, index)
        self.ndarray[slices] = value
        return

    assert isinstance(index, tuple)
    if all(isinstance(idx, slice) or common.is_int_index(idx) or idx is Ellipsis for idx in index):
        self.ndarray[index] = value
        return

    raise IndexError(f"Unsupported index type: {index}")


# -- Concrete array implementations --
# NumPy
_nd_array_implementations = [np]


@dataclasses.dataclass(frozen=True)
class NumPyArrayField(_BaseNdArrayField):
    array_ns: ClassVar[ModuleType] = np

    __setitem__ = _np_cp_setitem


common.field.register(np.ndarray, NumPyArrayField.from_array)

# CuPy
if cp:
    _nd_array_implementations.append(cp)

    @dataclasses.dataclass(frozen=True)
    class CuPyArrayField(_BaseNdArrayField):
        array_ns: ClassVar[ModuleType] = cp

        __setitem__ = _np_cp_setitem

    common.field.register(cp.ndarray, CuPyArrayField.from_array)

# JAX
if jnp:
    _nd_array_implementations.append(jnp)

    @dataclasses.dataclass(frozen=True)
    class JaxArrayField(_BaseNdArrayField):
        array_ns: ClassVar[ModuleType] = jnp

        def __setitem__(
            self,
            index: common.FieldSlice,
            value: common.Field | core_defs.NDArrayObject | core_defs.ScalarT,
        ) -> None:
            # use `self.ndarray.at(index).set(value)`
            raise NotImplementedError("`__setitem__` for JaxArrayField not yet implemented.")

    common.field.register(jnp.ndarray, JaxArrayField.from_array)


def _tuplize_field_slice(v: common.FieldSlice) -> common.FieldSlice:
    if (
        not isinstance(v, tuple)
        and not common.is_domain_slice(v)
        or common.is_named_index(v)
        or common.is_named_range(v)
    ):
        return cast(common.FieldSlice, (v,))
    return v


def _find_index_of_dim(
    dim: common.Dimension,
    domain_slice: common.Domain | Sequence[common.NamedRange | common.NamedIndex | Any],
) -> Optional[int]:
    for i, (d, _) in enumerate(domain_slice):
        if dim == d:
            return i
    return None


def _broadcast(field: common.Field, new_dimensions: tuple[common.Dimension, ...]) -> common.Field:
    domain_slice: list[slice | None] = []
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
            new_domain_ranges.append(
                common.UnitRange(common.Infinity.negative(), common.Infinity.positive())
            )
    return common.field(
        field.ndarray[tuple(domain_slice)],
        domain=common.Domain(tuple(new_domain_dims), tuple(new_domain_ranges)),
    )


def _builtins_broadcast(
    field: common.Field | core_defs.Scalar, new_dimensions: tuple[common.Dimension, ...]
) -> common.Field:  # separated for typing reasons
    if common.is_field(field):
        return _broadcast(field, new_dimensions)
    raise AssertionError("Scalar case not reachable from `fbuiltins.broadcast`.")


_BaseNdArrayField.register_builtin_func(fbuiltins.broadcast, _builtins_broadcast)


def _get_slices_from_domain_slice(
    domain: common.Domain,
    domain_slice: common.Domain | Sequence[common.NamedRange | common.NamedIndex | Any],
) -> tuple[slice | common.IntIndex | None, ...]:
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
    slice_indices: list[slice | common.IntIndex | None] = []

    for pos_old, (dim, _) in enumerate(domain):
        if (pos := _find_index_of_dim(dim, domain_slice)) is not None:
            index_or_range = domain_slice[pos][1]
            slice_indices.append(_compute_slice(index_or_range, domain, pos_old))
        else:
            slice_indices.append(slice(None))
    return tuple(slice_indices)


def _compute_slice(
    rng: common.DomainRange, domain: common.Domain, pos: int
) -> slice | common.IntIndex:
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
        if domain.ranges[pos] == common.UnitRange.infinity():
            return slice(None)
        else:
            return slice(
                rng.start - domain.ranges[pos].start,
                rng.stop - domain.ranges[pos].start,
            )
    elif common.is_int_index(rng):
        return rng - domain.ranges[pos].start
    else:
        raise ValueError(f"Can only use integer or UnitRange ranges, provided type: {type(rng)}")


def _slice_range(input_range: common.UnitRange, slice_obj: slice) -> common.UnitRange:
    # handle slice(None) case
    if slice_obj == slice(None):
        return common.UnitRange(input_range.start, input_range.stop)

    start = (
        input_range.start if slice_obj.start is None or slice_obj.start >= 0 else input_range.stop
    ) + (slice_obj.start or 0)
    stop = (
        input_range.start if slice_obj.stop is None or slice_obj.stop >= 0 else input_range.stop
    ) + (slice_obj.stop or len(input_range))

    return common.UnitRange(start, stop)


def _expand_ellipsis(
    indices: tuple[common.IntIndex | slice | EllipsisType, ...], target_size: int
) -> tuple[common.IntIndex | slice, ...]:
    expanded_indices: list[common.IntIndex | slice] = []
    for idx in indices:
        if idx is Ellipsis:
            expanded_indices.extend([slice(None)] * (target_size - (len(indices) - 1)))
        else:
            expanded_indices.append(idx)
    return tuple(expanded_indices)


jnp.asarray([1, 2, 3]).__setitem__
