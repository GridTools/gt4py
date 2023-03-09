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

import abc
from numbers import Integral
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    TypeAlias,
    TypeGuard,
    overload,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from gt4py.eve import extended_typing as xtyping
from gt4py.next.iterator import utils
from gt4py.storage import DimensionIdentifier, StorageProtocol


IntIndex: TypeAlias = Integral

FieldIndex: TypeAlias = (
    range | slice | IntIndex
)  # A `range` FieldIndex can be negative indicating a relative position with respect to origin, not wrap-around semantics like `slice` TODO(havogt): remove slice here
FieldIndices: TypeAlias = tuple[FieldIndex, ...]
FieldIndexOrIndices: TypeAlias = FieldIndex | FieldIndices


ArrayIndex: TypeAlias = slice | IntIndex
ArrayIndexOrIndices: TypeAlias = ArrayIndex | tuple[ArrayIndex, ...]


def is_int_index(p: Any) -> TypeGuard[IntIndex]:
    return isinstance(p, Integral)


@runtime_checkable
class LocatedField(StorageProtocol, Protocol):
    """A field with named dimensions providing read access."""

    @property
    @abc.abstractmethod
    def __gt_dims__(self) -> tuple[DimensionIdentifier, ...]:
        ...

    # TODO(havogt): define generic Protocol to provide a concrete return type
    @abc.abstractmethod
    def field_getitem(self, indices: FieldIndexOrIndices) -> Any:
        ...


class MutableLocatedField(LocatedField, Protocol):
    """A LocatedField with write access."""

    # TODO(havogt): define generic Protocol to provide a concrete return type
    @abc.abstractmethod
    def field_setitem(self, indices: FieldIndexOrIndices, value: Any) -> None:
        ...


class LocatedFieldImpl(MutableLocatedField):
    """A Field with named dimensions/axes."""

    @property
    def __gt_dims__(self) -> tuple[DimensionIdentifier, ...]:
        return self._axes

    def __init__(
        self,
        getter: Callable[[FieldIndexOrIndices], Any],
        axes: tuple[DimensionIdentifier, ...],
        dtype,
        *,
        setter: Callable[[FieldIndexOrIndices, Any], None],
        array: Callable[[], npt.NDArray],
    ):
        self.getter = getter
        self._axes = axes
        self.setter = setter
        self.array = array
        self.dtype = dtype

    def __getitem__(self, indices: ArrayIndexOrIndices) -> Any:
        return self.array()[indices]

    # TODO in a stable implementation of the Field concept we should make this behavior the default behavior for __getitem__
    def field_getitem(self, indices: FieldIndexOrIndices) -> Any:
        indices = utils.tupelize(indices)
        return self.getter(indices)

    def __setitem__(self, indices: ArrayIndexOrIndices, value: Any):
        self.array()[indices] = value

    def field_setitem(self, indices: FieldIndexOrIndices, value: Any):
        self.setter(indices, value)

    def __array__(self) -> np.ndarray:
        return self.array()

    @property
    def shape(self):
        if self.array is None:
            raise TypeError("`shape` not supported for this field")
        return self.array().shape


@overload
def _shift_range(range_or_index: range, offset: int) -> slice:
    ...


@overload
def _shift_range(range_or_index: IntIndex, offset: int) -> IntIndex:
    ...


def _shift_range(range_or_index: range | IntIndex, offset: int) -> ArrayIndex:
    if isinstance(range_or_index, range):
        # range_or_index describes a range in the field
        assert range_or_index.step == 1
        return slice(range_or_index.start + offset, range_or_index.stop + offset)
    else:
        assert is_int_index(range_or_index)
        return range_or_index + offset


@overload
def _range2slice(r: range) -> slice:
    ...


@overload
def _range2slice(r: IntIndex) -> IntIndex:
    ...


def _range2slice(r: range | IntIndex) -> slice | IntIndex:
    if isinstance(r, range):
        assert r.start >= 0 and r.stop >= r.start
        return slice(r.start, r.stop)
    return r


def _shift_field_indices(
    ranges_or_indices: tuple[range | IntIndex, ...],
    offsets: tuple[int, ...],
) -> tuple[ArrayIndex, ...]:
    return tuple(
        _range2slice(r) if o == 0 else _shift_range(r, o)
        for r, o in zip(ranges_or_indices, offsets)
    )


def array_as_located_field(
    *axes: DimensionIdentifier, origin: Optional[dict[DimensionIdentifier, int]] = None
) -> Callable[[np.ndarray], LocatedFieldImpl]:
    def _maker(a: "ArrayLike") -> LocatedFieldImpl:
        from gt4py.next.iterator.embedded import get_ordered_indices

        if a.ndim != len(axes):
            raise TypeError("ndarray.ndim incompatible with number of given axes")

        if origin is not None:
            offsets = get_ordered_indices(axes, {k.value: v for k, v in origin.items()})
        else:
            offsets = None

        def setter(indices, value):
            indices = utils.tupelize(indices)
            a[_shift_field_indices(indices, offsets) if offsets else indices] = value

        def getter(indices):
            return a[_shift_field_indices(indices, offsets) if offsets else indices]

        return LocatedFieldImpl(
            getter,
            axes,
            dtype=a.dtype,
            setter=setter,
            array=a.__array__,
        )

    return _maker


class IndexField(LocatedField):
    def __init__(self, axis: DimensionIdentifier, dtype: npt.DTypeLike) -> None:
        self.axis = axis
        self.dtype = np.dtype(dtype)

    def field_getitem(self, index: FieldIndexOrIndices) -> Any:
        if isinstance(index, int):
            return self.dtype.type(index)
        else:
            assert isinstance(index, tuple) and len(index) == 1 and isinstance(index[0], int)
            return self.dtype.type(index[0])

    @property
    def __gt_dims__(self) -> tuple[DimensionIdentifier, ...]:
        return (self.axis,)


def index_field(axis: DimensionIdentifier, dtype: npt.DTypeLike = int) -> LocatedField:
    return IndexField(axis, dtype)


class ConstantField(LocatedField):
    def __init__(self, value: Any, dtype: npt.DTypeLike):
        self.value = value
        self.dtype = np.dtype(dtype).type

    def field_getitem(self, _: FieldIndexOrIndices) -> Any:
        return self.dtype(self.value)

    @property
    def __gt_dims__(self) -> tuple[()]:
        return ()


def constant_field(value: Any, dtype: Optional[npt.DTypeLike] = None) -> LocatedField:
    if dtype is None:
        dtype = xtyping.infer_type(value)
    return ConstantField(value, dtype)
