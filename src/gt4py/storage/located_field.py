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

from numbers import Integral
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, overload

import numpy as np
import numpy.typing as npt

from gt4py.eve import extended_typing as xtyping

from . import utils
from .typing import (
    ArrayIndex,
    ArrayIndexOrIndices,
    ArrayLike,
    DimensionIdentifier,
    FieldIndexOrIndices,
    IntIndex,
    LocatedField,
    MutableLocatedField,
)


def is_int_index(p: Any) -> xtyping.TypeGuard[IntIndex]:
    return isinstance(p, Integral)


class LocatedFieldImpl(MutableLocatedField):
    """A Field with named dimensions/axes."""

    @property
    def __gt_dims__(self) -> Tuple[DimensionIdentifier, ...]:
        return self._axes

    def __init__(
        self,
        getter: Callable[[FieldIndexOrIndices], Any],
        axes: Tuple[DimensionIdentifier, ...],
        dtype,
        *,
        setter: Callable[[FieldIndexOrIndices, Any], None],
        array: ArrayLike,
        origin: Optional[Tuple[IntIndex, ...]] = None,
    ):
        self._getter = getter
        self._axes = axes
        self._setter = setter
        self._array = array
        self.dtype = dtype

        if origin is not None:
            self.__gt_origin__ = origin

    def __getattr__(self, item):
        return getattr(self._array, item)

    @property
    def device(self):
        if hasattr(self, "__cuda_array_interface__"):
            return "gpu"
        elif hasattr(self, "__array_interface__"):
            return "cpu"
        return None

    def __getitem__(self, indices: ArrayIndexOrIndices) -> Any:
        return self.array[indices]

    # TODO in a stable implementation of the Field concept we should make this behavior the default behavior for __getitem__
    def field_getitem(self, indices: FieldIndexOrIndices) -> Any:
        indices = utils.tupelize(indices)
        return self._getter(indices)

    def __setitem__(self, indices: ArrayIndexOrIndices, value: Any):
        self.array[indices] = value

    def field_setitem(self, indices: FieldIndexOrIndices, value: Any):
        self._setter(indices, value)

    @property
    def array(self):
        return self._array

    def __array__(self) -> np.ndarray:
        return self.array

    def __descriptor__(self):
        import dace

        res = dace.data.create_datadescriptor(self.array)
        if hasattr(self, "__gt_origin__"):
            res.__gt_origin__ = self.__gt_origin__
        res.storage = (
            dace.StorageType.GPU_Global if self.device == "gpu" else dace.StorageType.CPU_Heap
        )
        return res

    @property
    def shape(self):
        if self.array is None:
            raise TypeError("`shape` not supported for this field")
        return self.array.shape


@overload
def _shift_range(range_or_index: range, offset: int) -> slice:
    ...


@overload
def _shift_range(range_or_index: IntIndex, offset: int) -> IntIndex:
    ...


def _shift_range(range_or_index: Union[range, IntIndex], offset: int) -> ArrayIndex:
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


def _range2slice(r: Union[range, IntIndex]) -> Union[slice, IntIndex]:
    if isinstance(r, range):
        assert r.start >= 0 and r.stop >= r.start
        return slice(r.start, r.stop)
    return r


def _shift_field_indices(
    ranges_or_indices: Tuple[Union[range, IntIndex], ...],
    offsets: Tuple[int, ...],
) -> Tuple[ArrayIndex, ...]:
    return tuple(
        _range2slice(r) if o == 0 else _shift_range(r, o)
        for r, o in zip(ranges_or_indices, offsets)
    )


def array_as_located_field(
    *axes: DimensionIdentifier,
    origin: Optional[Union[Dict[DimensionIdentifier, int], Sequence[int]]] = None,
) -> Callable[[np.ndarray], LocatedFieldImpl]:
    if origin is not None and not len(axes) == len(origin):
        raise ValueError(f"axes and origin do not match ({len(axes)}!={len(origin)})")
    if isinstance(origin, dict):
        origin = tuple(origin[ax] for ax in axes)

    def _maker(a: ArrayLike) -> LocatedFieldImpl:
        if a.ndim != len(axes):
            raise TypeError("ndarray.ndim incompatible with number of given axes")

        def setter(indices, value):
            indices = utils.tupelize(indices)
            a[_shift_field_indices(indices, origin) if origin else indices] = value

        def getter(indices):
            return a[_shift_field_indices(indices, origin) if origin else indices]

        return LocatedFieldImpl(getter, axes, dtype=a.dtype, setter=setter, array=a, origin=origin)

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
    def __gt_dims__(self) -> Tuple[DimensionIdentifier, ...]:
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
    def __gt_dims__(self) -> Tuple[()]:
        return ()


def constant_field(value: Any, dtype: Optional[npt.DTypeLike] = None) -> LocatedField:
    if dtype is None:
        dtype = xtyping.infer_type(value)
    return ConstantField(value, dtype)
