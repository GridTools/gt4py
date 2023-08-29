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
import itertools
from types import EllipsisType
from typing import Callable, ClassVar, Optional, Sequence, cast

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.common import Infinity
from gt4py.next.embedded import nd_array_field
from gt4py.next.embedded.nd_array_field import (
    _P,
    _R,
    _broadcast,
    _expand_ellipsis,
    _find_index_of_dim,
    _get_slices_from_domain_slice,
    _slice_range,
)
from gt4py.next.ffront import fbuiltins


def cf_operand_adapter(method):
    def wrapper(self, other):
        if isinstance(other, common.Field):
            new = other
        elif isinstance(other, (int, float, complex)):
            new = ConstantField(other)
        return method(self, new)

    return wrapper


def _constant_field_broadcast(
    cf: ConstantField, new_dimensions: tuple[common.Dimension, ...]
) -> ConstantField:
    domain_slice: list[slice | None] = []
    new_domain_dims = []
    new_domain_ranges = []
    for dim in new_dimensions:
        if (pos := common._find_index_of_dim(dim, cf.domain)) is not None:
            domain_slice.append(slice(None))
            new_domain_dims.append(dim)
            new_domain_ranges.append(cf.domain[pos][1])
        else:
            domain_slice.append(np.newaxis)
            new_domain_dims.append(dim)
            new_domain_ranges.append(
                common.UnitRange(common.Infinity.negative(), common.Infinity.positive())
            )
    return ConstantField(
        cf.value,
        common.Domain(tuple(new_domain_dims), tuple(new_domain_ranges)),
    )


@dataclasses.dataclass(frozen=True)
class ConstantField(common.FieldABC[common.DimsT, core_defs.ScalarT]):
    value: core_defs.ScalarT
    _domain: Optional[common.Domain] = dataclasses.field(default=common.Domain((), ()))

    _builtin_func_map: ClassVar[dict[fbuiltins.BuiltInFunction, Callable]] = {
        fbuiltins.broadcast,
        _constant_field_broadcast,
    }

    @classmethod
    def __gt_builtin_func__(cls, func: fbuiltins.BuiltInFunction[_R, _P], /) -> Callable[_P, _R]:
        return cls._builtin_func_map.get(func, NotImplemented)

    def remap(self, index_field: common.Field) -> common.Field:
        raise NotImplementedError()

    def __getitem__(
        self, index: common.Domain | Sequence[common.NamedRange]
    ) -> "ConstantField" | core_defs.ScalarT:
        if not self._domain:
            raise IndexError("Cannot slice ConstantField without domain.")

        if (
            not isinstance(index, tuple)
            and not common.is_domain_slice(index)
            or common.is_named_index(index)
            or common.is_named_range(index)
        ):
            index = cast(common.FieldSlice, (index,))

        if common.is_domain_slice(index):
            return self._getitem_absolute_slice(index)

        assert isinstance(index, tuple)
        if all(isinstance(idx, (slice, int)) or idx is Ellipsis for idx in index):
            return self._getitem_relative_slice(index)

        raise IndexError(f"Unsupported index type: {index}")

    def _getitem_absolute_slice(
        self, index: common.DomainSlice
    ) -> ConstantField | core_defs.ScalarT:
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
            return self.__class__(new, new_domain)

    def _getitem_relative_slice(
        self, indices: tuple[slice | int | EllipsisType, ...]
    ) -> ConstantField | core_defs.ScalarT:
        new = self.ndarray[indices]
        new_dims = []
        new_ranges = []

        for (dim, rng), idx in itertools.zip_longest(
            # type: ignore[misc] # "slice" object is not iterable, not sure which slice...
            self.domain,
            _expand_ellipsis(indices, len(self.domain)),
            fillvalue=slice(None),
        ):
            if isinstance(idx, slice):
                new_dims.append(dim)
                new_ranges.append(_slice_range(rng, idx))
            else:
                assert isinstance(idx, int)  # not in new_domain

        new_domain = common.Domain(dims=tuple(new_dims), ranges=tuple(new_ranges))

        if len(new_domain) == 0:
            assert core_defs.is_scalar_type(new), new
            return new  # type: ignore[return-value] # I don't think we can express that we return `ScalarT` here
        else:
            return self.__class__(new, new_domain)

    @property
    def domain(self) -> common.Domain:
        return self._domain

    @property
    def dtype(self) -> core_defs.DType[core_defs.ScalarT]:
        return type(self.value)

    @property
    def value_type(self) -> type[core_defs.ScalarT]:
        return type(self.value)

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        if len(self._domain) < 1:
            raise ValueError("Cannot get ndarray for ConstantField without Domain.")

        shape = []

        for _, rng in self.domain:
            if Infinity.positive() in (abs(rng.start), abs(rng.stop)):
                shape.append(1)
            else:
                shape.append(len(rng))

        return np.full(tuple(shape), self.value)

    restrict = __getitem__

    def __call__(self, *args, **kwargs) -> common.Field:
        raise NotImplementedError()

    def _binary_op_wrapper(self, other: ConstantField | common.Field, op: Callable):
        if isinstance(other, nd_array_field._BaseNdArrayField):
            if len(self.domain) < 1:
                self_broadcasted = _constant_field_broadcast(self, other.domain.dims)
                broadcasted_ndarray = self_broadcasted.ndarray
                new_data = op(broadcasted_ndarray, other.ndarray)
                return other.__class__.from_array(new_data, domain=other.domain)
            else:
                domain_intersection = self.domain & other.domain
                self_broadcasted = _constant_field_broadcast(self, domain_intersection.dims)

                other_broadcasted = _broadcast(other, domain_intersection.dims)
                other_slices = _get_slices_from_domain_slice(
                    other_broadcasted.domain, domain_intersection
                )

                new_data = op(self_broadcasted.ndarray, other_broadcasted.ndarray[other_slices])
                return other.__class__.from_array(new_data, domain=domain_intersection)

        return self.__class__(op(self.value, other.value))

    @cf_operand_adapter
    def __add__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: x + y)

    @cf_operand_adapter
    def __sub__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: x - y)

    @cf_operand_adapter
    def __mul__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: x * y)

    @cf_operand_adapter
    def __truediv__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: x / y)

    @cf_operand_adapter
    def __floordiv__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: x // y)

    @cf_operand_adapter
    def __rfloordiv__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: y // x)

    @cf_operand_adapter
    def __pow__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: x**y)

    @cf_operand_adapter
    def __rtruediv__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: y / x)

    @cf_operand_adapter
    def __radd__(self, other):
        return self._binary_op_wrapper(other, lambda x, y: y + x)

    @cf_operand_adapter
    def __rmul__(self, other):
        return self._binary_op_wrapper(other, lambda x, y: y * x)

    @cf_operand_adapter
    def __rsub__(self, other):
        return self._binary_op_wrapper(other, lambda x, y: y - x)

    def __abs__(self):
        return self.__class__(abs(self.value))

    def __neg__(self):
        return self.__class__(-self.value)
