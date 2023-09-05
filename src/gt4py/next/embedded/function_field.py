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
import operator
from typing import Any, Callable, TypeGuard, overload

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import (
    common as embedded_common,
    exceptions as embedded_exceptions,
    nd_array_field as nd,
)
from gt4py.next.ffront import fbuiltins


@dataclasses.dataclass(frozen=True)
class FunctionField(common.FieldBuiltinFuncRegistry):
    func: Callable
    domain: common.Domain = common.Domain()
    _constant: bool = False

    def restrict(self, index: common.AnyIndexSpec) -> FunctionField:
        if _has_empty_domain(self):
            raise embedded_exceptions.EmptyDomainIndexError(self.__class__.__name__)
        new_domain = embedded_common.sub_domain(self.domain, index)
        return self.__class__(self.func, new_domain)

    __getitem__ = restrict

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        if _has_empty_domain(self):
            raise embedded_exceptions.InvalidDomainForNdarrayError(self.__class__.__name__)

        if not self.domain.is_finite():
            embedded_exceptions.InfiniteRangeNdarrayError(self.__class__.__name__, self.domain)

        shape = [len(rng) for rng in self.domain.ranges]

        if self._constant:
            return np.full(shape, self.func())

        return np.fromfunction(lambda *indices: self.func(*indices), shape)

    def _handle_function_field_op(
        self, other: FunctionField, operator_func: Callable
    ) -> FunctionField:
        domain_intersection = self.domain & other.domain
        broadcasted_self = _broadcast(self, domain_intersection.dims)
        broadcasted_other = _broadcast(other, domain_intersection.dims)
        return self.__class__(
            _compose(operator_func, broadcasted_self, broadcasted_other), domain_intersection
        )

    @overload
    def _binary_operation(self, op: Callable, other: common.Field) -> common.Field:
        ...

    @overload
    def _binary_operation(self, op: Callable, other: FunctionField) -> FunctionField:
        ...

    def _binary_operation(self, op, other):
        if isinstance(other, self.__class__):
            return self._handle_function_field_op(other, op)
        else:
            return op(other, self)

    def __add__(self, other: common.Field | FunctionField) -> common.Field | FunctionField:
        return self._binary_operation(operator.add, other)

    def __sub__(self, other: common.Field | FunctionField) -> common.Field | FunctionField:
        return self._binary_operation(operator.sub, other)

    def __mul__(self, other: common.Field | FunctionField) -> common.Field | FunctionField:
        return self._binary_operation(operator.mul, other)

    def __truediv__(self, other: common.Field | FunctionField) -> common.Field | FunctionField:
        return self._binary_operation(operator.truediv, other)

    def __floordiv__(self, other: common.Field | FunctionField) -> common.Field | FunctionField:
        return self._binary_operation(operator.floordiv, other)

    def __mod__(self, other: common.Field | FunctionField) -> common.Field | FunctionField:
        return self._binary_operation(operator.mod, other)

    def __pow__(self, other: common.Field | FunctionField) -> common.Field | FunctionField:
        return self._binary_operation(operator.pow, other)

    def __lt__(self, other: common.Field | FunctionField) -> common.Field | FunctionField:
        return self._binary_operation(operator.lt, other)

    def __le__(self, other: common.Field | FunctionField) -> common.Field | FunctionField:
        return self._binary_operation(operator.le, other)

    def __gt__(self, other: common.Field | FunctionField) -> common.Field | FunctionField:
        return self._binary_operation(operator.gt, other)

    def __ge__(self, other: common.Field | FunctionField) -> common.Field | FunctionField:
        return self._binary_operation(operator.ge, other)

    def __pos__(self) -> FunctionField:
        return self.__class__(_compose(operator.pos, self), self.domain)

    def __neg__(self) -> FunctionField:
        return self.__class__(_compose(operator.neg, self), self.domain)

    def __invert__(self) -> FunctionField:
        return self.__class__(_compose(operator.invert, self), self.domain)

    def __abs__(self) -> FunctionField:
        return self.__class__(_compose(abs, self), self.domain)


def _compose(operation: Callable, *fields: FunctionField) -> Callable:
    return lambda *args: operation(*[f.func(*args) for f in fields])


def _broadcast(field: FunctionField, dims: tuple[common.Dimension, ...]) -> FunctionField:
    def broadcasted_func(*args: int):
        if not _has_empty_domain(field):
            selected_args = [args[i] for i, dim in enumerate(dims) if dim in field.domain.dims]
            return field.func(*selected_args)
        return field.func(*args)

    named_ranges = embedded_common._compute_named_ranges(field, dims)
    return FunctionField(broadcasted_func, common.Domain(*named_ranges))


def _is_nd_array(other: Any) -> TypeGuard[nd._BaseNdArrayField]:
    return isinstance(other, nd._BaseNdArrayField)


def _has_empty_domain(field: FunctionField) -> bool:
    return len(field.domain) < 1


def constant_field(
    value: core_defs.ScalarT, domain: common.Domain = common.Domain()
) -> FunctionField:
    return FunctionField(lambda: value, domain, _constant=True)


FunctionField.register_builtin_func(fbuiltins.broadcast, _broadcast)
