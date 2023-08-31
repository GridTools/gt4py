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
from typing import Callable

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.common import Infinity
from gt4py.next.embedded import common as embedded_common, nd_array_field as nd
from gt4py.next.embedded.nd_array_field import _get_slices_from_domain_slice

_EMPTY_DOMAIN = common.Domain((), ())

def cf_operand_adapter(method: Callable):
    def wrapper(self, other):
        if isinstance(other, (ConstantField, FunctionField, nd._BaseNdArrayField)):
            new = other
        elif isinstance(other, (int, float, complex)):
            new = ConstantField(other)
        return method(self, new)

    return wrapper


@dataclasses.dataclass(frozen=True)
class ConstantField(common.Field[common.DimsT, core_defs.ScalarT]):
    value: core_defs.ScalarT
    _domain: common.Domain = _EMPTY_DOMAIN

    def _has_empty_domain(self) -> bool:
        return len(self._domain) < 1

    def restrict(self, index: common.FieldSlice) -> common.Field | core_defs.ScalarT:
        if self._has_empty_domain():
            raise IndexError("Cannot slice ConstantField without a Domain.")
        new_domain = embedded_common.sub_domain(self.domain, index)
        return self.__class__(self.value, new_domain)

    __getitem__ = restrict

    @property
    def domain(self) -> common.Domain:
        return self._domain

    @property
    def dtype(self) -> core_defs.DType[core_defs.ScalarT]:
        return core_defs.dtype(type(self.value))

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        if self._has_empty_domain():
            raise ValueError("Cannot get ndarray for ConstantField without Domain.")

        shape = []
        for _, rng in self.domain:
            if Infinity.positive() in (abs(rng.start), abs(rng.stop)):
                shape.append(1)
            else:
                shape.append(len(rng))

        return np.full(tuple(shape), self.value)

    def _binary_op_wrapper(self, other: ConstantField | nd._BaseNdArrayField, op: Callable):
        if isinstance(other, nd._BaseNdArrayField):
            if self._has_empty_domain():
                return self._handle_empty_domain_op(other, op)
            else:
                return self._handle_non_empty_domain_op(other, op)
        elif isinstance(other, self.__class__):
            return self._handle_identity_op(other, op)
        else:
            raise ValueError(f"Unsupported type in binary operation between {self.__class__} and {other.__class__}")

    def _handle_empty_domain_op(self, other: nd._BaseNdArrayField, op: Callable) -> nd._BaseNdArrayField:
        self_broadcasted = self._broadcast(other.domain.dims)
        new_data = op(self_broadcasted.ndarray, other.ndarray)
        return other.__class__.from_array(new_data, domain=other.domain)

    def _handle_non_empty_domain_op(self, other: nd._BaseNdArrayField, op: Callable) -> nd._BaseNdArrayField:
        domain_intersection = self.domain & other.domain
        self_broadcasted = self._broadcast(domain_intersection.dims)
        other_broadcasted = nd._broadcast(other, domain_intersection.dims)
        other_slices = _get_slices_from_domain_slice(other_broadcasted.domain, domain_intersection)
        new_data = op(self_broadcasted.ndarray, other_broadcasted.ndarray[other_slices])
        return other.__class__.from_array(new_data, domain=domain_intersection)

    def _handle_identity_op(self, other: ConstantField, op: Callable) -> ConstantField:
        return self.__class__(op(self.value, other.value))

    def _broadcast(self, new_dimensions: tuple[common.Dimension, ...]) -> ConstantField:
        new_domain_dims, new_domain_ranges, _ = embedded_common._compute_new_domain_info(
            self, new_dimensions
        )
        return self.__class__(
            self.value,
            common.Domain(tuple(new_domain_dims), tuple(new_domain_ranges)),
        )

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

    def remap(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

# TODO: Cleanup
@dataclasses.dataclass(frozen=True)
class FunctionField:
    func: Callable
    _domain: common.Domain = _EMPTY_DOMAIN

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        if self._has_empty_domain():
            raise ValueError("Cannot get ndarray for FunctionField without Domain.")

        shape = []
        for _, rng in self.domain:
            if Infinity.positive() in (abs(rng.start), abs(rng.stop)):
                shape.append(1)
            else:
                shape.append(len(rng))

        values = np.fromfunction(lambda *indices: self.func(*indices), shape)
        return values

    def _has_empty_domain(self) -> bool:
        return len(self._domain) < 1

    def restrict(self, index: common.FieldSlice) -> common.Field | core_defs.ScalarT:
        if self._has_empty_domain():
            raise IndexError("Cannot slice ConstantField without a Domain.")
        new_domain = embedded_common.sub_domain(self.domain, index)
        return self.__class__(self.func, new_domain)

    __getitem__ = restrict

    @property
    def domain(self) -> common.Domain:
        return self._domain

    @property
    def dtype(self) -> core_defs.DType[core_defs.ScalarT]:
        return core_defs.dtype(type(self.ndarray))

    def _binary_op_wrapper(self, other: FunctionField | nd._BaseNdArrayField, op: Callable):
        if isinstance(other, nd._BaseNdArrayField):
            if self._has_empty_domain():
                return self._handle_empty_domain_op(other, op)
            else:
                return self._handle_non_empty_domain_op(other, op)
        elif isinstance(other, self.__class__):
            return self._handle_identity_op(other, op)  # TODO
        else:
            raise ValueError(f"Unsupported type in binary operation between {self.__class__} and {other.__class__}")

    def _handle_empty_domain_op(self, other: nd._BaseNdArrayField, op: Callable) -> nd._BaseNdArrayField:
        self_broadcasted = self._broadcast(other.domain.dims)
        new_data = op(self_broadcasted.ndarray, other.ndarray)
        return other.__class__.from_array(new_data, domain=other.domain)

    def _handle_non_empty_domain_op(self, other: nd._BaseNdArrayField, op: Callable) -> nd._BaseNdArrayField:
        domain_intersection = self.domain & other.domain
        self_broadcasted = self._broadcast(domain_intersection.dims)
        other_broadcasted = nd._broadcast(other, domain_intersection.dims)
        other_slices = _get_slices_from_domain_slice(other_broadcasted.domain, domain_intersection)
        new_data = op(self_broadcasted.ndarray, other_broadcasted.ndarray[other_slices])
        return other.__class__.from_array(new_data, domain=domain_intersection)

    # TODO
    # def _handle_identity_op(self, other: FunctionField, op: Callable) -> FunctionField:
    #     new_func = op(other, self.func)
    #     return self.__class__(new_func, self.domain)

    def _broadcast(self, new_dimensions: tuple[common.Dimension, ...]) -> ConstantField:
        new_domain_dims, new_domain_ranges, _ = embedded_common._compute_new_domain_info(
            self, new_dimensions
        )
        return self.__class__(
            self.func,
            common.Domain(tuple(new_domain_dims), tuple(new_domain_ranges)),
        )

    # TODO
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
        return self._binary_op_wrapper(other, lambda x, y: x ** y)

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

    def _compose(self, operation):
        return lambda *args, **kwargs: operation(self.func(*args, **kwargs))

    def __abs__(self):
        new_func = self._compose(abs)
        return self.__class__(new_func, self.domain)

    def __neg__(self):
        new_func = self._compose(lambda x: -x)
        return self.__class__(new_func, self.domain)

    def remap(self, *args, **kwargs):
        raise NotImplementedError()
