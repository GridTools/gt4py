from __future__ import annotations

import dataclasses
from typing import Callable, Optional, ClassVar, Sequence

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.common import Infinity
from gt4py.next.embedded import nd_array_field
from gt4py.next.ffront import fbuiltins
from gt4py.next.embedded.nd_array_field import _R, _P, _get_slices_from_domain_slice, _broadcast


def cf_operand_adapter(method):
    def wrapper(self, other):
        if isinstance(other, common.Field):
            new = other
        elif isinstance(other, (int, float, complex)):
            new = ConstantField(other)
        return method(self, new)

    return wrapper


def _constant_field_broadcast(cf: ConstantField, new_dimensions: tuple[common.Dimension, ...]) -> ConstantField:
    domain_slice: list[slice | None] = []
    new_domain_dims = []
    new_domain_ranges = []
    for dim in new_dimensions:
        if (pos := nd_array_field._find_index_of_dim(dim, cf.domain)) is not None:
            domain_slice.append(slice(None))
            new_domain_dims.append(dim)
            new_domain_ranges.append(cf.domain[pos][1])
        else:
            domain_slice.append(np.newaxis)
            new_domain_dims.append(dim)
            new_domain_ranges.append(common.UnitRange(common.Infinity.negative(), common.Infinity.positive()))
    return ConstantField(
        cf.value,
        common.Domain(tuple(new_domain_dims), tuple(new_domain_ranges)),
    )


@dataclasses.dataclass(frozen=True)
class ConstantField(common.FieldABC[common.DimsT, core_defs.ScalarT]):
    value: core_defs.ScalarT
    _domain: Optional[common.Domain] = dataclasses.field(default=common.Domain((),()))

    _builtin_func_map: ClassVar[dict[fbuiltins.BuiltInFunction, Callable]] = {fbuiltins.broadcast, _constant_field_broadcast}

    @classmethod
    def __gt_builtin_func__(cls, func: fbuiltins.BuiltInFunction[_R, _P], /) -> Callable[_P, _R]:
        return cls._builtin_func_map.get(func, NotImplemented)

    def remap(self, index_field: common.Field) -> common.Field:
        raise NotImplementedError()

    def __getitem__(
        self, index: common.Domain | Sequence[common.NamedRange]
    ) -> "ConstantField" | core_defs.ScalarT:
        if len(self._domain) < 1:
            raise IndexError("Cannot slice ConstantField without domain.")

        # TODO: Implement slicing when domain is not None. Should be the same as done in Field.

        return self.value

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
        return self

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

                other_broadcasted = _broadcast(other, other.domain.dims)
                other_slices = _get_slices_from_domain_slice(other_broadcasted.domain, domain_intersection)

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
