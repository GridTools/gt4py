from __future__ import annotations

import dataclasses
from typing import Callable, Optional, ClassVar, Sequence

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded.nd_array_field import _find_index_of_dim, _R, _P
from gt4py.next.ffront import fbuiltins


def _constant_field_op(method):
    def wrapper(self, other):
        new = ConstantField(other) if not isinstance(other, ConstantField) else other
        return method(self, new)

    return wrapper


def _constant_field_broadcast(field_to_broadcast: common.Field, new_dimensions: tuple[common.Dimension, ...]) -> ConstantField:
    domain_slice: list[slice | None] = []
    new_domain_dims = []
    new_domain_ranges = []
    for dim in new_dimensions:
        if (pos := _find_index_of_dim(dim, field_to_broadcast.domain)) is not None:
            domain_slice.append(slice(None))
            new_domain_dims.append(dim)
            new_domain_ranges.append(field_to_broadcast.domain[pos][1])
        else:
            domain_slice.append(np.newaxis)
            new_domain_dims.append(dim)
            new_domain_ranges.append(common.UnitRange(common.Infinity.negative(), common.Infinity.positive()))
    return ConstantField(
        field_to_broadcast.value,
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

        # TODO: Implement slicing when domain is not None.

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
        if self._domain is None:
            return None

        shape = [len(rng) for _, rng in self.domain]
        return np.full(tuple(shape), self.value)

    restrict = __getitem__

    def __call__(self, *args, **kwargs) -> common.Field:
        return self

    def _binary_op_wrapper(self, other: ConstantField | common.Field, op: Callable):
        # TODO: binary operations between Constant Field and Field
        if type(other) == common.Field:
            if not self.domain:
                domain_intersection = self.domain & other.domain
                self_broadcasted = _constant_field_broadcast(self, domain_intersection.dims)
                other_broadcasted = _constant_field_broadcast(other, domain_intersection.dims)

        return self.__class__(op(self.value, other.value))

    @_constant_field_op
    def __add__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: x + y)

    @_constant_field_op
    def __sub__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: x - y)

    @_constant_field_op
    def __mul__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: x * y)

    @_constant_field_op
    def __truediv__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: x / y)

    @_constant_field_op
    def __floordiv__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: x // y)

    @_constant_field_op
    def __rfloordiv__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: y // x)

    @_constant_field_op
    def __pow__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: x**y)

    @_constant_field_op
    def __rtruediv__(self, other: ConstantField):
        return self._binary_op_wrapper(other, lambda x, y: y / x)

    @_constant_field_op
    def __radd__(self, other):
        return self._binary_op_wrapper(other, lambda x, y: y + x)

    @_constant_field_op
    def __rmul__(self, other):
        return self._binary_op_wrapper(other, lambda x, y: y * x)

    @_constant_field_op
    def __rsub__(self, other):
        return self._binary_op_wrapper(other, lambda x, y: y - x)

    def __abs__(self):
        return self.__class__(abs(self.value))

    def __neg__(self):
        return self.__class__(-self.value)
