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

import abc
import collections
import dataclasses
import enum
import functools
import numbers
import types
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from gt4py._core import definitions as core_defs
from gt4py.eve.extended_typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Final,
    Generic,
    Never,
    Optional,
    ParamSpec,
    Protocol,
    Self,
    TypeAlias,
    TypeGuard,
    TypeVar,
    cast,
    extended_runtime_checkable,
    overload,
    runtime_checkable,
)
from gt4py.eve.type_definitions import StrEnum


DimT = TypeVar("DimT", bound="Dimension")  # , covariant=True)
DimsT = TypeVar("DimsT", bound=Sequence["Dimension"], covariant=True)


Tag: TypeAlias = str


@enum.unique
class DimensionKind(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    LOCAL = "local"

    def __str__(self):
        return self.value


@dataclasses.dataclass(frozen=True)
class Dimension:
    value: str
    kind: DimensionKind = dataclasses.field(default=DimensionKind.HORIZONTAL)

    def __str__(self):
        return f"{self.value}[{self.kind}]"

    def __call__(self, val: int) -> NamedIndex:
        return self, val


class Infinity(enum.Enum):
    """Describes an unbounded `UnitRange`."""

    NEGATIVE = enum.auto()
    POSITIVE = enum.auto()

    def __add__(self, _: int) -> Self:
        return self

    __radd__ = __add__

    def __sub__(self, _: int) -> Self:
        return self

    __rsub__ = __sub__

    def __le__(self, other: int | Infinity) -> bool:
        return self is self.NEGATIVE or other is self.POSITIVE

    def __lt__(self, other: int | Infinity) -> bool:
        return self is self.NEGATIVE and other is not self

    def __ge__(self, other: int | Infinity) -> bool:
        return self is self.POSITIVE or other is self.NEGATIVE

    def __gt__(self, other: int | Infinity) -> bool:
        return self is self.POSITIVE and other is not self


def _as_int(v: core_defs.IntegralScalar | Infinity) -> int | Infinity:
    return v if isinstance(v, Infinity) else int(v)


_Left = TypeVar("_Left", int, Infinity)
_Right = TypeVar("_Right", int, Infinity)


@dataclasses.dataclass(frozen=True, init=False)
class UnitRange(Sequence[int], Generic[_Left, _Right]):
    """Range from `start` to `stop` with step size one."""

    start: _Left
    stop: _Right

    def __init__(
        self, start: core_defs.IntegralScalar | Infinity, stop: core_defs.IntegralScalar | Infinity
    ) -> None:
        if start < stop:
            object.__setattr__(self, "start", _as_int(start))
            object.__setattr__(self, "stop", _as_int(stop))
        else:
            # make UnitRange(0,0) the single empty UnitRange
            object.__setattr__(self, "start", 0)
            object.__setattr__(self, "stop", 0)

    @classmethod
    def infinite(
        cls,
    ) -> UnitRange:
        return cls(Infinity.NEGATIVE, Infinity.POSITIVE)

    def __len__(self) -> int:
        if UnitRange.is_finite(self):
            return max(0, self.stop - self.start)
        raise ValueError("Cannot compute length of open 'UnitRange'.")

    @classmethod
    def is_finite(cls, obj: UnitRange) -> TypeGuard[FiniteUnitRange]:
        # classmethod since TypeGuards requires the guarded obj as separate argument
        return obj.start is not Infinity.NEGATIVE and obj.stop is not Infinity.POSITIVE

    @classmethod
    def is_right_finite(cls, obj: UnitRange) -> TypeGuard[UnitRange[_Left, int]]:
        # classmethod since TypeGuards requires the guarded obj as separate argument
        return obj.stop is not Infinity.POSITIVE

    @classmethod
    def is_left_finite(cls, obj: UnitRange) -> TypeGuard[UnitRange[int, _Right]]:
        # classmethod since TypeGuards requires the guarded obj as separate argument
        return obj.start is not Infinity.NEGATIVE

    def __repr__(self) -> str:
        return f"UnitRange({self.start}, {self.stop})"

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> UnitRange: ...

    def __getitem__(self, index: int | slice) -> int | UnitRange:
        assert UnitRange.is_finite(self)
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("'UnitRange': step required to be '1'.")
            new_start = self.start + (start or 0)
            new_stop = (self.start if stop > 0 else self.stop) + stop
            return UnitRange(new_start, new_stop)
        else:
            if index < 0:
                index += len(self)

            if 0 <= index < len(self):
                return self.start + index
            else:
                raise IndexError("'UnitRange' index out of range")

    def __and__(self, other: UnitRange) -> UnitRange:
        return UnitRange(max(self.start, other.start), min(self.stop, other.stop))

    def __contains__(self, value: Any) -> bool:
        # TODO(egparedes): use core_defs.IntegralScalar for `isinstance()` checks (see PEP 604)
        #   and remove int cast, once the related mypy bug (#16358) gets fixed
        if isinstance(value, core_defs.INTEGRAL_TYPES):
            return self.start <= cast(int, value) < self.stop
        else:
            return False

    def __le__(self, other: UnitRange) -> bool:
        return self.start >= other.start and self.stop <= other.stop

    def __lt__(self, other: UnitRange) -> bool:
        return (self.start > other.start and self.stop <= other.stop) or (
            self.start >= other.start and self.stop < other.stop
        )

    def __ge__(self, other: UnitRange) -> bool:
        return self.start <= other.start and self.stop >= other.stop

    def __gt__(self, other: UnitRange) -> bool:
        return (self.start < other.start and self.stop >= other.stop) or (
            self.start <= other.start and self.stop > other.stop
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, UnitRange):
            return self.start == other.start and self.stop == other.stop
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __add__(self, other: int) -> UnitRange:
        return UnitRange(self.start + other, self.stop + other)

    def __sub__(self, other: int) -> UnitRange:
        return UnitRange(self.start - other, self.stop - other)

    def __str__(self) -> str:
        return f"({self.start}:{self.stop})"


FiniteUnitRange: TypeAlias = UnitRange[int, int]


RangeLike: TypeAlias = (
    UnitRange
    | range
    | tuple[core_defs.IntegralScalar, core_defs.IntegralScalar]
    | core_defs.IntegralScalar
    | None
)


def unit_range(r: RangeLike) -> UnitRange:
    if isinstance(r, UnitRange):
        return r
    if isinstance(r, range):
        if r.step != 1:
            raise ValueError(f"'UnitRange' requires step size 1, got '{r.step}'.")
        return UnitRange(r.start, r.stop)
    # TODO(egparedes): use core_defs.IntegralScalar for `isinstance()` checks (see PEP 604)
    #   once the related mypy bug (#16358) gets fixed
    if (
        isinstance(r, tuple)
        and (isinstance(r[0], core_defs.INTEGRAL_TYPES) or r[0] in (None, Infinity.NEGATIVE))
        and (isinstance(r[1], core_defs.INTEGRAL_TYPES) or r[1] in (None, Infinity.POSITIVE))
    ):
        start = r[0] if r[0] is not None else Infinity.NEGATIVE
        stop = r[1] if r[1] is not None else Infinity.POSITIVE
        return UnitRange(start, stop)
    if isinstance(r, core_defs.INTEGRAL_TYPES):
        return UnitRange(0, cast(core_defs.IntegralScalar, r))
    if r is None:
        return UnitRange.infinite()
    raise ValueError(f"'{r!r}' cannot be interpreted as 'UnitRange'.")


IntIndex: TypeAlias = int | core_defs.IntegralScalar
NamedIndex: TypeAlias = tuple[Dimension, IntIndex]  # TODO: convert to NamedTuple
NamedRange: TypeAlias = tuple[Dimension, UnitRange]  # TODO: convert to NamedTuple
FiniteNamedRange: TypeAlias = tuple[Dimension, FiniteUnitRange]  # TODO: convert to NamedTuple
RelativeIndexElement: TypeAlias = IntIndex | slice | types.EllipsisType
NamedSlice: TypeAlias = slice  # once slice is generic we should do: slice[NamedIndex, NamedIndex, Literal[1]], see https://peps.python.org/pep-0696/
AbsoluteIndexElement: TypeAlias = NamedIndex | NamedRange | NamedSlice
AnyIndexElement: TypeAlias = RelativeIndexElement | AbsoluteIndexElement
AbsoluteIndexSequence: TypeAlias = Sequence[NamedRange | NamedIndex]
RelativeIndexSequence: TypeAlias = tuple[
    slice | IntIndex | types.EllipsisType, ...
]  # is a tuple but called Sequence for symmetry
AnyIndexSequence: TypeAlias = RelativeIndexSequence | AbsoluteIndexSequence
AnyIndexSpec: TypeAlias = AnyIndexElement | AnyIndexSequence


def is_int_index(p: Any) -> TypeGuard[IntIndex]:
    # should be replaced by isinstance(p, IntIndex), but mypy complains with
    # `Argument 2 to "isinstance" has incompatible type "<typing special form>"; expected "_ClassInfo"  [arg-type]`
    return isinstance(p, (int, core_defs.INTEGRAL_TYPES))


def is_named_range(v: AnyIndexSpec) -> TypeGuard[NamedRange]:
    return (
        isinstance(v, tuple)
        and len(v) == 2
        and isinstance(v[0], Dimension)
        and isinstance(v[1], UnitRange)
    )


def is_finite_named_range(v: NamedRange) -> TypeGuard[FiniteNamedRange]:
    return UnitRange.is_finite(v[1])


def is_named_index(v: AnyIndexSpec) -> TypeGuard[NamedRange]:
    return (
        isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], Dimension) and is_int_index(v[1])
    )


def is_named_slice(obj: AnyIndexSpec) -> TypeGuard[NamedRange]:
    return isinstance(obj, slice) and (is_named_index(obj.start) and is_named_index(obj.stop))


def is_any_index_element(v: AnyIndexSpec) -> TypeGuard[AnyIndexElement]:
    return (
        is_int_index(v)
        or is_named_range(v)
        or is_named_index(v)
        or isinstance(v, slice)
        or v is Ellipsis
    )


def is_absolute_index_sequence(v: AnyIndexSequence) -> TypeGuard[AbsoluteIndexSequence]:
    return isinstance(v, Sequence) and all(is_named_range(e) or is_named_index(e) for e in v)


def is_relative_index_sequence(v: AnyIndexSequence) -> TypeGuard[RelativeIndexSequence]:
    return isinstance(v, tuple) and all(
        isinstance(e, slice) or is_int_index(e) or e is Ellipsis for e in v
    )


def as_any_index_sequence(index: AnyIndexSpec) -> AnyIndexSequence:
    # `cast` because mypy/typing doesn't special case 1-element tuples, i.e. `tuple[A|B] != tuple[A]|tuple[B]`
    return cast(
        AnyIndexSequence,
        (index,) if is_any_index_element(index) else index,
    )


def named_range(v: tuple[Dimension, RangeLike]) -> NamedRange:
    return (v[0], unit_range(v[1]))


_Rng = TypeVar(
    "_Rng",
    UnitRange[int, int],
    UnitRange[Infinity, int],
    UnitRange[int, Infinity],
    UnitRange[Infinity, Infinity],
)


@dataclasses.dataclass(frozen=True, init=False)
class Domain(Sequence[tuple[Dimension, _Rng]], Generic[_Rng]):
    """Describes the `Domain` of a `Field` as a `Sequence` of `NamedRange` s."""

    dims: tuple[Dimension, ...]
    ranges: tuple[_Rng, ...]

    def __init__(
        self,
        *args: tuple[Dimension, _Rng],
        dims: Optional[Sequence[Dimension]] = None,
        ranges: Optional[Sequence[_Rng]] = None,
    ) -> None:
        if dims is not None or ranges is not None:
            if dims is None and ranges is None:
                raise ValueError("Either both none of 'dims' and 'ranges' must be specified.")
            if len(args) > 0:
                raise ValueError(
                    "No extra 'args' allowed when constructing fomr 'dims' and 'ranges'."
                )

            assert dims is not None and ranges is not None  # for mypy
            if not all(isinstance(dim, Dimension) for dim in dims):
                raise ValueError(
                    f"'dims' argument needs to be a 'tuple[Dimension, ...]', got '{dims}'."
                )
            if not all(isinstance(rng, UnitRange) for rng in ranges):
                raise ValueError(
                    f"'ranges' argument needs to be a 'tuple[UnitRange, ...]', got '{ranges}'."
                )
            if len(dims) != len(ranges):
                raise ValueError(
                    f"Number of provided dimensions ({len(dims)}) does not match number of provided ranges ({len(ranges)})."
                )

            object.__setattr__(self, "dims", tuple(dims))
            object.__setattr__(self, "ranges", tuple(ranges))
        else:
            if not all(is_named_range(arg) for arg in args):
                raise ValueError(
                    f"Elements of 'Domain' need to be instances of 'NamedRange', got '{args}'."
                )
            dims, ranges = zip(*args) if args else ((), ())
            object.__setattr__(self, "dims", tuple(dims))
            object.__setattr__(self, "ranges", tuple(ranges))

        if len(set(self.dims)) != len(self.dims):
            raise NotImplementedError(f"Domain dimensions must be unique, not '{self.dims}'.")

    def __len__(self) -> int:
        return len(self.ranges)

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(r) for r in self.ranges)

    @classmethod
    def is_finite(cls, obj: Domain) -> TypeGuard[FiniteDomain]:
        # classmethod since TypeGuards requires the guarded obj as separate argument
        return all(UnitRange.is_finite(rng) for rng in obj.ranges)

    @overload
    def __getitem__(self, index: int) -> tuple[Dimension, _Rng]: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    @overload
    def __getitem__(self, index: Dimension) -> tuple[Dimension, _Rng]: ...

    def __getitem__(self, index: int | slice | Dimension) -> NamedRange | Domain:
        if isinstance(index, int):
            return self.dims[index], self.ranges[index]
        elif isinstance(index, slice):
            dims_slice = self.dims[index]
            ranges_slice = self.ranges[index]
            return Domain(dims=dims_slice, ranges=ranges_slice)
        elif isinstance(index, Dimension):
            try:
                index_pos = self.dims.index(index)
                return self.dims[index_pos], self.ranges[index_pos]
            except ValueError as ex:
                raise KeyError(f"No Dimension of type '{index}' is present in the Domain.") from ex
        else:
            raise KeyError("Invalid index type, must be either int, slice, or Dimension.")

    def __and__(self, other: Domain) -> Domain:
        """
        Intersect `Domain`s, missing `Dimension`s are considered infinite.

        Examples:
        ---------
        >>> I = Dimension("I")
        >>> J = Dimension("J")

        >>> Domain((I, UnitRange(-1, 3))) & Domain((I, UnitRange(1, 6)))
        Domain(dims=(Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>),), ranges=(UnitRange(1, 3),))

        >>> Domain((I, UnitRange(-1, 3)), (J, UnitRange(2, 4))) & Domain((I, UnitRange(1, 6)))
        Domain(dims=(Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)), ranges=(UnitRange(1, 3), UnitRange(2, 4)))
        """
        broadcast_dims = tuple(promote_dims(self.dims, other.dims))
        intersected_ranges = tuple(
            rng1 & rng2
            for rng1, rng2 in zip(
                _broadcast_ranges(broadcast_dims, self.dims, self.ranges),
                _broadcast_ranges(broadcast_dims, other.dims, other.ranges),
            )
        )
        return Domain(dims=broadcast_dims, ranges=intersected_ranges)

    def __str__(self) -> str:
        return f"Domain({', '.join(f'{e[0]}={e[1]}' for e in self)})"

    def dim_index(self, dim: Dimension) -> Optional[int]:
        return self.dims.index(dim) if dim in self.dims else None

    def pop(self, index: int | Dimension = -1) -> Domain:
        return self.replace(index)

    def insert(self, index: int | Dimension, *named_ranges: NamedRange) -> Domain:
        if isinstance(index, int) and index == len(self.dims):
            new_dims, new_ranges = zip(*named_ranges)
            return Domain(dims=self.dims + new_dims, ranges=self.ranges + new_ranges)
        else:
            return self.replace(index, *named_ranges)

    def replace(self, index: int | Dimension, *named_ranges: NamedRange) -> Domain:
        assert all(is_named_range(nr) for nr in named_ranges)
        if isinstance(index, Dimension):
            dim_index = self.dim_index(index)
            if dim_index is None:
                raise ValueError(f"Dimension '{index}' not found in Domain.")
            index = dim_index
        if not (-len(self.dims) <= index < len(self.dims)):
            raise IndexError(
                f"Index '{index}' out of bounds for Domain of length {len(self.dims)}."
            )
        if index < 0:
            index += len(self.dims)
        new_dims, new_ranges = zip(*named_ranges) if len(named_ranges) > 0 else ((), ())
        dims = self.dims[:index] + new_dims + self.dims[index + 1 :]
        ranges = self.ranges[:index] + new_ranges + self.ranges[index + 1 :]

        return Domain(dims=dims, ranges=ranges)


FiniteDomain: TypeAlias = Domain[FiniteUnitRange]


DomainLike: TypeAlias = (
    Sequence[tuple[Dimension, RangeLike]] | Mapping[Dimension, RangeLike]
)  # `Domain` is `Sequence[NamedRange]` and therefore a subset


def domain(domain_like: DomainLike) -> Domain:
    """
    Construct `Domain` from `DomainLike` object.

    Examples:
    ---------
    >>> I = Dimension("I")
    >>> J = Dimension("J")

    >>> domain(((I, (2, 4)), (J, (3, 5))))
    Domain(dims=(Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)), ranges=(UnitRange(2, 4), UnitRange(3, 5)))

    >>> domain({I: (2, 4), J: (3, 5)})
    Domain(dims=(Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)), ranges=(UnitRange(2, 4), UnitRange(3, 5)))

    >>> domain(((I, 2), (J, 4)))
    Domain(dims=(Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)), ranges=(UnitRange(0, 2), UnitRange(0, 4)))

    >>> domain({I: 2, J: 4})
    Domain(dims=(Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)), ranges=(UnitRange(0, 2), UnitRange(0, 4)))
    """
    if isinstance(domain_like, Domain):
        return domain_like
    if isinstance(domain_like, Sequence):
        return Domain(*tuple(named_range(d) for d in domain_like))
    if isinstance(domain_like, Mapping):
        if all(isinstance(elem, core_defs.INTEGRAL_TYPES) for elem in domain_like.values()):
            return Domain(
                dims=tuple(domain_like.keys()),
                ranges=tuple(
                    UnitRange(0, s)  # type: ignore[arg-type] # type of `s` is checked in condition
                    for s in domain_like.values()
                ),
            )
        return Domain(
            dims=tuple(domain_like.keys()),
            ranges=tuple(unit_range(r) for r in domain_like.values()),
        )
    raise ValueError(f"'{domain_like}' is not 'DomainLike'.")


def _broadcast_ranges(
    broadcast_dims: Sequence[Dimension], dims: Sequence[Dimension], ranges: Sequence[UnitRange]
) -> tuple[UnitRange, ...]:
    return tuple(
        ranges[dims.index(d)] if d in dims else UnitRange.infinite() for d in broadcast_dims
    )


if TYPE_CHECKING:
    import gt4py.next.ffront.fbuiltins as fbuiltins

    _Value: TypeAlias = "Field" | core_defs.ScalarT
    _P = ParamSpec("_P")
    _R = TypeVar("_R", _Value, tuple[_Value, ...])

    class GTBuiltInFuncDispatcher(Protocol):
        def __call__(self, func: fbuiltins.BuiltInFunction[_R, _P], /) -> Callable[_P, _R]: ...


# TODO(havogt): we need to describe when this interface should be used instead of the `Field` protocol.
class GTFieldInterface(core_defs.GTDimsInterface, core_defs.GTOriginInterface, Protocol):
    """
    Protocol for object providing the `__gt_domain__` property, specifying the :class:`Domain` of a :class:`Field`.

    Note:
    - A default implementation of the `__gt_dims__` interface from `gt4py.cartesian` is provided.
    - No implementation of `__gt_origin__` is provided because of infinite fields.
    """

    @property
    def __gt_domain__(self) -> Domain:
        # TODO probably should be changed to `DomainLike` (with a new concept `DimensionLike`)
        # to allow implementations without having to import gtx.Domain.
        ...

    @property
    def __gt_dims__(self) -> tuple[str, ...]:
        return tuple(d.value for d in self.__gt_domain__.dims)


@extended_runtime_checkable
class Field(GTFieldInterface, Protocol[DimsT, core_defs.ScalarT]):
    __gt_builtin_func__: ClassVar[GTBuiltInFuncDispatcher]

    @property
    def domain(self) -> Domain: ...

    @property
    def __gt_domain__(self) -> Domain:
        return self.domain

    @property
    def codomain(self) -> type[core_defs.ScalarT] | Dimension: ...

    @property
    def dtype(self) -> core_defs.DType[core_defs.ScalarT]: ...

    @property
    def ndarray(self) -> core_defs.NDArrayObject: ...

    def __str__(self) -> str:
        return f"⟨{self.domain!s} → {self.dtype}⟩"

    @abc.abstractmethod
    def asnumpy(self) -> np.ndarray: ...

    @abc.abstractmethod
    def remap(self, index_field: ConnectivityField | fbuiltins.FieldOffset) -> Field: ...

    @abc.abstractmethod
    def restrict(self, item: AnyIndexSpec) -> Field: ...

    @abc.abstractmethod
    def as_scalar(self) -> core_defs.ScalarT: ...

    # Operators
    @abc.abstractmethod
    def __call__(self, index_field: ConnectivityField | fbuiltins.FieldOffset) -> Field: ...

    @abc.abstractmethod
    def __getitem__(self, item: AnyIndexSpec) -> Field: ...

    @abc.abstractmethod
    def __abs__(self) -> Field: ...

    @abc.abstractmethod
    def __neg__(self) -> Field: ...

    @abc.abstractmethod
    def __invert__(self) -> Field:
        """Only defined for `Field` of value type `bool`."""

    @abc.abstractmethod
    def __eq__(self, other: Any) -> Field:  # type: ignore[override] # mypy wants return `bool`
        ...

    @abc.abstractmethod
    def __ne__(self, other: Any) -> Field:  # type: ignore[override] # mypy wants return `bool`
        ...

    @abc.abstractmethod
    def __add__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __radd__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __sub__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __rsub__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __mul__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __rmul__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __floordiv__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __rfloordiv__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __truediv__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __rtruediv__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __pow__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __and__(self, other: Field | core_defs.ScalarT) -> Field:
        """Only defined for `Field` of value type `bool`."""

    @abc.abstractmethod
    def __or__(self, other: Field | core_defs.ScalarT) -> Field:
        """Only defined for `Field` of value type `bool`."""

    @abc.abstractmethod
    def __xor__(self, other: Field | core_defs.ScalarT) -> Field:
        """Only defined for `Field` of value type `bool`."""


def is_field(
    v: Any,
) -> TypeGuard[Field]:
    # This function is introduced to localize the `type: ignore` because
    # extended_runtime_checkable does not make the protocol runtime_checkable
    # for mypy.
    # TODO(egparedes): remove it when extended_runtime_checkable is fixed
    return isinstance(v, Field)  # type: ignore[misc] # we use extended_runtime_checkable


@extended_runtime_checkable
class MutableField(Field[DimsT, core_defs.ScalarT], Protocol[DimsT, core_defs.ScalarT]):
    @abc.abstractmethod
    def __setitem__(self, index: AnyIndexSpec, value: Field | core_defs.ScalarT) -> None: ...


def is_mutable_field(
    v: Field,
) -> TypeGuard[MutableField]:
    # This function is introduced to localize the `type: ignore` because
    # extended_runtime_checkable does not make the protocol runtime_checkable
    # for mypy.
    # TODO(egparedes): remove it when extended_runtime_checkable is fixed
    return isinstance(v, MutableField)  # type: ignore[misc] # we use extended_runtime_checkable


class ConnectivityKind(enum.Flag):
    MODIFY_DIMS = enum.auto()
    MODIFY_RANK = enum.auto()
    MODIFY_STRUCTURE = enum.auto()


@extended_runtime_checkable
# type: ignore[misc] # DimT should be covariant, but breaks in another place
class ConnectivityField(Field[DimsT, core_defs.IntegralScalar], Protocol[DimsT, DimT]):
    @property
    @abc.abstractmethod
    def codomain(self) -> DimT: ...

    @property
    def kind(self) -> ConnectivityKind:
        return (
            ConnectivityKind.MODIFY_DIMS
            | ConnectivityKind.MODIFY_RANK
            | ConnectivityKind.MODIFY_STRUCTURE
        )

    @abc.abstractmethod
    def inverse_image(self, image_range: UnitRange | NamedRange) -> Sequence[NamedRange]: ...

    @property
    @abc.abstractmethod
    def skip_value(self) -> Optional[core_defs.IntegralScalar]: ...

    # Operators
    def __abs__(self) -> Never:
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __neg__(self) -> Never:
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __invert__(self) -> Never:
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __eq__(self, other: Any) -> Never:
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __ne__(self, other: Any) -> Never:
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __add__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __radd__(self, other: Field | core_defs.IntegralScalar) -> Never:  # type: ignore[misc] # Forward operator not callalbe
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __sub__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __rsub__(self, other: Field | core_defs.IntegralScalar) -> Never:  # type: ignore[misc] # Forward operator not callalbe
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __mul__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __rmul__(self, other: Field | core_defs.IntegralScalar) -> Never:  # type: ignore[misc] # Forward operator not callalbe
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __truediv__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __rtruediv__(self, other: Field | core_defs.IntegralScalar) -> Never:  # type: ignore[misc] # Forward operator not callalbe
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __floordiv__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __rfloordiv__(self, other: Field | core_defs.IntegralScalar) -> Never:  # type: ignore[misc] # Forward operator not callalbe
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __pow__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __and__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __or__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'ConnectivityField' does not support this operation.")

    def __xor__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'ConnectivityField' does not support this operation.")


def is_connectivity_field(
    v: Any,
) -> TypeGuard[ConnectivityField]:
    # This function is introduced to localize the `type: ignore` because
    # extended_runtime_checkable does not make the protocol runtime_checkable
    # for mypy.
    # TODO(egparedes): remove it when extended_runtime_checkable is fixed
    return isinstance(v, ConnectivityField)  # type: ignore[misc] # we use extended_runtime_checkable


# Utility function to construct a `Field` from different buffer representations.
# Consider removing this function and using `Field` constructor directly. See also `_connectivity`.
@functools.singledispatch
def _field(
    definition: Any,
    /,
    *,
    domain: Optional[DomainLike] = None,
    dtype: Optional[core_defs.DType] = None,
) -> Field:
    raise NotImplementedError


# See comment for `_field`.
@functools.singledispatch
def _connectivity(
    definition: Any,
    /,
    codomain: Dimension,
    *,
    domain: Optional[DomainLike] = None,
    dtype: Optional[core_defs.DType] = None,
    skip_value: Optional[core_defs.IntegralScalar] = None,
) -> ConnectivityField:
    raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class GTInfo:
    definition: Any
    ir: Any


@dataclasses.dataclass(frozen=True)
class Backend:
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    # TODO : proper definition and implementation
    def generate_operator(self, ir):
        return ir


@runtime_checkable
class Connectivity(Protocol):
    max_neighbors: int
    has_skip_values: bool
    origin_axis: Dimension
    neighbor_axis: Dimension
    index_type: type[int] | type[np.int32] | type[np.int64]

    def mapped_index(
        self, cur_index: int | np.integer, neigh_index: int | np.integer
    ) -> Optional[int | np.integer]:
        """Return neighbor index."""


@runtime_checkable
class NeighborTable(Connectivity, Protocol):
    table: npt.NDArray


OffsetProviderElem: TypeAlias = Dimension | Connectivity
OffsetProvider: TypeAlias = Mapping[Tag, OffsetProviderElem]


@dataclasses.dataclass(frozen=True, eq=False)
class CartesianConnectivity(ConnectivityField[DimsT, DimT]):
    dimension: DimT
    offset: int = 0

    @classmethod
    def __gt_builtin_func__(cls, _: fbuiltins.BuiltInFunction) -> Never:  # type: ignore[override]
        raise NotImplementedError()

    @property
    def ndarray(self) -> Never:
        raise NotImplementedError()

    def asnumpy(self) -> Never:
        raise NotImplementedError()

    def as_scalar(self) -> Never:
        raise NotImplementedError()

    @functools.cached_property
    def domain(self) -> Domain:
        return Domain(dims=(self.dimension,), ranges=(UnitRange.infinite(),))

    @property
    def __gt_origin__(self) -> Never:
        raise TypeError("'CartesianConnectivity' does not support this operation.")

    @property
    def dtype(self) -> core_defs.DType[core_defs.IntegralScalar]:
        return core_defs.Int32DType()  # type: ignore[return-value]

    @functools.cached_property
    def codomain(self) -> DimT:
        return self.dimension

    @property
    def skip_value(self) -> None:
        return None

    @functools.cached_property
    def kind(self) -> ConnectivityKind:
        return ConnectivityKind(0)

    @classmethod
    def from_offset(
        cls,
        definition: int,
        /,
        codomain: DimT,
        *,
        domain: Optional[DomainLike] = None,
        dtype: Optional[core_defs.DTypeLike] = None,
    ) -> CartesianConnectivity:
        assert domain is None
        assert dtype is None
        return cls(codomain, definition)

    def inverse_image(self, image_range: UnitRange | NamedRange) -> Sequence[NamedRange]:
        if not isinstance(image_range, UnitRange):
            if image_range[0] != self.codomain:
                raise ValueError(
                    f"Dimension '{image_range[0]}' does not match the codomain dimension '{self.codomain}'."
                )

            image_range = image_range[1]

        assert isinstance(image_range, UnitRange)
        return ((self.codomain, image_range - self.offset),)

    def remap(self, index_field: ConnectivityField | fbuiltins.FieldOffset) -> ConnectivityField:
        raise NotImplementedError()

    __call__ = remap

    def restrict(self, index: AnyIndexSpec) -> Never:
        raise NotImplementedError()  # we could possibly implement with a FunctionField, but we don't have a use-case

    __getitem__ = restrict


_connectivity.register(numbers.Integral, CartesianConnectivity.from_offset)


@enum.unique
class GridType(StrEnum):
    CARTESIAN = "cartesian"
    UNSTRUCTURED = "unstructured"


def promote_dims(*dims_list: Sequence[Dimension]) -> list[Dimension]:
    """
    Find a unique ordering of multiple (individually ordered) lists of dimensions.

    The resulting list of dimensions contains all dimensions of the arguments
    in the order they originally appear. If no unique order exists or a
    contradicting order is found an exception is raised.

    A modified version (ensuring uniqueness of the order) of
    `Kahn's algorithm <https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm>`_
    is used to topologically sort the arguments.
    >>> from gt4py.next.common import Dimension
    >>> I, J, K = (Dimension(value=dim) for dim in ["I", "J", "K"])
    >>> promote_dims([I, J], [I, J, K]) == [I, J, K]
    True
    >>> promote_dims([I, J], [K])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
     ...
    ValueError: Dimensions can not be promoted. Could not determine order of the following dimensions: J, K.
    >>> promote_dims([I, J], [J, I])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
     ...
    ValueError: Dimensions can not be promoted. The following dimensions appear in contradicting order: I, J.
    """
    # build a graph with the vertices being dimensions and edges representing
    #  the order between two dimensions. The graph is encoded as a dictionary
    #  mapping dimensions to their predecessors, i.e. a dictionary containing
    #  adjacency lists. Since graphlib.TopologicalSorter uses predecessors
    #  (contrary to successors) we also use this directionality here.
    graph: dict[Dimension, set[Dimension]] = {}
    for dims in dims_list:
        if len(dims) == 0:
            continue
        # create a vertex for each dimension
        for dim in dims:
            graph.setdefault(dim, set())
        # add edges
        predecessor = dims[0]
        for dim in dims[1:]:
            graph[dim].add(predecessor)
            predecessor = dim

    # modified version of Kahn's algorithm
    topologically_sorted_list: list[Dimension] = []

    # compute in-degree for each vertex
    in_degree = {v: 0 for v in graph.keys()}
    for v1 in graph:
        for v2 in graph[v1]:
            in_degree[v2] += 1

    # process vertices with in-degree == 0
    # TODO(tehrengruber): avoid recomputation of zero_in_degree_vertex_list
    while zero_in_degree_vertex_list := [v for v, d in in_degree.items() if d == 0]:
        if len(zero_in_degree_vertex_list) != 1:
            raise ValueError(
                f"Dimensions can not be promoted. Could not determine "
                f"order of the following dimensions: "
                f"{', '.join((dim.value for dim in zero_in_degree_vertex_list))}."
            )
        v = zero_in_degree_vertex_list[0]
        del in_degree[v]
        topologically_sorted_list.insert(0, v)
        # update in-degree
        for predecessor in graph[v]:
            in_degree[predecessor] -= 1

    if len(in_degree.items()) > 0:
        raise ValueError(
            f"Dimensions can not be promoted. The following dimensions "
            f"appear in contradicting order: {', '.join((dim.value for dim in in_degree.keys()))}."
        )

    return topologically_sorted_list


class FieldBuiltinFuncRegistry:
    """
    Mixin for adding `fbuiltins` registry to a `Field`.

    Subclasses of a `Field` with `FieldBuiltinFuncRegistry` get their own registry,
    dispatching (via ChainMap) to its parent's registries.
    """

    _builtin_func_map: collections.ChainMap[fbuiltins.BuiltInFunction, Callable] = (
        collections.ChainMap()
    )

    def __init_subclass__(cls, **kwargs):
        cls._builtin_func_map = collections.ChainMap(
            {},  # New empty `dict` for new registrations on this class
            *[
                c.__dict__["_builtin_func_map"].maps[0]  # adding parent `dict`s in mro order
                for c in cls.__mro__
                if "_builtin_func_map" in c.__dict__
            ],
        )

    @classmethod
    def register_builtin_func(
        cls, /, op: fbuiltins.BuiltInFunction[_R, _P], op_func: Optional[Callable[_P, _R]] = None
    ) -> Any:
        assert op not in cls._builtin_func_map
        if op_func is None:  # when used as a decorator
            return functools.partial(cls.register_builtin_func, op)
        return cls._builtin_func_map.setdefault(op, op_func)

    @classmethod
    def __gt_builtin_func__(cls, /, func: fbuiltins.BuiltInFunction[_R, _P]) -> Callable[_P, _R]:
        return cls._builtin_func_map.get(func, NotImplemented)


#: Numeric value used to represent missing values in connectivities.
#: Equivalent to the `_FillValue` attribute in the UGRID Conventions
#: (see: http://ugrid-conventions.github.io/ugrid-conventions/).
_DEFAULT_SKIP_VALUE: Final[int] = -1
