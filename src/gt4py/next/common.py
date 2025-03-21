# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import collections
import dataclasses
import enum
import functools
import math
import types
from collections.abc import Mapping, Sequence

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.eve import utils
from gt4py.eve.extended_typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Final,
    Generic,
    Literal,
    NamedTuple,
    Never,
    Optional,
    ParamSpec,
    Protocol,
    Self,
    TypeAlias,
    TypeGuard,
    TypeVar,
    TypeVarTuple,
    Unpack,
    cast,
    overload,
    runtime_checkable,
)
from gt4py.eve.type_definitions import StrEnum


DimT = TypeVar("DimT", bound="Dimension")  # , covariant=True)
ShapeTs = TypeVarTuple("ShapeTs")


class Dims(tuple[Unpack[ShapeTs]]): ...


DimsT = TypeVar("DimsT", bound=Dims, covariant=True)

Tag: TypeAlias = str


@enum.unique
class DimensionKind(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    LOCAL = "local"

    def __str__(self) -> str:
        return self.value


_DIM_KIND_ORDER = {DimensionKind.HORIZONTAL: 0, DimensionKind.LOCAL: 1, DimensionKind.VERTICAL: 2}


def dimension_to_implicit_offset(dim: str) -> str:
    """
    Return name of offset implicitly defined by a dimension.

    Each dimension implicitly also defines an offset, such that we can allow syntax like::

        field(TDim + 1)

    without having to explicitly define an offset for ``TDim``. This function defines the respective
    naming convention.
    """
    return f"_{dim}Off"


@dataclasses.dataclass(frozen=True)
class Dimension:
    value: str
    kind: DimensionKind = dataclasses.field(default=DimensionKind.HORIZONTAL)

    def __str__(self) -> str:
        return f"{self.value}[{self.kind}]"

    def __call__(self, val: int) -> NamedIndex:
        return NamedIndex(self, val)

    def __add__(self, offset: int) -> Connectivity:
        # TODO(sf-n): just to avoid circular import. Move or refactor the FieldOffset to avoid this.
        from gt4py.next.ffront import fbuiltins

        assert isinstance(self.value, str)
        return fbuiltins.FieldOffset(
            dimension_to_implicit_offset(self.value), source=self, target=(self,)
        )[offset]

    def __sub__(self, offset: int) -> Connectivity:
        return self + (-offset)


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
    def infinite(cls) -> UnitRange:
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

    def is_empty(self) -> bool:
        return (
            self.start == 0 and self.stop == 0
        )  # post_init ensures that empty is represented as UnitRange(0, 0)

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

_Rng = TypeVar(
    "_Rng",
    FiniteUnitRange,
    UnitRange[Infinity, int],
    UnitRange[int, Infinity],
    UnitRange[Infinity, Infinity],
)

RangeLike: TypeAlias = (
    _Rng
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


class NamedRange(NamedTuple, Generic[_Rng]):
    dim: Dimension
    unit_range: _Rng

    def __str__(self) -> str:
        return f"{self.dim}={self.unit_range}"


IntIndex: TypeAlias = int | core_defs.IntegralScalar


class NamedIndex(NamedTuple):
    dim: Dimension
    value: IntIndex

    def __str__(self) -> str:
        return f"{self.dim}={self.value}"


FiniteNamedRange: TypeAlias = NamedRange[FiniteUnitRange]
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


def is_finite_named_range(v: NamedRange) -> TypeGuard[FiniteNamedRange]:
    return UnitRange.is_finite(v.unit_range)


def is_named_slice(obj: AnyIndexSpec) -> TypeGuard[slice]:
    return isinstance(obj, slice) and (
        isinstance(obj.start, NamedIndex) and isinstance(obj.stop, NamedIndex)
    )


def is_any_index_element(v: AnyIndexSpec) -> TypeGuard[AnyIndexElement]:
    return is_int_index(v) or isinstance(v, (NamedRange, NamedIndex, slice)) or v is Ellipsis


def is_absolute_index_sequence(v: AnyIndexSequence) -> TypeGuard[AbsoluteIndexSequence]:
    return isinstance(v, Sequence) and all(isinstance(e, (NamedRange, NamedIndex)) for e in v)


def is_relative_index_sequence(v: AnyIndexSequence) -> TypeGuard[RelativeIndexSequence]:
    return isinstance(v, tuple) and all(
        isinstance(e, slice) or is_int_index(e) or e is Ellipsis for e in v
    )


def as_any_index_sequence(index: AnyIndexSpec) -> AnyIndexSequence:
    # `cast` because mypy/typing doesn't special case 1-element tuples, i.e. `tuple[A|B] != tuple[A]|tuple[B]`
    return cast(AnyIndexSequence, (index,) if is_any_index_element(index) else index)


def named_range(v: tuple[Dimension, RangeLike]) -> NamedRange:
    if isinstance(v, NamedRange):
        return v
    return NamedRange(v[0], unit_range(v[1]))


@dataclasses.dataclass(frozen=True, init=False)
class Domain(Sequence[NamedRange[_Rng]], Generic[_Rng]):
    """Describes the `Domain` of a `Field` as a `Sequence` of `NamedRange` s."""

    dims: tuple[Dimension, ...]
    ranges: tuple[_Rng, ...]

    def __init__(
        self,
        *args: NamedRange[_Rng],
        dims: Optional[Sequence[Dimension]] = None,
        ranges: Optional[Sequence[_Rng]] = None,
    ) -> None:
        if dims is not None or ranges is not None:
            if dims is None and ranges is None:
                raise ValueError("Either specify both 'dims' and 'ranges' or neither.")
            if len(args) > 0:
                raise ValueError(
                    "No extra 'args' allowed when constructing from 'dims' and 'ranges'."
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
            if not all(isinstance(arg, NamedRange) for arg in args):
                raise ValueError(
                    f"Elements of 'Domain' need to be instances of 'NamedRange', got '{args}'."
                )
            dims, ranges = zip(*args) if args else ((), ())
            object.__setattr__(self, "dims", tuple(dims))
            object.__setattr__(self, "ranges", tuple(ranges))

        if len(set(self.dims)) != len(self.dims):
            raise NotImplementedError(f"Domain dimensions must be unique, not '{self.dims}'.")

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(r) for r in self.ranges)

    @property
    def size(self) -> Optional[int]:
        return math.prod(self.shape) if all(UnitRange.is_finite(r) for r in self.ranges) else None

    def __len__(self) -> int:
        return len(self.ranges)

    def __str__(self) -> str:
        return f"Domain({', '.join(f'{e}' for e in self)})"

    @overload
    def __getitem__(self, index: int) -> NamedRange: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    @overload
    def __getitem__(self, index: Dimension) -> NamedRange: ...

    def __getitem__(self, index: int | slice | Dimension) -> NamedRange | Domain:
        if isinstance(index, Dimension):
            try:
                index = self.dims.index(index)
            except ValueError as ex:
                raise KeyError(f"No Dimension of type '{index}' is present in the Domain.") from ex
        if isinstance(index, int):
            return NamedRange(dim=self.dims[index], unit_range=self.ranges[index])
        if isinstance(index, slice):
            dims_slice = self.dims[index]
            ranges_slice = self.ranges[index]
            return Domain(dims=dims_slice, ranges=ranges_slice)

        raise KeyError("Invalid index type, must be either int, slice, or Dimension.")

    def __and__(self, other: Domain) -> Domain:
        """
        Intersect `Domain`s, missing `Dimension`s are considered infinite.

        Examples:
            >>> I = Dimension("I")
            >>> J = Dimension("J")

            >>> Domain(NamedRange(I, UnitRange(-1, 3))) & Domain(NamedRange(I, UnitRange(1, 6)))
            Domain(dims=(Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>),), ranges=(UnitRange(1, 3),))

            >>> Domain(NamedRange(I, UnitRange(-1, 3)), NamedRange(J, UnitRange(2, 4))) & Domain(
            ...     NamedRange(I, UnitRange(1, 6))
            ... )
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

    @functools.cached_property
    def slice_at(self) -> utils.IndexerCallable[slice, Domain]:
        """
        Create a new domain by slicing the domain ranges at the provided relative slices.

        Examples:
            >>> I, J = Dimension("I"), Dimension("J")
            >>> domain = Domain(NamedRange(I, UnitRange(0, 10)), NamedRange(J, UnitRange(5, 15)))
            >>> domain.slice_at[2:3, 2:5]
            Domain(dims=(Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)), ranges=(UnitRange(2, 3), UnitRange(7, 10)))
        """

        def _domain_slicer(*args: slice) -> Domain:
            if not all(isinstance(a, slice) for a in args):
                raise TypeError(f"Indices must be 'slice's but got '{args}'")
            if len(args) != len(self):
                raise ValueError(
                    f"Number of provided slices ({len(args)}) does not match the number of dimensions ({len(self)})."
                )
            return Domain(dims=self.dims, ranges=[r[s] for r, s in zip(self.ranges, args)])

        return utils.IndexerCallable(_domain_slicer)

    @classmethod
    def is_finite(cls, obj: Domain) -> TypeGuard[FiniteDomain]:
        # classmethod since TypeGuards requires the guarded obj as separate argument
        return all(UnitRange.is_finite(rng) for rng in obj.ranges)

    def is_empty(self) -> bool:
        return any(rng.is_empty() for rng in self.ranges)

    @overload
    def dim_index(self, dim: Dimension, *, allow_missing: Literal[False]) -> int: ...

    @overload
    def dim_index(
        self, dim: Dimension, *, allow_missing: Literal[True] = True
    ) -> Optional[int]: ...

    def dim_index(self, dim: Dimension, *, allow_missing: bool = True) -> Optional[int]:
        if dim in self.dims:
            return self.dims.index(dim)
        elif allow_missing:
            return None
        else:
            raise ValueError(f"Dimension '{dim}' not found in Domain.")

    def pop(self, index: int | Dimension = -1) -> Domain:
        return self.replace(index)

    def insert(self, index: int | Dimension, *named_ranges: NamedRange) -> Domain:
        if isinstance(index, int) and index == len(self.dims):
            new_dims, new_ranges = zip(*named_ranges)
            return Domain(dims=self.dims + new_dims, ranges=self.ranges + new_ranges)
        else:
            return self.replace(index, *named_ranges)

    def replace(self, index: int | Dimension, *named_ranges: NamedRange) -> Domain:
        assert all(isinstance(nr, NamedRange) for nr in named_ranges)
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
        new_dims = (arg.dim for arg in named_ranges) if len(named_ranges) > 0 else ()
        new_ranges = (arg.unit_range for arg in named_ranges) if len(named_ranges) > 0 else ()
        dims = self.dims[:index] + tuple(new_dims) + self.dims[index + 1 :]
        ranges = self.ranges[:index] + tuple(new_ranges) + self.ranges[index + 1 :]

        return Domain(dims=dims, ranges=ranges)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # remove cached property
        state.pop("slice_at", None)
        return state


FiniteDomain: TypeAlias = Domain[FiniteUnitRange]


DomainLike: TypeAlias = (
    Sequence[tuple[Dimension, RangeLike]] | Mapping[Dimension, RangeLike]
)  # `Domain` is `Sequence[NamedRange]` and therefore a subset


def domain(domain_like: DomainLike) -> Domain:
    """
    Construct `Domain` from `DomainLike` object.

    Examples:
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
                ranges=tuple(UnitRange(0, s) for s in domain_like.values()),
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


@runtime_checkable
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

    # TODO(havogt)
    # This property is wrong, because for a function field we would not know to which NDArrayObject we want to convert
    # at the very least, we need to take an allocator and rename this to `as_ndarray`.
    @property
    def ndarray(self) -> core_defs.NDArrayObject: ...

    def __str__(self) -> str:
        return f"⟨{self.domain!s} → {self.dtype}⟩"

    @abc.abstractmethod
    def asnumpy(self) -> np.ndarray: ...

    @abc.abstractmethod
    def premap(self, index_field: Connectivity | fbuiltins.FieldOffset) -> Field: ...

    @abc.abstractmethod
    def restrict(self, item: AnyIndexSpec) -> Self: ...

    @abc.abstractmethod
    def as_scalar(self) -> core_defs.ScalarT: ...

    # Operators
    @abc.abstractmethod
    def __call__(
        self,
        index_field: Connectivity | fbuiltins.FieldOffset,
        *args: Connectivity | fbuiltins.FieldOffset,
    ) -> Field: ...

    @abc.abstractmethod
    def __getitem__(self, item: AnyIndexSpec) -> Self: ...

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


@runtime_checkable
class MutableField(Field[DimsT, core_defs.ScalarT], Protocol[DimsT, core_defs.ScalarT]):
    @abc.abstractmethod
    def __setitem__(self, index: AnyIndexSpec, value: Field | core_defs.ScalarT) -> None: ...


class ConnectivityKind(enum.Flag):
    """
    Describes the kind of connectivity field.

    - `ALTER_DIMS`: change the dimensions of the data field domain.
    - `ALTER_STRUCT`: transform structured of the data inside the field (non-compact transformation).

    | Dims \ Struct |    No                    |    Yes                   |
    | ------------- | ------------------------ | ------------------------ |
    |   No          | Translation (I -> I)     | Reshuffling (I x K -> K) |
    |   Yes         | Relocation (I -> I_half) | Remapping (V x V2E -> E) |

    """

    ALTER_DIMS = enum.auto()
    ALTER_STRUCT = enum.auto()

    @classmethod
    def translation(cls) -> ConnectivityKind:
        return cls(0)

    @classmethod
    def relocation(cls) -> ConnectivityKind:
        return cls.ALTER_DIMS

    @classmethod
    def reshuffling(cls) -> ConnectivityKind:
        return cls.ALTER_STRUCT

    @classmethod
    def remapping(cls) -> ConnectivityKind:
        return cls.ALTER_DIMS | cls.ALTER_STRUCT


@dataclasses.dataclass(frozen=True)
class ConnectivityType:  # TODO(havogt): would better live in type_specifications but would have to solve a circular import
    domain: tuple[Dimension, ...]
    codomain: Dimension
    skip_value: Optional[core_defs.IntegralScalar]
    dtype: core_defs.DType

    @property
    def has_skip_values(self) -> bool:
        return self.skip_value is not None


@dataclasses.dataclass(frozen=True)
class NeighborConnectivityType(ConnectivityType):
    # TODO(havogt): refactor towards encoding this information in the local dimensions of the ConnectivityType.domain
    max_neighbors: int

    @property
    def source_dim(self) -> Dimension:
        return self.domain[0]

    @property
    def neighbor_dim(self) -> Dimension:
        return self.domain[1]


@runtime_checkable
# type: ignore[misc] # DimT should be covariant, but then it breaks in other places
class Connectivity(Field[DimsT, core_defs.IntegralScalar], Protocol[DimsT, DimT]):
    @property
    @abc.abstractmethod
    def codomain(self) -> DimT:
        """
        The `codomain` is the set of all indices in a certain `Dimension`.

        We use the `Dimension` itself to describe the (infinite) set of all indices.

        Note:
        We could restrict the infinite codomain to only the indices that are actually contained in the mapping.
        Currently, this would just complicate implementation as we do not use this information.
        """

    def __gt_type__(self) -> ConnectivityType:
        if is_neighbor_connectivity(self):
            return NeighborConnectivityType(
                domain=self.domain.dims,
                codomain=self.codomain,
                dtype=self.dtype,
                skip_value=self.skip_value,
                max_neighbors=self.ndarray.shape[1],
            )
        else:
            return ConnectivityType(
                domain=self.domain.dims,
                codomain=self.codomain,
                dtype=self.dtype,
                skip_value=self.skip_value,
            )

    @property
    def kind(self) -> ConnectivityKind:
        return ConnectivityKind.remapping()

    @abc.abstractmethod
    def inverse_image(self, image_range: UnitRange | NamedRange) -> Sequence[NamedRange]: ...

    @property
    @abc.abstractmethod
    def skip_value(self) -> Optional[core_defs.IntegralScalar]: ...

    # Operators
    def __abs__(self) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __neg__(self) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __invert__(self) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __eq__(self, other: Any) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __ne__(self, other: Any) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __add__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __radd__(self, other: Field | core_defs.IntegralScalar) -> Never:  # type: ignore[misc] # Forward operator not callalbe
        raise TypeError("'Connectivity' does not support this operation.")

    def __sub__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __rsub__(self, other: Field | core_defs.IntegralScalar) -> Never:  # type: ignore[misc] # Forward operator not callalbe
        raise TypeError("'Connectivity' does not support this operation.")

    def __mul__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __rmul__(self, other: Field | core_defs.IntegralScalar) -> Never:  # type: ignore[misc] # Forward operator not callalbe
        raise TypeError("'Connectivity' does not support this operation.")

    def __truediv__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __rtruediv__(self, other: Field | core_defs.IntegralScalar) -> Never:  # type: ignore[misc] # Forward operator not callalbe
        raise TypeError("'Connectivity' does not support this operation.")

    def __floordiv__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __rfloordiv__(self, other: Field | core_defs.IntegralScalar) -> Never:  # type: ignore[misc] # Forward operator not callalbe
        raise TypeError("'Connectivity' does not support this operation.")

    def __pow__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __and__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __or__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __xor__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")


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
) -> Connectivity:
    raise NotImplementedError


class NeighborConnectivity(Connectivity, Protocol):
    # TODO(havogt): work towards encoding this properly in the type
    def __gt_type__(self) -> NeighborConnectivityType: ...


def is_neighbor_connectivity(obj: Any) -> TypeGuard[NeighborConnectivity]:
    if not isinstance(obj, Connectivity):
        return False
    domain_dims = obj.domain.dims
    return (
        len(domain_dims) == 2
        and domain_dims[0].kind is DimensionKind.HORIZONTAL
        and domain_dims[1].kind is DimensionKind.LOCAL
    )


class NeighborTable(
    NeighborConnectivity, Protocol
):  # TODO(havogt): try to express by inheriting from NdArrayConnectivityField (but this would require a protocol to move it out of `embedded.nd_array_field`)
    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        # Note that this property is currently already there from inheriting from `Field`,
        # however this seems wrong, therefore we explicitly introduce it here (or it should come
        # implicitly from the `NdArrayConnectivityField` protocol).
        ...


def is_neighbor_table(obj: Any) -> TypeGuard[NeighborTable]:
    return is_neighbor_connectivity(obj) and hasattr(obj, "ndarray")


OffsetProviderElem: TypeAlias = Dimension | NeighborConnectivity
OffsetProviderTypeElem: TypeAlias = Dimension | NeighborConnectivityType
OffsetProvider: TypeAlias = Mapping[Tag, OffsetProviderElem]
OffsetProviderType: TypeAlias = Mapping[Tag, OffsetProviderTypeElem]


def is_offset_provider(obj: Any) -> TypeGuard[OffsetProvider]:
    if not isinstance(obj, Mapping):
        return False
    return all(isinstance(el, OffsetProviderElem) for el in obj.values())


def is_offset_provider_type(obj: Any) -> TypeGuard[OffsetProviderType]:
    if not isinstance(obj, Mapping):
        return False
    return all(isinstance(el, OffsetProviderTypeElem) for el in obj.values())


def offset_provider_to_type(
    offset_provider: OffsetProvider | OffsetProviderType,
) -> OffsetProviderType:
    return {
        k: v.__gt_type__() if isinstance(v, Connectivity) else v for k, v in offset_provider.items()
    }


DomainDimT = TypeVar("DomainDimT", bound="Dimension")


@dataclasses.dataclass(frozen=True, eq=False)
class CartesianConnectivity(Connectivity[Dims[DomainDimT], DimT]):
    domain_dim: DomainDimT
    codomain: DimT
    offset: int = 0

    def __init__(
        self, domain_dim: DomainDimT, offset: int = 0, *, codomain: Optional[DimT] = None
    ) -> None:
        object.__setattr__(self, "domain_dim", domain_dim)
        object.__setattr__(self, "codomain", codomain if codomain is not None else domain_dim)
        object.__setattr__(self, "offset", offset)

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
        return Domain(dims=(self.domain_dim,), ranges=(UnitRange.infinite(),))

    @property
    def __gt_origin__(self) -> Never:
        raise TypeError("'CartesianConnectivity' does not support this operation.")

    @property
    def dtype(self) -> core_defs.DType[core_defs.IntegralScalar]:
        return core_defs.Int32DType()  # type: ignore[return-value]

    # This is a workaround to make this class concrete, since `codomain` is an
    # abstract property of the `Connectivity` Protocol.
    if not TYPE_CHECKING:

        @functools.cached_property
        def codomain(self) -> DimT:
            raise RuntimeError("This property should be always set in the constructor.")

    @property
    def skip_value(self) -> None:
        return None

    @functools.cached_property
    def kind(self) -> ConnectivityKind:
        return (
            ConnectivityKind.translation()
            if self.domain_dim == self.codomain
            else ConnectivityKind.relocation()
        )

    @classmethod
    def for_translation(
        cls, dimension: DomainDimT, offset: int
    ) -> CartesianConnectivity[DomainDimT, DomainDimT]:
        return cast(CartesianConnectivity[DomainDimT, DomainDimT], cls(dimension, offset))

    @classmethod
    def for_relocation(cls, old: DimT, new: DomainDimT) -> CartesianConnectivity[DomainDimT, DimT]:
        return cls(new, codomain=old)

    def inverse_image(self, image_range: UnitRange | NamedRange) -> Sequence[NamedRange]:
        if not isinstance(image_range, UnitRange):
            if image_range.dim != self.codomain:
                raise ValueError(
                    f"Dimension '{image_range.dim}' does not match the codomain dimension '{self.codomain}'."
                )

            image_range = image_range.unit_range

        assert isinstance(image_range, UnitRange)
        return (named_range((self.domain_dim, image_range - self.offset)),)

    def premap(
        self,
        index_field: Connectivity | fbuiltins.FieldOffset,
        *args: Connectivity | fbuiltins.FieldOffset,
    ) -> Connectivity:
        raise NotImplementedError()

    __call__ = premap

    def restrict(self, index: AnyIndexSpec) -> Never:
        raise NotImplementedError()  # we could possibly implement with a FunctionField, but we don't have a use-case

    __getitem__ = restrict


@enum.unique
class GridType(StrEnum):
    CARTESIAN = "cartesian"
    UNSTRUCTURED = "unstructured"


def _ordered_dims(dims: list[Dimension] | set[Dimension]) -> list[Dimension]:
    return sorted(dims, key=lambda dim: (_DIM_KIND_ORDER[dim.kind], dim.value))


def check_dims(dims: list[Dimension]) -> None:
    if sum(1 for dim in dims if dim.kind == DimensionKind.LOCAL) > 1:
        raise ValueError("There are more than one dimension with DimensionKind 'LOCAL'.")

    if dims != _ordered_dims(dims):
        raise ValueError(
            f"Dimensions '{', '.join(map(str, dims))}' are not ordered correctly, expected '{', '.join(map(str, _ordered_dims(dims)))}'."
        )


def promote_dims(*dims_list: Sequence[Dimension]) -> list[Dimension]:
    """
    Find an ordering of multiple lists of dimensions.

    The resulting list contains all unique dimensions from the input lists,
    sorted first by dims_kind_order, i.e., `Dimension.kind` (`HORIZONTAL` < `LOCAL` < `VERTICAL`) and then
    lexicographically by `Dimension.value`.

    Examples:
        >>> from gt4py.next.common import Dimension
        >>> I = Dimension("I", DimensionKind.HORIZONTAL)
        >>> J = Dimension("J", DimensionKind.HORIZONTAL)
        >>> K = Dimension("K", DimensionKind.VERTICAL)
        >>> E2V = Dimension("E2V", kind=DimensionKind.LOCAL)
        >>> E2C = Dimension("E2C", kind=DimensionKind.LOCAL)
        >>> promote_dims([J, K], [I, K]) == [I, J, K]
        True
        >>> promote_dims([K, J], [I, K])
        Traceback (most recent call last):
        ...
        ValueError: Dimensions 'K[vertical], J[horizontal]' are not ordered correctly, expected 'J[horizontal], K[vertical]'.
        >>> promote_dims([I, K], [J, E2V]) == [I, J, E2V, K]
        True
        >>> promote_dims([I, E2C], [E2V, K])
        Traceback (most recent call last):
        ...
        ValueError: There are more than one dimension with DimensionKind 'LOCAL'.
    """

    for dims in dims_list:
        check_dims(list(dims))
    unique_dims = {dim for dims in dims_list for dim in dims}

    promoted_dims = _ordered_dims(unique_dims)
    check_dims(promoted_dims)
    return promoted_dims


class FieldBuiltinFuncRegistry:
    """
    Mixin for adding `fbuiltins` registry to a `Field`.

    Subclasses of a `Field` with `FieldBuiltinFuncRegistry` get their own registry,
    dispatching (via ChainMap) to its parent's registries.
    """

    _builtin_func_map: collections.ChainMap[fbuiltins.BuiltInFunction, Callable] = (
        collections.ChainMap()
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
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
