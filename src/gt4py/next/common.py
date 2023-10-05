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
import sys
import types
from collections.abc import Mapping, Sequence, Set
from typing import overload

import numpy as np
import numpy.typing as npt

from gt4py._core import definitions as core_defs
from gt4py.eve.extended_typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Optional,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeGuard,
    TypeVar,
    cast,
    extended_runtime_checkable,
    runtime_checkable,
)
from gt4py.eve.type_definitions import StrEnum


DimsT = TypeVar(
    "DimsT", covariant=True
)  # bound to `Sequence[Dimension]` if instance of Dimension would be a type


class Infinity(int):
    @classmethod
    def positive(cls) -> Infinity:
        return cls(sys.maxsize)

    @classmethod
    def negative(cls) -> Infinity:
        return cls(-sys.maxsize)


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


@dataclasses.dataclass(frozen=True)
class UnitRange(Sequence[int], Set[int]):
    """Range from `start` to `stop` with step size one."""

    start: int
    stop: int

    def __post_init__(self):
        if self.stop <= self.start:
            # make UnitRange(0,0) the single empty UnitRange
            object.__setattr__(self, "start", 0)
            object.__setattr__(self, "stop", 0)

    @classmethod
    def infinity(cls) -> UnitRange:
        return cls(Infinity.negative(), Infinity.positive())

    def __len__(self) -> int:
        if Infinity.positive() in (abs(self.start), abs(self.stop)):
            return Infinity.positive()
        return max(0, self.stop - self.start)

    def __repr__(self) -> str:
        return f"UnitRange({self.start}, {self.stop})"

    @overload
    def __getitem__(self, index: int) -> int:
        ...

    @overload
    def __getitem__(self, index: slice) -> UnitRange:
        ...

    def __getitem__(self, index: int | slice) -> int | UnitRange:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("UnitRange: step required to be `1`.")
            new_start = self.start + (start or 0)
            new_stop = (self.start if stop > 0 else self.stop) + stop
            return UnitRange(new_start, new_stop)
        else:
            if index < 0:
                index += len(self)

            if 0 <= index < len(self):
                return self.start + index
            else:
                raise IndexError("UnitRange index out of range")

    def __and__(self, other: Set[Any]) -> UnitRange:
        if isinstance(other, UnitRange):
            start = max(self.start, other.start)
            stop = min(self.stop, other.stop)
            return UnitRange(start, stop)
        else:
            raise NotImplementedError("Can only find the intersection between UnitRange instances.")

    def __str__(self) -> str:
        return f"({self.start}:{self.stop})"


RangeLike: TypeAlias = UnitRange | range | tuple[int, int]


def unit_range(r: RangeLike) -> UnitRange:
    if isinstance(r, UnitRange):
        return r
    if isinstance(r, range):
        if r.step != 1:
            raise ValueError(f"`UnitRange` requires step size 1, got `{r.step}`.")
        return UnitRange(r.start, r.stop)
    if isinstance(r, tuple) and isinstance(r[0], int) and isinstance(r[1], int):
        return UnitRange(r[0], r[1])
    raise ValueError(f"`{r}` cannot be interpreted as `UnitRange`.")


IntIndex: TypeAlias = int | core_defs.IntegralScalar
NamedIndex: TypeAlias = tuple[Dimension, IntIndex]
NamedRange: TypeAlias = tuple[Dimension, UnitRange]
RelativeIndexElement: TypeAlias = IntIndex | slice | types.EllipsisType
AbsoluteIndexElement: TypeAlias = NamedIndex | NamedRange
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


def is_named_index(v: AnyIndexSpec) -> TypeGuard[NamedRange]:
    return (
        isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], Dimension) and is_int_index(v[1])
    )


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


@dataclasses.dataclass(frozen=True, init=False)
class Domain(Sequence[NamedRange]):
    """Describes the `Domain` of a `Field` as a `Sequence` of `NamedRange` s."""

    dims: tuple[Dimension, ...]
    ranges: tuple[UnitRange, ...]

    def __init__(
        self,
        *args: NamedRange,
        dims: Optional[tuple[Dimension, ...]] = None,
        ranges: Optional[tuple[UnitRange, ...]] = None,
    ) -> None:
        if dims is not None or ranges is not None:
            if dims is None and ranges is None:
                raise ValueError("Either both none of `dims` and `ranges` must be specified.")
            if len(args) > 0:
                raise ValueError(
                    "No extra `args` allowed when constructing fomr `dims` and `ranges`."
                )

            assert dims is not None and ranges is not None  # for mypy
            if not all(isinstance(dim, Dimension) for dim in dims):
                raise ValueError(
                    f"`dims` argument needs to be a `tuple[Dimension, ...], got `{dims}`."
                )
            if not all(isinstance(rng, UnitRange) for rng in ranges):
                raise ValueError(
                    f"`ranges` argument needs to be a `tuple[UnitRange, ...], got `{ranges}`."
                )
            if len(dims) != len(ranges):
                raise ValueError(
                    f"Number of provided dimensions ({len(dims)}) does not match number of provided ranges ({len(ranges)})."
                )

            object.__setattr__(self, "dims", dims)
            object.__setattr__(self, "ranges", ranges)
        else:
            if not all(is_named_range(arg) for arg in args):
                raise ValueError(f"Elements of `Domain` need to be `NamedRange`s, got `{args}`.")
            dims, ranges = zip(*args) if args else ((), ())
            object.__setattr__(self, "dims", tuple(dims))
            object.__setattr__(self, "ranges", tuple(ranges))

        if len(set(self.dims)) != len(self.dims):
            raise NotImplementedError(f"Domain dimensions must be unique, not {self.dims}.")

    def __len__(self) -> int:
        return len(self.ranges)

    @overload
    def __getitem__(self, index: int) -> NamedRange:
        ...

    @overload
    def __getitem__(self, index: slice) -> Domain:
        ...

    @overload
    def __getitem__(self, index: Dimension) -> NamedRange:
        ...

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
            except ValueError:
                raise KeyError(f"No Dimension of type {index} is present in the Domain.")
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
    """
    if isinstance(domain_like, Domain):
        return domain_like
    if isinstance(domain_like, Sequence):
        return Domain(*tuple(named_range(d) for d in domain_like))
    if isinstance(domain_like, Mapping):
        return Domain(
            dims=tuple(domain_like.keys()),
            ranges=tuple(unit_range(r) for r in domain_like.values()),
        )
    raise ValueError(f"`{domain_like}` is not `DomainLike`.")


def _broadcast_ranges(
    broadcast_dims: Sequence[Dimension], dims: Sequence[Dimension], ranges: Sequence[UnitRange]
) -> tuple[UnitRange, ...]:
    return tuple(
        ranges[dims.index(d)] if d in dims else UnitRange.infinity() for d in broadcast_dims
    )


if TYPE_CHECKING:
    import gt4py.next.ffront.fbuiltins as fbuiltins

    _Value: TypeAlias = "Field" | core_defs.ScalarT
    _P = ParamSpec("_P")
    _R = TypeVar("_R", _Value, tuple[_Value, ...])

    class GTBuiltInFuncDispatcher(Protocol):
        def __call__(self, func: fbuiltins.BuiltInFunction[_R, _P], /) -> Callable[_P, _R]:
            ...


class NextGTDimsInterface(Protocol):
    """
    A `GTDimsInterface` is an object providing the `__gt_dims__` property, naming :class:`Field` dimensions.

    The dimension names are objects of type :class:`Dimension`, in contrast to :mod:`gt4py.cartesian`,
    where the labels are `str` s with implied semantics, see :class:`~gt4py._core.definitions.GTDimsInterface` .
    """

    # TODO(havogt): unify with GTDimsInterface, ideally in backward compatible way
    @property
    def __gt_dims__(self) -> tuple[Dimension, ...]:
        ...


@extended_runtime_checkable
class Field(NextGTDimsInterface, core_defs.GTOriginInterface, Protocol[DimsT, core_defs.ScalarT]):
    __gt_builtin_func__: ClassVar[GTBuiltInFuncDispatcher]

    @property
    def domain(self) -> Domain:
        ...

    @property
    def dtype(self) -> core_defs.DType[core_defs.ScalarT]:
        ...

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        ...

    def __str__(self) -> str:
        return f"⟨{self.domain!s} → {self.dtype}⟩"

    @abc.abstractmethod
    def remap(self, index_field: Field) -> Field:
        ...

    @abc.abstractmethod
    def restrict(self, item: AnyIndexSpec) -> Field | core_defs.ScalarT:
        ...

    # Operators
    @abc.abstractmethod
    def __call__(self, index_field: Field) -> Field:
        ...

    @abc.abstractmethod
    def __getitem__(self, item: AnyIndexSpec) -> Field | core_defs.ScalarT:
        ...

    @abc.abstractmethod
    def __abs__(self) -> Field:
        ...

    @abc.abstractmethod
    def __neg__(self) -> Field:
        ...

    @abc.abstractmethod
    def __invert__(self) -> Field:
        """Only defined for `Field` of value type `bool`."""

    @abc.abstractmethod
    def __add__(self, other: Field | core_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __radd__(self, other: Field | core_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __sub__(self, other: Field | core_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __rsub__(self, other: Field | core_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __mul__(self, other: Field | core_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __rmul__(self, other: Field | core_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __floordiv__(self, other: Field | core_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __rfloordiv__(self, other: Field | core_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __truediv__(self, other: Field | core_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __rtruediv__(self, other: Field | core_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __pow__(self, other: Field | core_defs.ScalarT) -> Field:
        ...

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
    def __setitem__(self, index: AnyIndexSpec, value: Field | core_defs.ScalarT) -> None:
        ...


def is_mutable_field(
    v: Field,
) -> TypeGuard[MutableField]:
    # This function is introduced to localize the `type: ignore` because
    # extended_runtime_checkable does not make the protocol runtime_checkable
    # for mypy.
    # TODO(egparedes): remove it when extended_runtime_checkable is fixed
    return isinstance(v, MutableField)  # type: ignore[misc] # we use extended_runtime_checkable


@functools.singledispatch
def field(
    definition: Any,
    /,
    *,
    domain: Optional[DomainLike] = None,
    dtype: Optional[core_defs.DType] = None,
) -> Field:
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
    >>> promote_dims([I, J], [K]) # doctest: +ELLIPSIS
    Traceback (most recent call last):
     ...
    ValueError: Dimensions can not be promoted. Could not determine order of the following dimensions: J, K.
    >>> promote_dims([I, J], [J, I]) # doctest: +ELLIPSIS
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

    _builtin_func_map: collections.ChainMap[
        fbuiltins.BuiltInFunction, Callable
    ] = collections.ChainMap()

    def __init_subclass__(cls, **kwargs):
        cls._builtin_func_map = collections.ChainMap(
            {},  # New empty `dict`` for new registrations on this class
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
