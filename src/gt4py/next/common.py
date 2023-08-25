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
import dataclasses
import enum
import functools
import sys
from collections.abc import Sequence, Set
from types import EllipsisType
from typing import TypeGuard, overload

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
    TypeVar,
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
        return f"{type(self).__name__}.{self.name}"


@dataclasses.dataclass(frozen=True)
class Dimension:
    value: str
    kind: DimensionKind = dataclasses.field(default=DimensionKind.HORIZONTAL)

    def __str__(self):
        return f'Dimension(value="{self.value}", kind={self.kind})'


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


IntIndex: TypeAlias = int | np.integer
DomainRange: TypeAlias = UnitRange | IntIndex
NamedRange: TypeAlias = tuple[Dimension, UnitRange]
NamedIndex: TypeAlias = tuple[Dimension, IntIndex]
DomainSlice: TypeAlias = Sequence[NamedRange | NamedIndex]
FieldSlice: TypeAlias = (
    DomainSlice
    | tuple[slice | IntIndex | EllipsisType, ...]
    | slice
    | IntIndex
    | EllipsisType
    | NamedRange
    | NamedIndex
)


def is_int_index(p: Any) -> TypeGuard[IntIndex]:
    return isinstance(p, (int, np.integer))


def is_named_range(v: Any) -> TypeGuard[NamedRange]:
    return (
        isinstance(v, tuple)
        and len(v) == 2
        and isinstance(v[0], Dimension)
        and isinstance(v[1], UnitRange)
    )


def is_named_index(v: Any) -> TypeGuard[NamedRange]:
    return (
        isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], Dimension) and is_int_index(v[1])
    )


def is_domain_slice(v: Any) -> TypeGuard[DomainSlice]:
    return isinstance(v, Sequence) and all(is_named_range(e) or is_named_index(e) for e in v)


@dataclasses.dataclass(frozen=True)
class Domain(Sequence[NamedRange]):
    dims: tuple[Dimension, ...]
    ranges: tuple[UnitRange, ...]

    def __post_init__(self):
        if len(set(self.dims)) != len(self.dims):
            raise NotImplementedError(f"Domain dimensions must be unique, not {self.dims}.")

        if len(self.dims) != len(self.ranges):
            raise ValueError(
                f"Number of provided dimensions ({len(self.dims)}) does not match number of provided ranges ({len(self.ranges)})."
            )

    def __len__(self) -> int:
        return len(self.ranges)

    @overload
    def __getitem__(self, index: int) -> NamedRange:
        ...

    @overload
    def __getitem__(self, index: slice) -> "Domain":
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
            return Domain(dims_slice, ranges_slice)
        elif isinstance(index, Dimension):
            try:
                index_pos = self.dims.index(index)
                return self.dims[index_pos], self.ranges[index_pos]
            except ValueError:
                raise KeyError(f"No Dimension of type {index} is present in the Domain.")
        else:
            raise KeyError("Invalid index type, must be either int, slice, or Dimension.")

    def __and__(self, other: "Domain") -> "Domain":
        broadcast_dims = tuple(promote_dims(self.dims, other.dims))
        intersected_ranges = tuple(
            rng1 & rng2
            for rng1, rng2 in zip(
                _broadcast_ranges(broadcast_dims, self.dims, self.ranges),
                _broadcast_ranges(broadcast_dims, other.dims, other.ranges),
            )
        )
        return Domain(broadcast_dims, intersected_ranges)


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

    The dimension names are objects of type :class:`Dimension`, in contrast to :py:mod:`gt4py.cartesian`,
    where the labels are `str` s with implied semantics, see :py:class:`~gt4py._core.definitions.GTDimsInterface` .
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
    def restrict(self, item: FieldSlice) -> Field | core_defs.ScalarT:
        ...

    # Operators
    @abc.abstractmethod
    def __call__(self, index_field: Field) -> Field:
        ...

    @abc.abstractmethod
    def __getitem__(self, item: FieldSlice) -> Field | core_defs.ScalarT:
        ...

    @abc.abstractmethod
    def __abs__(self) -> Field:
        ...

    @abc.abstractmethod
    def __neg__(self) -> Field:
        ...

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
    def __setitem__(self, index: FieldSlice, value: Field | core_defs.ScalarT) -> None:
        ...


def is_mutable_field(
    v: Any,
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
    domain: Optional[Any] = None,  # TODO(havogt): provide domain_like to Domain conversion
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
