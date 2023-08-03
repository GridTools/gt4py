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
from collections.abc import Sequence, Set, Collection
from typing import overload, Tuple

import numpy as np
import numpy.typing as npt

from gt4py._core import definitions as gt4py_defs
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
    Union,
    extended_runtime_checkable,
    final,
    runtime_checkable,
)
from gt4py.eve.type_definitions import StrEnum

DimT = TypeVar("DimT", bound="Dimension")
DimsT = TypeVar("DimsT", bound=Sequence["Dimension"], covariant=True)

DType = gt4py_defs.DType
Scalar: TypeAlias = gt4py_defs.Scalar
NDArrayObject = gt4py_defs.NDArrayObject


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
    kind: DimensionKind = DimensionKind.HORIZONTAL

    def __str__(self):
        return f'Dimension(value="{self.value}", kind={self.kind})'


DomainLike = Union[Sequence[Dimension], Dimension, str]


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

    def __len__(self) -> int:
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

    def __and__(self, other: UnitRange) -> UnitRange:
        start = max(self.start, other.start)
        stop = min(self.stop, other.stop)

        # Handle the case where there is no overlap
        if stop <= start:
            return UnitRange(0, 0)

        return UnitRange(start, stop)


DomainT: TypeAlias = tuple[tuple[Dimension, UnitRange], ...]


def promote_dims(*dims_list: list[Dimension]) -> list[Dimension]:
    """
    Find a unique ordering of multiple (individually ordered) lists of dimensions.

    The resulting list of dimensions contains all dimensions of the arguments
    in the order they originally appear. If no unique order exists or a
    contradicting order is found an exception is raised.

    A modified version (ensuring uniqueness of the order) of
    `Kahn's algorithm <https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm>`_
    is used to topologically sort the arguments.

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


class Domain(Collection):
    dims: Tuple[Dimension]
    ranges: Tuple[UnitRange]

    def __init__(self, dims: Tuple[Dimension, ...], ranges: Tuple[UnitRange, ...]):
        self.dims = dims
        self.ranges = ranges

    def __len__(self) -> int:
        return len(self.ranges)

    def __iter__(self):
        return iter(zip(self.dims, self.ranges))

    def __contains__(self, item: Union[int, UnitRange]) -> bool:
        if isinstance(item, int):
            return any(item in range_ for range_ in self.ranges)
        elif isinstance(item, UnitRange):
            return any(
                item.start >= range_.start and item.stop <= range_.stop for range_ in self.ranges
            )
        return False

    def __and__(self, other: Domain) -> Domain:
        broadcast_dims = tuple(promote_dims(self.dims, other.dims))
        broadcasted_ranges_self = self.broadcast_ranges(broadcast_dims)
        broadcasted_ranges_other = other.broadcast_ranges(broadcast_dims)

        intersected_ranges = []
        for range1, range2 in zip(broadcasted_ranges_self, broadcasted_ranges_other):
            intersected_range = range1 & range2
            intersected_ranges.append(intersected_range)

        return Domain(broadcast_dims, tuple(intersected_ranges))

    def broadcast_ranges(self, broadcast_dims: Tuple[Dimension, ...]) -> Tuple[UnitRange, ...]:
        if len(self.dims) == len(broadcast_dims):
            return self.ranges

        # Broadcast dimensions with infinite sizes for missing ranges
        broadcasted_ranges = list(self.ranges)
        for i in range(len(broadcast_dims) - len(self.dims)):
            broadcasted_ranges.append(UnitRange(float("-inf"), float("inf")))

        return tuple(broadcasted_ranges)


if TYPE_CHECKING:
    import gt4py.next.ffront.fbuiltins as fbuiltins

    _Value: TypeAlias = "Field" | gt4py_defs.ScalarT
    _P = ParamSpec("_P")
    _R = TypeVar("_R", _Value, tuple[_Value, ...])

    class GTBuiltInFuncDispatcher(Protocol):
        def __call__(self, func: fbuiltins.BuiltInFunction[_R, _P], /) -> Callable[_P, _R]:
            ...


@extended_runtime_checkable
class Field(Protocol[DimsT, gt4py_defs.ScalarT]):
    __gt_builtin_func__: ClassVar[GTBuiltInFuncDispatcher]

    @property
    def domain(self) -> DomainT:
        ...

    @property
    def dtype(self) -> DType[gt4py_defs.ScalarT]:
        ...

    @property
    def value_type(self) -> type[gt4py_defs.ScalarT]:
        ...

    @property
    def ndarray(self) -> NDArrayObject:
        ...

    def __str__(self) -> str:
        codomain = self.value_type.__name__
        return f"⟨{self.domain!s} → {codomain}⟩"

    @abc.abstractmethod
    def remap(self, index_field: Field) -> Field:
        ...

    @abc.abstractmethod
    def restrict(self, item: "DomainLike") -> Field:
        ...

    # Operators
    @abc.abstractmethod
    def __call__(self, index_field: Field) -> Field:
        ...

    @abc.abstractmethod
    def __getitem__(self, item: "DomainLike") -> Field:
        ...

    @abc.abstractmethod
    def __abs__(self) -> Field:
        ...

    @abc.abstractmethod
    def __neg__(self) -> Field:
        ...

    @abc.abstractmethod
    def __add__(self, other: Field | gt4py_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __radd__(self, other: Field | gt4py_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __sub__(self, other: Field | gt4py_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __rsub__(self, other: Field | gt4py_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __mul__(self, other: Field | gt4py_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __rmul__(self, other: Field | gt4py_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __floordiv__(self, other: Field | gt4py_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __rfloordiv__(self, other: Field | gt4py_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __truediv__(self, other: Field | gt4py_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __rtruediv__(self, other: Field | gt4py_defs.ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __pow__(self, other: Field | gt4py_defs.ScalarT) -> Field:
        ...


class FieldABC(Field[DimsT, gt4py_defs.ScalarT]):
    """Abstract base class for implementations of the :class:`Field` protocol."""

    @final
    def __setattr__(self, key, value) -> None:
        raise TypeError("Immutable type")

    @final
    def __setitem__(self, key, value) -> None:
        raise TypeError("Immutable type")


@functools.singledispatch
def field(
    definition: Any,
    /,
    *,
    domain: Optional[Any] = None,  # TODO(havogt): provide domain_like to DomainT conversion
    value_type: Optional[type] = None,
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
