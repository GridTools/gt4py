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
import math
from collections.abc import Sequence
from typing import (
    Any,
    Iterator,
    Optional,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,
    final,
    runtime_checkable,
    Literal,
)

import numpy as np
import numpy.typing as npt

from gt4py._core import definitions as gt4py_defs
from gt4py.eve.type_definitions import StrEnum


DimT = TypeVar("DimT", bound="Dimension")
DimsT = TypeVar("DimsT", bound=Sequence["Dimension"])

DType = gt4py_defs.DType
Scalar: TypeAlias = gt4py_defs.Scalar
ScalarT = gt4py_defs.ScalarT
NDArrayObject = gt4py_defs.NDArrayObject
Integer: TypeAlias = Union[int, Literal[math.inf, -math.inf]]
IntegerPair: TypeAlias = tuple[Integer, Integer]


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


# class SetProto(Protocol):
#     def empty_set(self) -> UnitRange:
#         ...

#     def universe(self) -> UnitRange:
#         ...


# class IntegerSet: # TODO use collection protocols
#     """A set containing integers."""

#     def empty_set(self) -> UnitRange:
#         return UnitRange(0, 0)

#     def universe(self) -> UnitRange:
#         return UnitRange(-math.inf, math.inf)


class UnitRange:  # TODO use collection protocols: Set, Sequence
    """Range from `start` to `stop` with step size one."""

    start: Integer
    stop: Integer

    def __init__(self, start: Integer, stop: Integer) -> None:
        assert stop >= start
        self.start = start
        self.stop = stop

        # canonicalize
        if self.empty:
            self.start = 0
            self.stop = 0

    @property
    def size(self) -> Integer:
        """Return the number of elements."""
        assert self.start <= self.stop
        return self.stop - self.start

    @property
    def empty(self) -> bool:
        """Return if the range is empty"""
        return self.start >= self.stop

    @property
    def bounds(self) -> UnitRange:
        """Smallest range containing all elements. In this case itself."""
        return self

    def __iter__(self) -> Iterator:
        """Return an iterator over all elements of the set."""
        return range(self.start, self.stop).__iter__()

    def as_tuple(self) -> IntegerPair:
        """Return the start and stop elements of the set as a tuple."""
        return self.start, self.stop

    def __str__(self) -> str:
        return f"UnitRange({self.start}, {self.stop})"


DomainT: TypeAlias = tuple[tuple[Dimension, UnitRange], ...]


class Field(Protocol[DimsT, ScalarT]):
    @property
    def domain(self) -> DimsT:
        ...

    @property
    def dtype(self) -> DType[ScalarT]:
        ...

    @property
    def value_type(self) -> ScalarT:
        ...

    @property
    def ndarray(self) -> NDArrayObject:
        ...

    @final
    def __setattr__(self, key, value) -> None:
        raise TypeError("Immutable type")

    @final
    def __setitem__(self, key, value) -> None:
        raise TypeError("Immutable type")

    def __str__(self) -> str:
        codomain = (
            f"{self.value_type!s}"
            if isinstance(self.value_type, Dimension)
            else self.value_type.__name__
        )
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
    def __add__(self, other: Field | ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __radd__(self, other: Field | ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __sub__(self, other: Field | ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __rsub__(self, other: Field | ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __mul__(self, other: Field | ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __rmul__(self, other: Field | ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __floordiv__(self, other: Field | ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __rfloordiv__(self, other: Field | ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __truediv__(self, other: Field | ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __rtruediv__(self, other: Field | ScalarT) -> Field:
        ...

    @abc.abstractmethod
    def __pow__(self, other: Field | ScalarT) -> Field:
        ...


@functools.singledispatch
def field(
    definition: Any,
    /,
    *,
    domain: Optional[DomainT] = None,
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
