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

import itertools
from abc import ABC, abstractmethod
import enum
import math
from collections.abc import Sequence
from functools import cached_property
from typing import (
    Any,
    Generic,
    Optional,
    Protocol,
    SupportsFloat,
    SupportsInt,
    TypeAlias,
    TypeVar,
    runtime_checkable, Union, Iterator, Literal, )

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from gt4py.eve.type_definitions import StrEnum

DimT = TypeVar("DimT", bound="Dimension")
DimsT = TypeVar("DimsT", bound=Sequence["Dimension"])
DT = TypeVar("DT", bound="Scalar")

Scalar: TypeAlias = SupportsInt | SupportsFloat | np.int32 | np.int64 | np.float32 | np.float64
Integer: TypeAlias = Union[int, Literal[math.inf, -math.inf]]
IntegerPair: TypeAlias = tuple[Integer, Integer]


@enum.unique
class DimensionKind(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    LOCAL = "local"

    def __str__(self):
        return f"{type(self).__name__}.{self.name}"


@dataclass(frozen=True)
class Dimension:
    value: str
    kind: DimensionKind = DimensionKind.HORIZONTAL

    def __str__(self):
        return f'Dimension(value="{self.value}", kind={self.kind})'


class DType:
    ...


class Field(Generic[DimsT, DT]):
    ...


@dataclass(frozen=True)
class GTInfo:
    definition: Any
    ir: Any


@dataclass(frozen=True)
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


class Set(ABC):
    @abstractmethod
    def empty_set(self) -> UnitRange:
        pass

    @abstractmethod
    def universe(self) -> UnitRange:
        pass


class IntegerSet(Set):
    """A set containing integers."""

    def empty_set(self) -> UnitRange:
        return UnitRange(0, 0)

    def universe(self) -> UnitRange:
        return UnitRange(-math.inf, math.inf)


class UnitRange(IntegerSet):
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


class CartesianSet:
    def __init__(self, ranges: list[UnitRange]) -> None:
        self.ranges = ranges

    @cached_property
    def dim(self) -> int:
        """Return the dimensionality of the Cartesian set."""
        return len(self.ranges)

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        """Return an iterator over all elements of the Cartesian set."""
        for elements in itertools.product(*self.ranges):
            yield elements

    def __str__(self) -> str:
        return " × ".join(str(range_obj) for range_obj in self.ranges)
