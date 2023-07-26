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
from collections.abc import Sequence, Set
from typing import (
    Any,
    Iterator,
    Literal,
    Optional,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,
    final,
    runtime_checkable,
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
class UnitRange(Sequence, Set):
    """Range from `start` to `stop` with step size one."""

    start: int
    stop: int

    def __len__(self) -> int:
        return max(0, self.stop - self.start)

    def __repr__(self) -> str:
        return f"UnitRange({self.start}, {self.stop})"

    def __getitem__(self, index: int | slice) -> int | list[int]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        if index < 0:
            index += len(self)

        if 0 <= index < len(self):
            return self.start + index
        else:
            raise IndexError("UnitRange index out of range")


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
