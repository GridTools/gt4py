# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Optional,
    Protocol,
    SupportsFloat,
    SupportsInt,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from gt4py.eve.type_definitions import StrEnum


DimT = TypeVar("DimT", bound="Dimension")
DimsT = TypeVar("DimsT", bound=Sequence["Dimension"])
DT = TypeVar("DT", bound="DType")

Scalar: TypeAlias = SupportsInt | SupportsFloat | np.int32 | np.int64 | np.float32 | np.float64


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


class GTError:
    """Base class for GridTools exceptions.

    Notes:
        This base class has to be always inherited together with a standard
        exception, and thus it should not be used as direct superclass
        for custom exceptions. Inherit directly from :class:`GTTypeError`,
        :class:`GTTypeError`, ...

    """

    ...


class GTRuntimeError(GTError, RuntimeError):
    """Base class for GridTools run-time errors."""

    ...


class GTSyntaxError(GTError, SyntaxError):
    """Base class for GridTools syntax errors."""

    ...


class GTTypeError(GTError, TypeError):
    """Base class for GridTools type errors."""

    ...


class GTValueError(GTError, ValueError):
    """Base class for GridTools value errors."""

    ...
