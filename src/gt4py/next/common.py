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
from collections.abc import Sequence, Set
from typing import overload

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
