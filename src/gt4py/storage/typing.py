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
from numbers import Integral
from typing import Any, Protocol, Tuple, Union

import numpy as np
from packaging import version

from gt4py.eve import extended_typing as xtyping


try:
    import cupy as cp
except ImportError:
    cp = None


if np.lib.NumpyVersion(np.__version__) >= "1.20.0":
    from numpy.typing import ArrayLike as _NpArrayLike, DTypeLike
else:
    _NpArrayLike = Any  # type: ignore[misc]  # assign multiple types in both branches
    DTypeLike = Any  # type: ignore[misc]  # assign multiple types in both branches

if cp is not None and version.parse(cp.__version__) >= version.parse("11.0.0a2"):
    from cupy.typing import ArrayLike as _CpArrayLike

    ArrayLike = Union[_NpArrayLike, _CpArrayLike]  # type: ignore[misc]  # assign multiple types in both branches
else:
    ArrayLike = _NpArrayLike  # type: ignore[misc]  # assign multiple types in both branches
if cp is not None:
    NdArray = Union[np.ndarray, cp.ndarray]  # type: ignore[misc]  # assign multiple types in both branches
else:
    NdArray = np.ndarray  # type: ignore[misc]  # assign multiple types in both branches

__all__ = ["ArrayLike", "DTypeLike"]

IntIndex: xtyping.TypeAlias = Integral

FieldIndex: xtyping.TypeAlias = Union[
    range, slice, IntIndex
]  # A `range` FieldIndex can be negative indicating a relative position with respect to origin, not wrap-around semantics like `slice` TODO(havogt): remove slice here
FieldIndices: xtyping.TypeAlias = Tuple[FieldIndex, ...]
FieldIndexOrIndices: xtyping.TypeAlias = Union[FieldIndex, FieldIndices]


ArrayIndex: xtyping.TypeAlias = Union[slice, IntIndex]
ArrayIndexOrIndices: xtyping.TypeAlias = Union[ArrayIndex, Tuple[ArrayIndex, ...]]


class DimensionIdentifier(Protocol):
    def __eq__(self, other: DimensionIdentifier) -> bool:
        ...


class StorageProtocol(Protocol):
    @property
    @abc.abstractmethod
    def __gt_dims__(self) -> tuple[DimensionIdentifier, ...]:
        ...


@xtyping.runtime_checkable
class LocatedField(StorageProtocol, Protocol):
    """A field with named dimensions providing read access."""

    @property
    @abc.abstractmethod
    def __gt_dims__(self) -> Tuple[DimensionIdentifier, ...]:
        ...

    # TODO(havogt): define generic Protocol to provide a concrete return type
    @abc.abstractmethod
    def field_getitem(self, indices: FieldIndexOrIndices) -> Any:
        ...


class MutableLocatedField(LocatedField, Protocol):
    """A LocatedField with write access."""

    # TODO(havogt): define generic Protocol to provide a concrete return type
    @abc.abstractmethod
    def field_setitem(self, indices: FieldIndexOrIndices, value: Any) -> None:
        ...
