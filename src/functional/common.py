# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar


DimT = TypeVar("DimT", bound="Dimension")
DimsT = TypeVar("DimsT", bound=Sequence["Dimension"])
DT = TypeVar("DT", bound="DType")


class _NoSubclassing(type):
    """Utility metaclass prohibiting subclassing."""

    def __new__(cls, name, bases, classdict):
        for b in bases:
            if isinstance(b, _NoSubclassing):
                raise TypeError(f"Type '{b.__name__}' must not be subclassed.")
        return type.__new__(cls, name, bases, classdict)


@dataclass(frozen=True)
class Dimension(metaclass=_NoSubclassing):
    # TODO(tehrengruber): Revisit. Touches to many open questions to be resolved
    #  meaningfully. For now we just prohibit subclassing. The "iterator frontend"
    #  needs to be able to create instances of Dimension from an axis literal to
    #  construct the domain. However the dimensions of a field are passed as is.
    #  As such using an instance of a subclass of Dimension will break all equality
    #  checks between dimensions obtained from the domain and those obtained
    #  from fields.

    value: str
    local: bool = dataclasses.field(default=False)


class DType:
    ...


class Field(Generic[DimsT, DT]):
    ...


@dataclass(frozen=True)
class GTInfo:
    definition: Any
    ir: Any


class FieldOperator(abc.ABC):
    __gt_info__: GTInfo

    def __call__(self, *args: Field, **kwds: Field) -> Field | Sequence[Field]:
        ...


@dataclass(frozen=True)
class Backend:
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    # TODO : proper definition and implementation
    def generate_operator(self, ir):
        return ir


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
