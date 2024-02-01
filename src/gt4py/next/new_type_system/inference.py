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

import dataclasses
import typing
from typing import Any, Callable, Optional

import numpy as np

from . import types


@dataclasses.dataclass
class Pattern:
    """
    Type inference patterns for a particular `Type`.

    Each pattern contains a function to determine the type from
    an instance or from an annotation. If a function is None, the pattern
    cannot infer the type from either an instance or an annotation.

    The inference functions should return None in case they failed to match
    the instance or the annotation. If the type can be determined to belong
    to this pattern, but there was an error while inferring the type, an
    exception may be raised instead of returning None.
    """

    annotation: Optional[Callable[[TypeInferrer, Any], Optional[types.Type]]]
    """Function to infer the type from an annotation (i.e. type hint)."""

    instance: Optional[Callable[[TypeInferrer, Any], Optional[types.Type]]]
    """Function to infer the type from an instance (i.e. object)."""


@dataclasses.dataclass
class TypeInferrer:
    """
    Infers the type from an annotation or an instance.

    Contains a list of patterns and tries to match them one by one until one
    succeeds or all of them fail.
    """

    patterns: list[Pattern]
    """The list of patterns this inferrer tries to match."""

    def from_annotation(self, annotation: Any) -> Optional[types.Type]:
        """Infer the type from an annotation (type hint)."""
        for pattern in self.patterns:
            if pattern.annotation is not None:
                maybe_type = pattern.annotation(self, annotation)
                if maybe_type is not None:
                    return maybe_type
        return None

    def from_instance(
        self,
        instance: Any,
    ) -> Optional[types.Type]:
        """Infer the type from an instance (object)."""
        for pattern in self.patterns:
            if pattern.instance is not None:
                maybe_type = pattern.instance(self, instance)
                if maybe_type is not None:
                    return maybe_type
        return None


def primitive_from_annotation(_: TypeInferrer, annotation: Any) -> Optional[types.Type]:
    """Infer the type of integers and floats from annotations."""
    try:
        dtype = np.dtype(annotation)
        if not np.issctype(annotation):
            return None
        if dtype.kind == "i":
            if dtype.itemsize not in [1, 2, 4, 8]:
                return None
            return types.IntegerType(8 * dtype.itemsize, True)
        if dtype.kind == "u":
            if dtype.itemsize not in [1, 2, 4, 8]:
                return None
            return types.IntegerType(8 * dtype.itemsize, False)
        if dtype.kind == "f":
            if dtype.itemsize not in [2, 4, 8]:
                return None
            return types.FloatType(8 * dtype.itemsize)
        if dtype.kind == "b":
            return types.IntegerType(1, False)
    except ValueError:
        return None
    except TypeError:
        return None
    return None


def primitive_from_instance(inferrer: TypeInferrer, instance: Any) -> Optional[types.Type]:
    """Infer the type of integers and floats from instances."""
    return primitive_from_annotation(inferrer, type(instance))


def tuple_from_annotation(inferrer: TypeInferrer, annotation: Any) -> Optional[types.TupleType]:
    """Infer the type of tuples from annotations."""
    if typing.get_origin(annotation) == tuple:
        elements = [inferrer.from_annotation(element) for element in typing.get_args(annotation)]
        if not all(elements):
            return None
        return types.TupleType(typing.cast(list[types.Type], elements))
    return None


def tuple_from_instance(inferrer: TypeInferrer, instance: Any) -> Optional[types.TupleType]:
    """Infer the type of tuples from instances."""
    if isinstance(instance, tuple):
        elements = [inferrer.from_instance(element) for element in instance]
        if not all(elements):
            return None
        return types.TupleType(typing.cast(list[types.Type], elements))
    return None


inferrer = TypeInferrer(
    [
        Pattern(primitive_from_annotation, primitive_from_instance),
        Pattern(tuple_from_annotation, tuple_from_instance),
    ]
)
"""A type inferrer that contains patterns for core types."""
