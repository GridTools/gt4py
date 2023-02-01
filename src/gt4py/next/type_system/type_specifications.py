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

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gt4py.eve.type_definitions import IntEnum
from gt4py.next import common as func_common


class TypeSpec:
    pass


@dataclass(frozen=True)
class DataType(TypeSpec):
    """
    A base type for all types that represent data storage.

    Derive floating point, integral or field types from this class.
    """


class CallableType:
    """
    A base type for all types are callable.

    Derive other callable types, such as FunctionType or FieldOperatorType from
    this class.
    """


@dataclass(frozen=True)
class DeferredType(TypeSpec):
    """Dummy used to represent a type not yet inferred."""

    constraint: Optional[type[TypeSpec] | tuple[type[TypeSpec], ...]]


@dataclass(frozen=True)
class VoidType(TypeSpec):
    """
    Return type of a function without return values.

    Note: only useful for stateful dialects.
    """


@dataclass(frozen=True)
class DimensionType(TypeSpec):
    dim: func_common.Dimension


@dataclass(frozen=True)
class OffsetType(TypeSpec):
    source: func_common.Dimension
    target: tuple[func_common.Dimension] | tuple[func_common.Dimension, func_common.Dimension]

    def __str__(self):
        return f"Offset[{self.source}, {self.target}]"


class ScalarKind(IntEnum):
    BOOL = 1
    INT32 = 32
    INT64 = 64
    # Python's "int" type in the Python AST should be mapped to ScalarKind.INT in our ASTs. The size, as
    # determined by numpy, varies by platform. (Size is the same as C's "long" type.)
    INT = INT32 if np.int_ == np.int32 else INT64
    FLOAT32 = 1032
    FLOAT64 = 1064
    DIMENSION = 2001
    STRING = 3001


@dataclass(frozen=True)
class ScalarType(DataType):
    kind: ScalarKind
    shape: Optional[list[int]] = None

    def __str__(self):
        kind_str = self.kind.name.lower()
        if self.shape is None:
            return kind_str
        return f"{kind_str}{self.shape}"


@dataclass(frozen=True)
class TupleType(DataType):
    types: list[DataType]

    def __str__(self):
        return f"tuple[{', '.join(map(str, self.types))}]"


@dataclass(frozen=True)
class FieldType(DataType, CallableType):
    dims: list[func_common.Dimension]
    dtype: ScalarType

    def __str__(self):
        dims = "..." if self.dims is Ellipsis else f"[{', '.join(dim.value for dim in self.dims)}]"
        return f"Field[{dims}, {self.dtype}]"


@dataclass(frozen=True)
class FunctionType(TypeSpec, CallableType):
    args: list[DataType | DeferredType]
    kwargs: dict[str, DataType | DeferredType]
    returns: DataType | DeferredType | VoidType

    def __str__(self):
        arg_strs = [str(arg) for arg in self.args]
        kwarg_strs = [f"{key}: {value}" for key, value in self.kwargs.items()]
        args_str = ", ".join((*arg_strs, *kwarg_strs))
        return f"({args_str}) -> {self.returns}"
