#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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

import re
import types
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

import eve
from eve import Node
from eve.traits import SymbolTableTrait
from eve.type_definitions import IntEnum, SourceLocation, StrEnum, SymbolRef
from functional import common


class Dimension(Node):
    name: str


class ScalarKind(IntEnum):
    BOOL = 1
    INT32 = 32
    INT64 = 64
    FLOAT32 = 1032
    FLOAT64 = 1064

    @staticmethod
    def from_numpy(value: npt.DTypeLike) -> ScalarKind:
        try:
            dt = np.dtype(value)
            if dt.shape == () and dt.fields is None:
                return ScalarKind.__DTYPE_TO_KIND[dt.base.type]

            raise common.GTValueError(f"Non-trivial dtypes like '{value}' are not yet supported")

        except TypeError as err:
            raise common.GTTypeError(f"Invalid type definition ({value})") from err


ScalarKind.__DTYPE_TO_KIND = types.MappingProxyType(
    {
        np.bool_: ScalarKind.BOOL,
        np.int32: ScalarKind.INT32,
        np.int64: ScalarKind.INT64,
        np.float32: ScalarKind.FLOAT32,
        np.float64: ScalarKind.FLOAT64,
    }
)


class Type(Node):
    @staticmethod
    def from_python(value: Any) -> Type:
        match value:
            case bool(), int(), float(), np.generic():
                return Type.from_python(type(value))
            case type() as t if issubclass(t, (bool, int, float, np.generic)):
                return ScalarKind.from_dtype(value)
            case tuple() as tuple_value:
                return TupleType(types=[DataType.from_python(t) for t in tuple_value])
            case types.FunctionType():
                args = []
                kwargs = []
                returns = []
                return FunctionType(args, kwargs, returns)
            case _:
                raise common.GTValueError(f"Impossible to map '{value}' value to a ScalarKind")


class DataType(Type):
    ...

class ScalarType(DataType):
    kind: ScalarKind
    shape: Optional[list[int]] = None


class TupleType(DataType):
    types: list[DataType]


class FieldType(DataType):
    dims: list[Dimension] | Ellipsis
    dtype: ScalarType


class FunctionType(Type):
    args: list[DataType]
    kwargs: dict[str, DataType]
    returns: DataType


class LocatedNode(Node):
    location: SourceLocation


class SymbolName(eve.traits.SymbolName):
    regex = re.compile(r"^[a-zA-Z_][\w$]*$")


class Symbol(LocatedNode):
    id: SymbolName  # noqa: A003


class DataSymbol(Symbol):
    type: DataType


class Function(Symbol):
    # proposal:
    #
    # signature sub-symbols must be named specifically, example:
    # Function( # noqa
    #     id="my_field_op" # noqa
    #     returns=[ # noqa
    #        Field(id="my_field_op$return#0, ...), # noqa
    #        Field(id="my_field_op$return#1, ...), # noqa
    #     ], # noqa
    #     params=[ # noqa
    #         Field(id=my_field_op$param#inp1, ...), # noqa
    #         Field(id=my_field_op$param#inp2, ...), # noqa
    #         ..., # noqa
    #     ], # noqa
    # ) # noqa
    # That should make it possible to type check what is passed in and out
    type: FunctionType
    body: list[Stmt]


class Expr(LocatedNode):
    type: Optional[Type] = None


class Name(Expr):
    id: SymbolRef  # noqa: A003


# class Constant(Expr):
#     value: str
#     dtype: Name


class Subscript(Expr):
    value: Expr
    index: int


class TupleExpr(Expr):
    elts: list[Expr]


class UnaryOperator(StrEnum):
    UADD = "plus"
    USUB = "minus"
    NOT = "not_"


class UnaryOp(Expr):
    op: UnaryOperator
    operand: Expr


class BinaryOperator(StrEnum):
    ADD = "plus"
    SUB = "minus"
    MULT = "multiplies"
    DIV = "divides"
    BIT_AND = "and_"
    BIT_OR = "or_"


class BinOp(Expr):
    op: BinaryOperator
    left: Expr
    right: Expr


class CompareOperator(StrEnum):
    GT = "greater"
    LT = "less"
    EQ = "eq"


class Compare(Expr):
    op: CompareOperator
    left: Expr
    right: Expr


class Call(Expr):
    func: Name
    args: list[Expr]


class Stmt(LocatedNode):
    ...


class Assign(Stmt):
    target: Name
    value: Expr


class Return(Stmt):
    value: Expr


class FieldOperator(LocatedNode, SymbolTableTrait):
    id: SymbolName  # noqa: A003
    params: list[Field]
    body: list[Stmt]
    # externals: list[Symbol]  # noqa
