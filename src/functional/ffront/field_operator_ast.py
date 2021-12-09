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
import typing
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt

import eve
from eve import Node
from eve.traits import SymbolTableTrait
from eve.type_definitions import IntEnum, SourceLocation, StrEnum, SymbolRef
from functional import common


def make_scalar_kind_from_value(value: npt.DTypeLike) -> ScalarKind:
    try:
        dt = np.dtype(value)
    except TypeError as err:
        raise common.GTTypeError(f"Invalid scalar type definition ({value})") from err

    if dt.shape == () and dt.fields is None:
        match dt:
            case np.bool_:
                return ScalarKind.BOOL
            case np.int32:
                return ScalarKind.INT32
            case np.int64:
                return ScalarKind.INT64
            case np.float32:
                return ScalarKind.FLOAT32
            case np.float64:
                return ScalarKind.FLOAT64
            case _:
                raise common.GTTypeError(f"Impossible to map '{value}' value to a ScalarKind")
    else:
        raise common.GTTypeError(f"Non-trivial dtypes like '{value}' are not yet supported")


def make_symbol_type_from_value(value: Any) -> SymbolType:
    match value:
        case bool() | int() | float() | np.generic():
            return make_symbol_type_from_value(type(value))
        case type() as t if issubclass(t, (bool, int, float, np.generic)):
            return ScalarType(kind=make_scalar_kind_from_value(value))
        case tuple() as tuple_value:
            return TupleType(types=[make_symbol_type_from_value(t) for t in tuple_value])
        case types.FunctionType():
            # TODO (egparedes): recover the function signature from FieldOperator
            args = []
            kwargs = {}
            returns = make_symbol_type_from_value(float)
            return FunctionType(args=args, kwargs=kwargs, returns=returns)
        case other if other.__module__ == "typing":
            return make_symbol_type_from_value(other.__origin__)
        case _:
            raise common.GTTypeError(f"Impossible to map '{value}' value to a SymbolType")


def make_symbol_from_value(
    name: str, value: Any, namespace: Namespace, location: SourceLocation
) -> Symbol:
    symbol_type = make_symbol_type_from_value(value)
    match symbol_type:
        case ScalarType() | TupleType():
            return DataSymbol(id=name, type=symbol_type, namespace=namespace, location=location)
        case FunctionType():
            return Function(
                id=name,
                type=symbol_type,
                namespace=namespace,
                params=[],
                returns=[],
                location=location,
            )
        case _:
            raise common.GTTypeError(f"Impossible to map '{value}' value to a Symbol")


class Namespace(StrEnum):
    LOCAL = "local"
    CLOSURE = "closure"
    EXTERNAL = "external"


class ScalarKind(IntEnum):
    BOOL = 1
    INT32 = 32
    INT64 = 64
    FLOAT32 = 1032
    FLOAT64 = 1064

    from_value = staticmethod(make_scalar_kind_from_value)


class Dimension(Node):
    name: str


class SymbolType(Node):
    ...

    from_value = staticmethod(make_symbol_type_from_value)


class DeferredSymbolType(SymbolType):
    constraint: typing.Optional[typing.Type[SymbolType]]


class SymbolTypeVariable(SymbolType):
    id: str  # noqa A003
    bound: typing.Type[SymbolType]


class DataType(SymbolType):
    ...


class ScalarType(DataType):
    kind: ScalarKind
    shape: Optional[list[int]] = None


class TupleType(DataType):
    types: list[DataType]


class FieldType(DataType):
    dims: Union[list[Dimension], Literal[Ellipsis]]  # type: ignore[valid-type,misc]
    dtype: ScalarType


class FunctionType(SymbolType):
    args: list[DataType]
    kwargs: dict[str, DataType]
    returns: DataType


class LocatedNode(Node):
    location: SourceLocation


class SymbolName(eve.traits.SymbolName):
    regex = re.compile(r"^[a-zA-Z_][\w$]*$")


class Symbol(LocatedNode):
    id: SymbolName  # noqa: A003
    type: SymbolType  # noqa A003
    namespace: Namespace = Namespace.LOCAL

    from_value = staticmethod(make_symbol_from_value)


class DataSymbol(Symbol):
    type: Union[DataType, DeferredSymbolType]  # noqa A003


class FieldSymbol(DataSymbol):
    type: Union[FieldType, DeferredSymbolType]  # noqa A003


class Function(Symbol):
    type: FunctionType  # noqa A003
    params: list[FieldType]
    returns: list[FieldType]


class Expr(LocatedNode):
    type: Optional[SymbolType] = None  # noqa A003


class Name(Expr):
    id: SymbolRef  # noqa: A003


class Constant(Expr):
    value: str
    dtype: Union[DataType, str]


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


class ExternalImport(Stmt):
    symbols: list[Symbol]


class Assign(Stmt):
    target: Symbol
    value: Expr


class Return(Stmt):
    value: Expr


class FieldOperator(LocatedNode, SymbolTableTrait):
    id: SymbolName  # noqa: A003
    params: list[DataSymbol]
    body: list[Stmt]
    closure: list[Symbol]
