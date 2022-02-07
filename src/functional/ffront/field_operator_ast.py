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


import re
from typing import Generic, Optional, TypeVar, Union

import eve
from eve import Node
from eve.traits import SymbolTableTrait
from eve.type_definitions import SourceLocation, StrEnum, SymbolRef
from functional.ffront import common_types as common_types


class Namespace(StrEnum):
    LOCAL = "local"
    CLOSURE = "closure"
    EXTERNAL = "external"


class LocatedNode(Node):
    location: SourceLocation


class SymbolName(eve.traits.SymbolName):
    regex = re.compile(r"^[a-zA-Z_][\w$]*$")


SymbolT = TypeVar("SymbolT", bound=common_types.SymbolType)


class Symbol(eve.GenericNode, LocatedNode, Generic[SymbolT]):
    id: SymbolName  # noqa: A003
    type: Union[SymbolT, common_types.DeferredSymbolType]  # noqa A003
    namespace: Namespace = Namespace(Namespace.LOCAL)


DataTypeT = TypeVar("DataTypeT", bound=common_types.DataType)
DataSymbol = Symbol[DataTypeT]

FieldTypeT = TypeVar("FieldTypeT", bound=common_types.FieldType)
FieldSymbol = Symbol[FieldTypeT]

ScalarTypeT = TypeVar("ScalarTypeT", bound=common_types.ScalarType)
ScalarSymbol = Symbol[ScalarTypeT]

TupleTypeT = TypeVar("TupleTypeT", bound=common_types.TupleType)
TupleSymbol = Symbol[TupleTypeT]


class Expr(LocatedNode):
    type: Optional[common_types.SymbolType] = None  # noqa A003


class Name(Expr):
    id: SymbolRef  # noqa: A003


class Constant(Expr):
    value: str
    dtype: Union[common_types.DataType, str]


class Subscript(Expr):
    value: Expr
    index: int


class TupleExpr(Expr):
    elts: list[Expr]


class UnaryOperator(StrEnum):
    UADD = "plus"
    USUB = "minus"
    NOT = "not_"

    def __str__(self) -> str:
        if self is self.UADD:
            return "+"
        elif self is self.USUB:
            return "-"
        elif self is self.NOT:
            return "not"
        return "Unknown UnaryOperator"


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

    def __str__(self) -> str:
        if self is self.ADD:
            return "+"
        elif self is self.SUB:
            return "-"
        elif self is self.MULT:
            return "*"
        elif self is self.DIV:
            return "/"
        elif self is self.BIT_AND:
            return "&"
        elif self is self.BIT_OR:
            return "|"
        return "Unknown BinaryOperator"


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


class Shift(Expr):
    offsets: list[Subscript]
    expr: Expr


class Stmt(LocatedNode):
    ...


class ExternalImport(Stmt):
    symbols: list[Symbol]


class Assign(Stmt):
    target: Union[FieldSymbol, TupleSymbol]
    value: Expr


class Return(Stmt):
    value: Expr


class FieldOperator(LocatedNode, SymbolTableTrait):
    id: SymbolName  # noqa: A003
    params: list[DataSymbol]
    body: list[Stmt]
    closure: list[Symbol]
