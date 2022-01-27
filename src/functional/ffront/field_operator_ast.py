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
import typing
from typing import Literal, Optional, Union

import eve
from eve import Node
from eve.traits import SymbolTableTrait
from eve.type_definitions import IntEnum, SourceLocation, StrEnum, SymbolRef


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


class Dimension(Node):
    name: str


class SymbolType(Node):
    ...


class DeferredSymbolType(SymbolType):
    constraint: typing.Optional[typing.Type[SymbolType]]


class SymbolTypeVariable(SymbolType):
    id: str  # noqa A003
    bound: typing.Type[SymbolType]


class OffsetType(SymbolType):
    ...

    def __str__(self):
        return f"Offset[{self.id}]"


class DataType(SymbolType):
    ...


class ScalarType(DataType):
    kind: ScalarKind
    shape: Optional[list[int]] = None

    def __str__(self):
        kind_str = self.kind.name.lower()
        if self.shape is None:
            return kind_str
        return f"{kind_str}{self.shape}"


class TupleType(DataType):
    types: list[DataType]

    def __str__(self):
        return f"tuple{self.types}"


class FieldType(DataType):
    dims: Union[list[Dimension], Literal[Ellipsis]]  # type: ignore[valid-type,misc]
    dtype: ScalarType

    def __str__(self):
        dims = "..." if self.dims is Ellipsis else f"[{', '.join(dim.name for dim in self.dims)}]"
        return f"Field[{dims}, dtype={self.dtype}]"


class FunctionType(SymbolType):
    args: list[DataType]
    kwargs: dict[str, DataType]
    returns: DataType

    def __str__(self):
        arg_strs = [str(arg) for arg in self.args]
        kwarg_strs = [f"{key}: {value}" for key, value in self.kwargs]
        args_str = ", ".join(*arg_strs, *kwarg_strs)
        return f"({args_str}) -> {self.returns}"


class LocatedNode(Node):
    location: SourceLocation


class SymbolName(eve.traits.SymbolName):
    regex = re.compile(r"^[a-zA-Z_][\w$]*$")


class Symbol(LocatedNode):
    id: SymbolName  # noqa: A003
    type: SymbolType  # noqa A003
    namespace: Namespace = Namespace(Namespace.LOCAL)


class DataSymbol(Symbol):
    type: Union[DataType, DeferredSymbolType]  # noqa A003


class FieldSymbol(DataSymbol):
    type: Union[FieldType, DeferredSymbolType]  # noqa A003


class TupleSymbol(DataSymbol):
    type: Union[TupleType, DeferredSymbolType]  # noqa A003


class Function(Symbol):
    type: FunctionType  # noqa A003


class OffsetSymbol(Symbol):
    type: OffsetType  # noqa A003


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
    target: Symbol
    value: Expr


class Return(Stmt):
    value: Expr


class FieldOperator(LocatedNode, SymbolTableTrait):
    id: SymbolName  # noqa: A003
    params: list[DataSymbol]
    body: list[Stmt]
    closure: list[Symbol]
