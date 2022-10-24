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


from typing import Any, Generic, Optional, TypeVar, Union

from eve import Coerced, Node, SourceLocation, SymbolName, SymbolRef
from eve.traits import SymbolTableTrait
from eve.type_definitions import StrEnum
from functional.ffront import common_types as ct


class LocatedNode(Node):
    location: SourceLocation


SymbolT = TypeVar("SymbolT", bound=ct.SymbolType)


# TODO(egparedes): this should be an actual generic datamodel but it is not fully working
#   due to nested specialization with bound typevars, so disabling specialization for now
#       class Symbol(eve.GenericNode, LocatedNode, Generic[SymbolT]):
#
class Symbol(LocatedNode, Generic[SymbolT]):
    id: Coerced[SymbolName]  # noqa: A003  # shadowing a python builtin
    type: Union[SymbolT, ct.DeferredSymbolType]  # noqa A003
    namespace: ct.Namespace = ct.Namespace(ct.Namespace.LOCAL)


DataTypeT = TypeVar("DataTypeT", bound=ct.DataType)
DataSymbol = Symbol[DataTypeT]

FieldTypeT = TypeVar("FieldTypeT", bound=ct.FieldType)
FieldSymbol = DataSymbol[FieldTypeT]

ScalarTypeT = TypeVar("ScalarTypeT", bound=ct.ScalarType)
ScalarSymbol = DataSymbol[ScalarTypeT]

TupleTypeT = TypeVar("TupleTypeT", bound=ct.TupleType)
TupleSymbol = DataSymbol[TupleTypeT]

DimensionTypeT = TypeVar("DimensionTypeT", bound=ct.DimensionType)
DimensionSymbol = DataSymbol[DimensionTypeT]


class Expr(LocatedNode):
    type: ct.SymbolType = ct.DeferredSymbolType(constraint=None)  # noqa A003


class Name(Expr):
    id: Coerced[SymbolRef]  # noqa: A003  # shadowing a python builtin


class Constant(Expr):
    value: Any  # TODO: be more specific


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


class BinOp(Expr):
    op: ct.BinaryOperator
    left: Expr
    right: Expr


class CompareOperator(StrEnum):
    EQ = "eq"
    NOTEQ = "not_eq"
    LT = "less"
    LTE = "less_equal"
    GT = "greater"
    GTE = "greater_equal"


class Compare(Expr):
    op: CompareOperator
    left: Expr
    right: Expr


class TernaryExpr(Expr):
    condition: Expr
    true_expr: Expr
    false_expr: Expr


class Call(Expr):
    func: Name
    args: list[Expr]
    kwargs: dict[str, Expr]


class Stmt(LocatedNode):
    ...


class Assign(Stmt):
    target: Union[FieldSymbol, TupleSymbol, ScalarSymbol]
    value: Expr


class Return(Stmt):
    value: Expr


class FunctionDefinition(LocatedNode, SymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003  # shadowing a python builtin
    params: list[DataSymbol]
    body: list[Stmt]
    closure_vars: list[Symbol]
    type: Optional[ct.FunctionType] = None  # noqa A003  # shadowing a python builtin


class FieldOperator(LocatedNode, SymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003  # shadowing a python builtin
    definition: FunctionDefinition
    type: Optional[ct.FieldOperatorType] = None  # noqa A003  # shadowing a python builtin


class ScanOperator(LocatedNode, SymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003 # shadowing a python builtin
    axis: Constant
    forward: Constant
    init: Constant
    definition: FunctionDefinition  # scan pass
    type: Optional[ct.ScanOperatorType] = None  # noqa A003 # shadowing a python builtin
