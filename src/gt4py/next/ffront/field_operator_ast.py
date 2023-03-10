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

from typing import Any, Generic, TypeVar, Union

from gt4py.eve import Coerced, Node, SourceLocation, SymbolName, SymbolRef
from gt4py.eve.traits import SymbolTableTrait
from gt4py.eve.type_definitions import StrEnum
from gt4py.next.ffront import dialect_ast_enums, type_specifications as ts_ffront
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.utils import RecursionGuard


class LocatedNode(Node):
    location: SourceLocation

    def __str__(self):
        from gt4py.next.ffront.foast_pretty_printer import pretty_format

        try:
            with RecursionGuard(self):
                return pretty_format(self)
        except RecursionGuard.RecursionDetected:
            # If `pretty_format` itself calls `__str__`, i.e. when printing
            # an error that happend during formatting, just return the regular
            # string representation, consequently avoiding an infinite recursion.
            # Note that avoiding circular calls to `__str__` is not possible
            # as the pretty printer depends on eve utilities that use the string
            # representation of a node in the error handling.
            return super().__str__()


SymbolT = TypeVar("SymbolT", bound=ts.TypeSpec)


# TODO(egparedes): this should be an actual generic datamodel but it is not fully working
#   due to nested specialization with bound typevars, so disabling specialization for now
#       class Symbol(eve.GenericNode, LocatedNode, Generic[SymbolT]):
#
class Symbol(LocatedNode, Generic[SymbolT]):
    id: Coerced[SymbolName]  # noqa: A003  # shadowing a python builtin
    type: Union[SymbolT, ts.DeferredType]  # noqa A003
    namespace: dialect_ast_enums.Namespace = dialect_ast_enums.Namespace(
        dialect_ast_enums.Namespace.LOCAL
    )


DataTypeT = TypeVar("DataTypeT", bound=ts.DataType)
DataSymbol = Symbol[DataTypeT]

FieldTypeT = TypeVar("FieldTypeT", bound=ts.FieldType)
FieldSymbol = DataSymbol[FieldTypeT]

ScalarTypeT = TypeVar("ScalarTypeT", bound=ts.ScalarType)
ScalarSymbol = DataSymbol[ScalarTypeT]

TupleTypeT = TypeVar("TupleTypeT", bound=ts.TupleType)
TupleSymbol = DataSymbol[TupleTypeT]

DimensionTypeT = TypeVar("DimensionTypeT", bound=ts.DimensionType)
DimensionSymbol = DataSymbol[DimensionTypeT]


class Expr(LocatedNode):
    type: ts.TypeSpec = ts.DeferredType(constraint=None)  # noqa A003


class Name(Expr):
    id: Coerced[SymbolRef]  # noqa: A003  # shadowing a python builtin


class Constant(Expr):
    value: Any  # TODO: be more specific


class Subscript(Expr):
    value: Expr
    index: int


class Attribute(Expr):
    value: Expr
    attr: str


class TupleExpr(Expr):
    elts: list[Expr]


class UnaryOp(Expr):
    op: dialect_ast_enums.UnaryOperator
    operand: Expr


class BinOp(Expr):
    op: dialect_ast_enums.BinaryOperator
    left: Expr
    right: Expr


class CompareOperator(StrEnum):
    EQ = "eq"
    NOTEQ = "not_eq"
    LT = "less"
    LTE = "less_equal"
    GT = "greater"
    GTE = "greater_equal"

    def __str__(self) -> str:
        if self is self.EQ:
            return "=="
        elif self is self.NOTEQ:
            return "!="
        elif self is self.LT:
            return "<"
        elif self is self.LTE:
            return "<="
        elif self is self.GT:
            return ">"
        elif self is self.GTE:
            return ">="
        return "Unknown CompareOperator"


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


class Starred(Expr):
    id: Union[FieldSymbol, TupleSymbol, ScalarSymbol]  # noqa: A003  # shadowing a python builtin


class Assign(Stmt):
    target: Union[FieldSymbol, TupleSymbol, ScalarSymbol]
    value: Expr


class TupleTargetAssign(Stmt):
    targets: list[FieldSymbol | TupleSymbol | ScalarSymbol | Starred]
    value: Expr


class Return(Stmt):
    value: Expr


class BlockStmt(Stmt, SymbolTableTrait):
    stmts: list[Stmt]


class FunctionDefinition(LocatedNode, SymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003  # shadowing a python builtin
    params: list[DataSymbol]
    body: BlockStmt
    closure_vars: list[Symbol]
    type: Union[ts.FunctionType, ts.DeferredType] = ts.DeferredType(  # noqa: A003
        constraint=ts.FunctionType
    )


class FieldOperator(LocatedNode, SymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003  # shadowing a python builtin
    definition: FunctionDefinition
    type: Union[ts_ffront.FieldOperatorType, ts.DeferredType] = ts.DeferredType(  # noqa: A003
        constraint=ts_ffront.FieldOperatorType
    )


class ScanOperator(LocatedNode, SymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003 # shadowing a python builtin
    axis: Constant
    forward: Constant
    init: Constant
    definition: FunctionDefinition  # scan pass
    type: Union[ts_ffront.ScanOperatorType, ts.DeferredType] = ts.DeferredType(  # noqa: A003
        constraint=ts_ffront.ScanOperatorType
    )
