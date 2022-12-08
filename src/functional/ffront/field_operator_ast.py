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

from typing import Any, Generic, TypeVar, Union, no_type_check

from eve import Coerced, Node, SourceLocation, SymbolName, SymbolRef, datamodels
from eve.traits import SymbolTableTrait
from eve.type_definitions import StrEnum
from functional.ffront import common_types as ct
from functional.utils import RecursionGuard


class LocatedNode(Node):
    location: SourceLocation

    def __str__(self):
        from functional.ffront.foast_pretty_printer import pretty_format

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


class Attribute(Expr):
    value: Expr
    attr: str


class TupleExpr(Expr):
    elts: list[Expr]


class UnaryOp(Expr):
    op: ct.UnaryOperator
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


class IfStmt(Stmt):
    condition: Expr
    true_branch: BlockStmt
    false_branch: BlockStmt

    @no_type_check
    @datamodels.root_validator
    def _collect_common_symbols(cls: type[IfStmt], instance: IfStmt) -> None:
        common_symbol_names = set(instance.true_branch.annex.symtable.keys()) & set(
            instance.false_branch.annex.symtable.keys()
        )
        instance.annex.propagated_symbols = {
            sym_name: Symbol(
                id=sym_name, type=ct.DeferredSymbolType(constraint=None), location=instance.location
            )
            for sym_name in common_symbol_names
        }


class FunctionDefinition(LocatedNode, SymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003  # shadowing a python builtin
    params: list[DataSymbol]
    body: BlockStmt
    closure_vars: list[Symbol]
    type: Union[ct.FunctionType, ct.DeferredSymbolType] = ct.DeferredSymbolType(  # noqa: A003
        constraint=ct.FunctionType
    )


class FieldOperator(LocatedNode, SymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003  # shadowing a python builtin
    definition: FunctionDefinition
    type: Union[ct.FieldOperatorType, ct.DeferredSymbolType] = ct.DeferredSymbolType(  # noqa: A003
        constraint=ct.FieldOperatorType
    )


class ScanOperator(LocatedNode, SymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003 # shadowing a python builtin
    axis: Constant
    forward: Constant
    init: Constant
    definition: FunctionDefinition  # scan pass
    type: Union[ct.ScanOperatorType, ct.DeferredSymbolType] = ct.DeferredSymbolType(  # noqa: A003
        constraint=ct.ScanOperatorType
    )
