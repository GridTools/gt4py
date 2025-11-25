# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Generic, Literal, Optional, TypeVar, Union

import gt4py.eve as eve
from gt4py.eve import Coerced, Node, SourceLocation, SymbolName, SymbolRef, datamodels
from gt4py.eve.traits import SymbolTableTrait
from gt4py.next.ffront import dialect_ast_enums, type_specifications as ts_ffront
from gt4py.next.type_system import type_specifications as ts


class LocatedNode(Node):
    location: SourceLocation


SymbolT = TypeVar("SymbolT", bound=ts.TypeSpec)


class Symbol(eve.GenericNode, LocatedNode, Generic[SymbolT]):
    id: Coerced[SymbolName]
    type: Union[SymbolT, ts.DeferredType]  # A003
    namespace: dialect_ast_enums.Namespace = dialect_ast_enums.Namespace(
        dialect_ast_enums.Namespace.LOCAL
    )


DataTypeT = TypeVar("DataTypeT", bound=ts.DataType)
DataSymbol = Symbol[DataTypeT]

FieldTypeT = TypeVar("FieldTypeT", bound=ts.FieldType)
FieldSymbol = Symbol[FieldTypeT]

ScalarTypeT = TypeVar("ScalarTypeT", bound=ts.ScalarType)
ScalarSymbol = Symbol[ScalarTypeT]

TupleTypeT = TypeVar("TupleTypeT", bound=ts.TupleType)
TupleSymbol = Symbol[TupleTypeT]


class Expr(LocatedNode):
    type: Optional[ts.TypeSpec] = None  # A003


class BinOp(Expr):
    op: dialect_ast_enums.BinaryOperator
    left: Expr
    right: Expr


class Name(Expr):
    id: Coerced[SymbolRef]


class Call(Expr):
    func: Name
    args: list[Expr]
    kwargs: dict[str, Expr]


class Subscript(Expr):
    value: Name
    slice_: Expr


class TupleExpr(Expr):
    elts: list[Expr]


class Attribute(Expr):
    attr: str
    value: Expr


class Constant(Expr):
    value: Any  # TODO(tehrengruber): be more restrictive


class Dict(Expr):
    keys_: list[Union[Name | Attribute]]
    values_: list[TupleExpr]

    @datamodels.root_validator
    @classmethod
    def keys_values_length_validation(cls: type["Dict"], instance: "Dict") -> None:
        if len(instance.keys_) != len(instance.values_):
            raise ValueError("`Dict` must have same number of keys as values.")


class Slice(Expr):
    lower: Optional[Constant]
    upper: Optional[Constant]
    step: Literal[None]


class Stmt(LocatedNode): ...


class Program(LocatedNode, SymbolTableTrait):
    id: Coerced[SymbolName]
    type: Union[ts_ffront.ProgramType, ts.DeferredType]  # A003
    params: list[DataSymbol]
    body: list[Call]
    closure_vars: list[Symbol]
