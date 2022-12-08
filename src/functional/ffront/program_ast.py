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

from typing import Any, Generic, Literal, Optional, TypeVar, Union

import eve
from eve import Coerced, Node, SourceLocation, SymbolName, SymbolRef
from eve.traits import SymbolTableTrait
from functional.type_system import type_specifications as ts


class LocatedNode(Node):
    location: SourceLocation


SymbolT = TypeVar("SymbolT", bound=ts.TypeSpec)


class Symbol(eve.GenericNode, LocatedNode, Generic[SymbolT]):
    id: Coerced[SymbolName]  # noqa: A003
    type: Union[SymbolT, ts.DeferredType]  # noqa A003
    namespace: ts.Namespace = ts.Namespace(ts.Namespace.LOCAL)


DataTypeT = TypeVar("DataTypeT", bound=ts.DataType)
DataSymbol = Symbol[DataTypeT]

FieldTypeT = TypeVar("FieldTypeT", bound=ts.FieldType)
FieldSymbol = Symbol[FieldTypeT]

ScalarTypeT = TypeVar("ScalarTypeT", bound=ts.ScalarType)
ScalarSymbol = Symbol[ScalarTypeT]

TupleTypeT = TypeVar("TupleTypeT", bound=ts.TupleType)
TupleSymbol = Symbol[TupleTypeT]


class Expr(LocatedNode):
    type: Optional[ts.TypeSpec] = None  # noqa A003


class BinOp(Expr):
    op: ts.BinaryOperator
    left: Expr
    right: Expr


class Name(Expr):
    id: Coerced[SymbolRef]  # noqa: A003


class Call(Expr):
    func: Name
    args: list[Expr]
    kwargs: dict[str, Expr]


class Subscript(Expr):
    value: Name
    slice_: Expr


class TupleExpr(Expr):
    elts: list[Expr]


class Constant(Expr):
    value: Any  # TODO(tehrengruber): be more restrictive


class Dict(Expr):
    keys_: list[Name]
    values_: list[TupleExpr]


class Slice(Expr):
    lower: Optional[Constant]
    upper: Optional[Constant]
    step: Literal[None]


class Stmt(LocatedNode):
    ...


class Program(LocatedNode, SymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003
    type: Union[ts.ProgramType, ts.DeferredType]  # noqa A003
    params: list[DataSymbol]
    body: list[Call]
    closure_vars: list[Symbol]
