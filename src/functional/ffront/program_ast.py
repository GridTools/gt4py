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
from eve import Node
from eve.traits import SymbolTableTrait
from eve.type_definitions import SourceLocation, SymbolName, SymbolRef
from functional.ffront import common_types


class LocatedNode(Node):
    location: SourceLocation


SymbolT = TypeVar("SymbolT", bound=common_types.SymbolType)


class Symbol(eve.GenericNode, LocatedNode, Generic[SymbolT]):
    id: SymbolName  # noqa: A003
    type: Union[SymbolT, common_types.DeferredSymbolType]  # noqa A003
    namespace: common_types.Namespace = common_types.Namespace(common_types.Namespace.LOCAL)


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


class Slice(Expr):
    lower: Optional[Constant]
    upper: Optional[Constant]
    step: Literal[None]


class Stmt(LocatedNode):
    ...


class Program(LocatedNode, SymbolTableTrait):
    id: SymbolName  # noqa: A003
    params: list[DataSymbol]
    body: list[Call]
    captured_vars: list[Symbol]
