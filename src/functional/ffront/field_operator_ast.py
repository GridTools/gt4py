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


import re

import eve
from eve import Node
from eve.type_definitions import SourceLocation, StrEnum, SymbolRef


class LocatedNode(Node):
    location: SourceLocation


class SymbolName(eve.traits.SymbolName):
    regex = re.compile(r"^[a-zA-Z_][\w$]*$")


class Sym(LocatedNode):
    id: SymbolName  # noqa: A003


class Expr(LocatedNode):
    ...


class SymRef(Expr):
    id: SymbolRef  # noqa: A003


class Name(Expr):
    id: SymbolName  # noqa: A003


class Subscript(Expr):
    value: Expr
    index: int


class Tuple(Expr):
    elts: list[Expr]


class UnaryOperator(StrEnum):
    PLUS = "plus"
    MINUS = "minus"


class UnaryOp(Expr):
    op: UnaryOperator
    operand: Expr


class BinaryOperator(StrEnum):
    PLUS = "plus"
    MINUS = "minus"
    MULT = "mult"
    DIV = "div"
    MOD = "mod"
    POW = "pow"


class BinOp(Expr):
    op: BinaryOperator
    left: Expr
    righ: Expr


class Stmt(LocatedNode):
    ...


class Assign(Stmt):
    target: Name
    value: Expr


class Return(Stmt):
    value: Expr


class FieldOperator(LocatedNode):
    id: SymbolName  # noqa: A003
    params: list[Sym]
    body: list[Stmt]
