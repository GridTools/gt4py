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
from eve.type_definitions import SymbolRef


class SymbolName(eve.traits.SymbolName):
    regex = re.compile(r"^[a-zA-Z_][\w$]*$")


class Sym(Node):
    id: SymbolName  # noqa: A003


class Expr(Node):
    ...


class SymExpr(Expr):
    id: SymbolName  # noqa: A003
    expr: Expr


class SymRef(Expr):
    id: SymbolRef  # noqa: A003


class Return(Expr):
    value: Expr


class Name(Expr):
    id: SymbolName  # noqa: A003


class Subscript(Expr):
    expr: Expr
    index: int


class Tuple(Expr):
    elts: list[Expr]


class FieldOperator(Node):
    id: SymbolName  # noqa: A003
    params: list[Sym]
    body: list[Expr]
