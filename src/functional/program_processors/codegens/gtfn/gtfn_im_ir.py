# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from typing import Union, List, Any

from eve import Coerced, SymbolName, Node
from functional.program_processors.codegens.gtfn.gtfn_ir import Sym, Expr, SymRef
from eve.traits import SymbolTableTrait


class Stmt(Node):
    ...


class AssignStmt(Stmt):
    op: str = "="
    lhs: Union[Sym, SymRef]  # TODO not sure what to use
    rhs: Union[Sym, Expr]


class InitStmt(AssignStmt):
    type: str = "auto"


class Conditional(Stmt):
    type: str
    init_stmt: Stmt
    cond: Union[SymRef, Expr]
    if_stmt: Stmt
    else_stmt: Stmt


class ReturnStmt(Stmt):
    ret: Union[Expr, Sym, SymRef]


class ImperativeFunctionDefinition(Node, SymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003
    params: List[Sym]
    fun: List[Stmt]
