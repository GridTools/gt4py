# GT4Py - GridTools Framework
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

from typing import List, Union

from gt4py.eve import Coerced, Node, SymbolName
from gt4py.eve.traits import SymbolTableTrait
from gt4py.next.program_processors.codegens.gtfn.gtfn_ir_common import Expr, Sym, SymRef


class Stmt(Node):
    ...


class AssignStmt(Stmt):
    op: str = "="
    lhs: Union[Sym, SymRef]
    rhs: Expr


class InitStmt(AssignStmt):
    init_type: str = "auto"


class EmptyListInitializer(Expr):
    ...


class Conditional(Stmt):
    cond_type: str
    init_stmt: InitStmt
    cond: Expr
    if_stmt: AssignStmt
    else_stmt: AssignStmt


class ReturnStmt(Stmt):
    ret: Expr


class ImperativeFunctionDefinition(Node, SymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003
    params: List[Sym]
    fun: List[Stmt]
