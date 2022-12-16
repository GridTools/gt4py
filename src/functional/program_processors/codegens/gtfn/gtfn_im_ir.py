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

from typing import Union, List, Tuple

from eve import Coerced, SymbolName, Node
import functional.program_processors.codegens.gtfn.gtfn_ir as gtfn_ir
from eve.traits import SymbolTableTrait


class Stmt(Node):
    ...


class AssignStmt(Stmt):
    op: str = "="
    lhs: Union[gtfn_ir.Sym, gtfn_ir.SymRef]
    rhs: Union[gtfn_ir.SymRef, gtfn_ir.Expr]


class InitStmt(AssignStmt):
    type: str = "auto"


class Conditional(Stmt):
    type: str
    init_stmt: Stmt
    cond: Union[gtfn_ir.SymRef, gtfn_ir.Expr]
    if_stmt: Stmt
    else_stmt: Stmt


class ForLoop(Stmt):
    num_iter: int
    stmt: Stmt


class ReturnStmt(Stmt):
    ret: Union[gtfn_ir.Expr, gtfn_ir.SymRef]


class ImperativeFunctionDefinition(Node, SymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003
    params: List[gtfn_ir.Sym]
    fun: List[Stmt]
