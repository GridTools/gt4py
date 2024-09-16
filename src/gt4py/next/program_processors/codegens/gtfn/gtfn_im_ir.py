# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import List, Union

from gt4py.eve import Coerced, Node, SymbolName
from gt4py.eve.traits import SymbolTableTrait
from gt4py.next.program_processors.codegens.gtfn.gtfn_ir_common import Expr, Sym, SymRef


class Stmt(Node): ...


class AssignStmt(Stmt):
    op: str = "="
    lhs: Union[Sym, SymRef]
    rhs: Expr


class InitStmt(AssignStmt):
    init_type: str = "auto"


class EmptyListInitializer(Expr): ...


class Conditional(Stmt):
    cond_type: str
    init_stmt: InitStmt
    cond: Expr
    if_stmt: AssignStmt
    else_stmt: AssignStmt


class ReturnStmt(Stmt):
    ret: Expr


class ImperativeFunctionDefinition(Node, SymbolTableTrait):
    id: Coerced[SymbolName]
    params: List[Sym]
    fun: List[Stmt]
