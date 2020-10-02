# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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

from typing import List, Optional

from devtools import debug  # noqa: F401
from eve import Node, SourceLocation, Str

from . import common


class LocNode(Node):
    loc: Optional[SourceLocation]


class Expr(LocNode):
    pass


class Stmt(LocNode):
    pass


class Literal(Expr):
    value: Str
    vtype: common.DataType


class Domain(LocNode):
    pass


class CartesianOffset(Node):
    i: int
    j: int
    k: int

    @classmethod
    def zero(cls):
        return cls(i=0, j=0, k=0)


class FieldAccess(Expr):
    name: Str  # via symbol table
    offset: CartesianOffset

    @classmethod
    def centered(cls, *, name, loc=None):
        return cls(name=name, loc=loc, offset=CartesianOffset.zero())


class AssignStmt(Stmt):
    left: FieldAccess  # there are no local variables in gtir, only fields
    right: Expr


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr


class FieldDecl(LocNode):
    name: Str
    dtype: common.DataType


class HorizontalLoop(LocNode):
    stmt: Stmt


class AxisBound(Node):
    level: common.LevelMarker
    offset: int = 0


class VerticalInterval(LocNode):
    horizontal_loops: List[HorizontalLoop]
    start: AxisBound
    end: AxisBound


class VerticalLoop(LocNode):
    # each statement inside a `with location_type` is interpreted as a
    # full horizontal loop (see parallel model of SIR)
    vertical_intervals: List[VerticalInterval]
    loop_order: common.LoopOrder


class Stencil(LocNode):
    vertical_loops: List[VerticalLoop]


class Computation(LocNode):
    name: Str
    params: List[FieldDecl]
    stencils: List[Stencil]
