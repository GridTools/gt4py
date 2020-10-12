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

import enum
from typing import Dict, List, Optional, Tuple

from devtools import debug  # noqa: F401
from eve import IntEnum, Node, SourceLocation, Str

from . import common


class LocNode(Node):
    loc: Optional[SourceLocation]


class Expr(LocNode):
    pass


class Stmt(LocNode):
    pass


class Literal(Expr):
    # TODO when coming from python AST we know more than just the string representation, I suppose
    value: Str
    dtype: common.DataType


class Domain(LocNode):
    pass


class CartesianOffset(Node):
    i: int
    j: int
    k: int

    @classmethod
    def zero(cls):
        return cls(i=0, j=0, k=0)

    def to_tuple(self):
        return (self.i, self.j, self.k)


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
    vertical_intervals: List[VerticalInterval]
    loop_order: common.LoopOrder


class Stencil(LocNode):
    vertical_loops: List[VerticalLoop]


@enum.unique
class AccessKind(IntEnum):
    READ_ONLY = 0
    READ_WRITE = 1


class FieldBoundary(Node):
    i: Tuple[int, int]
    j: Tuple[int, int]
    k: Tuple[int, int]

    def to_tuple(self):
        return (self.i, self.j, self.k)

    def set_at(self, index, value):
        if index == 0:
            self.i = value
        elif index == 1:
            self.j = value
        elif index == 2:
            self.k = value
        else:
            raise IndexError()

    def update_from_offset(self, offset: CartesianOffset):
        for i, (bound_n, offset_n) in enumerate(zip(self.to_tuple(), offset.to_tuple())):
            bound_n = list(bound_n)
            sign = -1 if offset_n < 0 else 1
            start_or_end = 0 if offset_n < 0 else 1
            bound_n[start_or_end] = max(sign * offset_n, bound_n[start_or_end])
            self.set_at(i, tuple(bound_n))


class FieldMetadata(Node):
    name: str
    access: AccessKind
    boundary: FieldBoundary
    dtype: Optional[common.DataType]


class FieldsMetadata(Node):
    metas: Dict[str, FieldMetadata] = {}


class Computation(LocNode):
    name: Str
    params: List[FieldDecl]
    stencils: List[Stencil]
    fields_metadata: FieldsMetadata = FieldsMetadata()

    @property
    def param_names(self) -> List:
        return [p.name for p in self.params]
