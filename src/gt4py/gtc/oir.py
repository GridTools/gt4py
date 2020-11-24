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

"""
Optimizable Intermediate Representation (working title)

OIR represents a computation at the level of GridTools stages and multistages,
e.g. stage merging, staged computations to compute-on-the-fly, cache annotations, etc.
"""

from typing import List, Optional, Union
from pydantic import validator

from devtools import debug  # noqa: F401
from eve import Node, SymbolName


from gt4py.gtc import common
from gt4py.gtc.common import LocNode


class Expr(common.Expr):
    dtype: common.DataType

    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args, **kwargs):
        if type(self) is Expr:
            raise TypeError("Trying to instantiate `Expr` abstract class.")
        super().__init__(*args, **kwargs)


class Stmt(common.Stmt):
    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args, **kwargs):
        if type(self) is Stmt:
            raise TypeError("Trying to instantiate `Stmt` abstract class.")
        super().__init__(*args, **kwargs)


class Literal(common.Literal, Expr):
    pass


class ScalarAccess(common.ScalarAccess, Expr):
    pass


class FieldAccess(common.FieldAccess, Expr):
    pass


class AssignStmt(common.AssignStmt[Union[ScalarAccess, FieldAccess], Expr], Stmt):
    @validator("left")
    def no_horizontal_offset_in_assignment(cls, v):
        if isinstance(v, FieldAccess) and (v.offset.i != 0 or v.offset.j != 0):
            raise ValueError("Lhs of assignment must not have a horizontal offset.")
        return v


# TODO
# class BlockStmt(common.BlockStmt[Stmt], Stmt):
#     pass


class IfStmt(common.IfStmt[List[Stmt], Expr], Stmt):  # TODO replace List[Stmt] by BlockStmt?
    pass


class UnaryOp(common.UnaryOp[Expr], Expr):
    pass


class BinaryOp(common.BinaryOp[Expr], Expr):
    pass


class TernaryOp(common.TernaryOp[Expr], Expr):
    pass


class Decl(LocNode):
    name: SymbolName
    dtype: common.DataType

    def __init__(self, *args, **kwargs):
        if type(self) is Decl:
            raise TypeError("Trying to instantiate `Decl` abstract class.")
        super().__init__(*args, **kwargs)


class FieldDecl(Decl):
    # TODO dimensions (or mask?)
    pass


class ScalarDecl(Decl):
    pass


# TODO move to common or do we need a different representation here?
class AxisBound(Node):
    level: common.LevelMarker
    offset: int = 0

    @classmethod
    def from_start(cls, offset: int):
        return cls(level=common.LevelMarker.START, offset=offset)

    @classmethod
    def from_end(cls, offset: int):
        return cls(level=common.LevelMarker.END, offset=offset)

    @classmethod
    def start(cls):
        return cls.from_start(0)

    @classmethod
    def end(cls):
        return cls.from_end(0)


class HorizontalExecution(LocNode):
    body: List[Stmt]
    mask: Optional[Expr]

    @validator("mask")
    def mask_is_boolean_field_expr(cls, v):
        if v:
            if v.dtype != common.DataType.BOOL:
                raise ValueError("Mask must be a boolean expression.")
            if v.kind != common.ExprKind.FIELD:
                raise ValueError("Mask must be a field expression.")
        return v


class Interval(LocNode):
    start: AxisBound
    end: AxisBound


class Temporary(LocNode):
    name: SymbolName
    dtype: common.DataType


class VerticalLoop(LocNode):
    interval: Interval
    horizontal_executions: List[HorizontalExecution]
    loop_order: common.LoopOrder
    declarations: List[Temporary]
    # caches: List[Union[IJCache,KCache]]


class Stencil(LocNode):
    # TODO remove `__main__.`` from name and use default SymbolName constraint
    name: SymbolName.constrained(r"[a-zA-Z_][\w\.]*")
    params: List[Decl]
    vertical_loops: List[VerticalLoop]
