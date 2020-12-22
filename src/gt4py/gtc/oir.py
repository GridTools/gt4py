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

from eve import Str, SymbolName, SymbolTableTrait
from pydantic import validator

from gt4py.gtc import common
from gt4py.gtc.common import AxisBound, LocNode


class Expr(common.Expr):
    dtype: Optional[common.DataType]

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

# TODO
# class IfStmt(common.IfStmt[List[Stmt], Expr], Stmt):  # TODO replace List[Stmt] by BlockStmt?
#     pass


class UnaryOp(common.UnaryOp[Expr], Expr):
    pass


class BinaryOp(common.BinaryOp[Expr], Expr):
    _dtype_propagation = common.binary_op_dtype_propagation(strict=True)


class TernaryOp(common.TernaryOp[Expr], Expr):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=True)


class Cast(common.Cast[Expr], Expr):
    pass


class NativeFuncCall(common.NativeFuncCall[Expr], Expr):
    _dtype_propagation = common.native_func_call_dtype_propagation(strict=True)


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


class Temporary(FieldDecl):
    pass


class HorizontalExecution(LocNode):
    body: List[Stmt]
    mask: Optional[Expr]

    @validator("mask")
    def mask_is_boolean_field_expr(cls, v):
        if v:
            if v.dtype != common.DataType.BOOL:
                raise ValueError("Mask must be a boolean expression.")
        return v


class Interval(LocNode):
    start: AxisBound
    end: AxisBound


class VerticalLoop(LocNode):
    interval: Interval
    horizontal_executions: List[HorizontalExecution]
    loop_order: common.LoopOrder
    declarations: List[Temporary]
    # caches: List[Union[IJCache,KCache]]


class Stencil(LocNode, SymbolTableTrait):
    name: Str
    params: List[Decl]
    vertical_loops: List[VerticalLoop]

    _validate_dtype_is_set = common.validate_dtype_is_set()
    _validate_symbol_refs = common.validate_symbol_refs()
