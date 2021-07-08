# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py - GridTools Framework
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

from typing import List, Optional, Tuple, Union, cast

from pydantic import validator

import eve
from gtc import common


@eve.utils.noninstantiable
class Expr(common.Expr):
    pass


class Literal(common.Literal, Expr):
    @validator("dtype")
    def is_defined(cls, dtype: common.DataType) -> common.DataType:
        undefined = [common.DataType.AUTO, common.DataType.DEFAULT, common.DataType.INVALID]
        if dtype in undefined:
            raise ValueError("npir.Literal may not have undefined data type.")
        return dtype


class Cast(common.Cast[Expr], Expr):
    pass


class NumericalOffset(eve.Node):
    value: int


class AxisName(eve.StrEnum):
    I = "I"  # noqa: E741 (ambiguous variable name)
    J = "J"
    K = "K"


class AxisOffset(eve.Node):
    offset: NumericalOffset
    axis_name: AxisName
    parallel: bool

    @classmethod
    def from_int(cls, *, axis_name: str, offset: int, parallel: bool) -> "AxisOffset":
        return cls(axis_name=axis_name, offset=NumericalOffset(value=offset), parallel=parallel)

    @classmethod
    def i(cls, offset: int, *, parallel: bool = True) -> "AxisOffset":
        return cls.from_int(axis_name=AxisName.I, offset=offset, parallel=parallel)

    @classmethod
    def j(cls, offset: int, *, parallel: bool = True) -> "AxisOffset":
        return cls.from_int(axis_name=AxisName.J, offset=offset, parallel=parallel)

    @classmethod
    def k(cls, offset: int, *, parallel: bool = False) -> "AxisOffset":
        return cls.from_int(axis_name=AxisName.K, offset=offset, parallel=parallel)


@eve.utils.noninstantiable
class VectorExpression(Expr):
    kind = cast(common.ExprKind, common.ExprKind.FIELD)


class BroadCast(VectorExpression):
    expr: Expr
    dims: int = 3


class VectorLValue(common.LocNode):
    pass


class FieldDecl(eve.Node):
    name: eve.SymbolName
    dtype: common.DataType
    dimensions: Tuple[bool, bool, bool]
    data_dims: Tuple[int, ...] = eve.field(default_factory=tuple)


class FieldSlice(VectorExpression, VectorLValue):
    name: str
    i_offset: Optional[AxisOffset] = None
    j_offset: Optional[AxisOffset] = None
    k_offset: Optional[AxisOffset] = None


class NamedScalar(common.ScalarAccess, Expr):
    pass


class VectorTemp(VectorExpression, VectorLValue):
    name: common.SymbolRef


class EmptyTemp(VectorExpression):
    dtype: common.DataType


class VectorArithmetic(common.BinaryOp[VectorExpression], VectorExpression):
    op: Union[common.ArithmeticOperator, common.ComparisonOperator]


class VectorLogic(common.BinaryOp[VectorExpression], VectorExpression):
    op: common.LogicalOperator


class VectorUnaryOp(common.UnaryOp[VectorExpression], VectorExpression):
    pass


class VectorTernaryOp(common.TernaryOp[VectorExpression], VectorExpression):
    pass


class VectorAssign(common.AssignStmt[VectorLValue, VectorExpression], VectorExpression):
    left: VectorLValue
    right: VectorExpression
    mask: Optional[VectorExpression]


class MaskBlock(common.Stmt):
    mask: VectorExpression
    mask_name: str
    body: List[VectorAssign]


class HorizontalBlock(common.LocNode):
    body: List[Union[VectorAssign, MaskBlock]]


class VerticalPass(common.LocNode):
    body: List[HorizontalBlock]
    temp_defs: List[VectorAssign]
    lower: common.AxisBound
    upper: common.AxisBound
    direction: common.LoopOrder


class Computation(common.LocNode, eve.SymbolTableTrait):
    field_decls: List[FieldDecl]
    field_params: List[str]
    params: List[str]
    vertical_passes: List[VerticalPass]


class NativeFuncCall(common.NativeFuncCall[Expr], VectorExpression):
    _dtype_propagation = common.native_func_call_dtype_propagation(strict=True)
