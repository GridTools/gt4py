# GTC Toolchain - GT4Py - GridTools Framework
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

from typing import List, Optional, Tuple, Union

from pydantic import validator

import eve
from gtc import common
from gtc.definitions import Extent


# --- Misc ---
class AxisName(eve.StrEnum):
    I = "I"  # noqa: E741 (ambiguous variable name)
    J = "J"
    K = "K"


# --- Decls ---
@eve.utils.noninstantiable
class Decl(eve.Node):
    name: eve.SymbolName
    dtype: common.DataType


class ScalarDecl(Decl):
    """Scalar per grid point.

    Used for API scalar parameters. Local scalars never have data_dims.

    """

    pass


class LocalScalarDecl(Decl):
    """Scalar per grid point.

    Used for API scalar parameters. Local scalars never have data_dims.

    """

    pass


class FieldDecl(Decl):
    """General field shared across HorizontalBlocks."""

    dimensions: Tuple[bool, bool, bool]
    data_dims: Tuple[int, ...] = eve.field(default_factory=tuple)
    extent: Extent


class TemporaryDecl(Decl):
    """
    Temporary field shared across HorizontalBlocks.

    Parameters
    ----------
    offset: Origin of the temporary field.
    padding: Buffer added to compute domain as field size.

    """

    data_dims: Tuple[int, ...] = eve.field(default_factory=tuple)
    offset: Tuple[int, int]
    padding: Tuple[int, int]


# --- Expressions ---
@eve.utils.noninstantiable
class Expr(common.Expr):
    pass


@eve.utils.noninstantiable
class VectorLValue(common.LocNode):
    pass


class ScalarLiteral(common.Literal, Expr):
    kind = common.ExprKind.SCALAR

    @validator("dtype")
    def is_defined(cls, dtype: common.DataType) -> common.DataType:
        undefined = [common.DataType.AUTO, common.DataType.DEFAULT, common.DataType.INVALID]
        if dtype in undefined:
            raise ValueError("npir.Literal may not have undefined data type.")
        return dtype


class ScalarCast(common.Cast[Expr], Expr):
    kind = common.ExprKind.SCALAR


class VectorCast(common.Cast[Expr], Expr):
    kind = common.ExprKind.FIELD


class Broadcast(Expr):
    expr: Expr
    kind = common.ExprKind.FIELD


class VarKOffset(common.VariableKOffset[Expr]):
    pass


class FieldSlice(Expr, VectorLValue):
    name: eve.SymbolRef
    i_offset: int
    j_offset: int
    k_offset: Union[int, VarKOffset]
    data_index: List[Expr] = []
    kind = common.ExprKind.FIELD

    @validator("data_index")
    def data_indices_are_scalar(cls, data_index: List[Expr]) -> List[Expr]:
        for index in data_index:
            if index.kind != common.ExprKind.SCALAR:
                raise ValueError("Data indices must be scalars")
        return data_index


class ParamAccess(Expr):
    name: eve.SymbolRef
    kind = common.ExprKind.SCALAR


class LocalScalarAccess(Expr, VectorLValue):
    name: eve.SymbolRef
    kind = common.ExprKind.FIELD


class VectorArithmetic(common.BinaryOp[Expr], Expr):
    op: Union[common.ArithmeticOperator, common.ComparisonOperator]

    _dtype_propagation = common.binary_op_dtype_propagation(strict=True)


class VectorLogic(common.BinaryOp[Expr], Expr):
    op: common.LogicalOperator


class VectorUnaryOp(common.UnaryOp[Expr], Expr):
    pass


class VectorTernaryOp(common.TernaryOp[Expr], Expr):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=True)


class NativeFuncCall(common.NativeFuncCall[Expr], Expr):
    _dtype_propagation = common.native_func_call_dtype_propagation(strict=True)


# --- Statements ---
@eve.utils.noninstantiable
class Stmt(eve.Node):
    pass


class VectorAssign(common.AssignStmt[VectorLValue, Expr], Stmt):
    left: VectorLValue
    right: Expr
    horizontal_mask: Optional[common.HorizontalMask] = None

    @validator("right")
    def right_is_field_kind(cls, right: Expr) -> Expr:
        if right.kind != common.ExprKind.FIELD:
            raise ValueError("right is not a common.ExprKind.FIELD")
        return right

    _dtype_validation = common.assign_stmt_dtype_validation(strict=True)


class While(common.While[Stmt, Expr], Stmt):
    pass


# --- Control Flow ---
class HorizontalBlock(common.LocNode, eve.SymbolTableTrait):
    body: List[Stmt]
    extent: Extent
    declarations: List[LocalScalarDecl]


class VerticalPass(common.LocNode):
    body: List[HorizontalBlock]
    lower: common.AxisBound
    upper: common.AxisBound
    direction: common.LoopOrder


class Computation(common.LocNode, eve.SymbolTableTrait):
    arguments: List[str]
    api_field_decls: List[FieldDecl]
    param_decls: List[ScalarDecl]
    temp_decls: List[TemporaryDecl]
    vertical_passes: List[VerticalPass]
