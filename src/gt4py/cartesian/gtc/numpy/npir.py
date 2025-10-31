# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, Tuple, Union

from gt4py import eve
from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.definitions import Extent
from gt4py.eve import datamodels


# --- Misc ---
class AxisName(eve.StrEnum):
    I = "I"  # noqa: E741 [ambiguous-variable-name]
    J = "J"
    K = "K"


# NOTE HorizontalMask in npir differs from common.HorizontalMask:
# - They are expressed relative to the iteration domain of the statement
# - Each axis is a tuple of two common.AxisBound instead of common.HorizontalInterval
class HorizontalMask(eve.Node):
    i: Tuple[common.AxisBound, common.AxisBound]
    j: Tuple[common.AxisBound, common.AxisBound]


# --- Decls ---
@eve.utils.noninstantiable
class Decl(eve.Node):
    name: eve.Coerced[eve.SymbolName]
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
    data_dims: Data dimensions sizes (static)
    dimensions: Mask which cartesian dimensions are present
    offset: Origin of the temporary field.
    padding: Buffer added to compute domain as field size.
    """

    data_dims: Tuple[int, ...] = eve.field(default_factory=tuple)
    dimensions: Tuple[bool, bool, bool]
    offset: Tuple[int, int]
    padding: Tuple[int, int]


# --- Expressions ---
@eve.utils.noninstantiable
class Expr(common.Expr):
    pass


@eve.utils.noninstantiable
class VectorLValue(Expr):
    pass


class ScalarLiteral(common.Literal, Expr):
    kind = common.ExprKind.SCALAR

    @datamodels.validator("dtype")
    def is_defined(self, attribute: datamodels.Attribute, dtype: common.DataType) -> None:
        undefined = [common.DataType.AUTO, common.DataType.DEFAULT, common.DataType.INVALID]
        if dtype in undefined:
            raise ValueError("npir.Literal may not have undefined data type.")


class ScalarCast(common.Cast[Expr], Expr):
    kind = common.ExprKind.SCALAR


class VectorCast(common.Cast[Expr], Expr):
    kind = common.ExprKind.FIELD


class Broadcast(Expr):
    expr: Expr
    kind: common.ExprKind = common.ExprKind.FIELD


class VarKOffset(common.VariableKOffset[Expr]):
    pass


class FieldSlice(VectorLValue):
    name: eve.Coerced[eve.SymbolRef]
    i_offset: int
    j_offset: int
    k_offset: Union[int, VarKOffset]
    data_index: List[Expr] = eve.field(default_factory=list)
    kind: common.ExprKind = common.ExprKind.FIELD

    @datamodels.validator("data_index")
    def data_indices_are_scalar(
        self, attribute: datamodels.Attribute, data_index: List[Expr]
    ) -> None:
        for index in data_index:
            if index.kind != common.ExprKind.SCALAR:
                raise ValueError("Data indices must be scalars")


class ParamAccess(Expr):
    name: eve.Coerced[eve.SymbolRef]
    kind: common.ExprKind = common.ExprKind.SCALAR


class LocalScalarAccess(VectorLValue):
    name: eve.Coerced[eve.SymbolRef]
    kind: common.ExprKind = common.ExprKind.FIELD


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
class Stmt(common.Stmt):
    pass


class VectorAssign(common.AssignStmt[VectorLValue, Expr], Stmt):
    # NOTE HorizontalMask in npir differs from common.HorizontalMask (see above)
    horizontal_mask: Optional[HorizontalMask] = None

    @datamodels.validator("right")
    def right_is_field_kind(self, attribute: datamodels.Attribute, right: Expr) -> None:
        if right.kind != common.ExprKind.FIELD:
            raise ValueError("right is not a common.ExprKind.FIELD")

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
