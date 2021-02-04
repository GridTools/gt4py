# -*- coding: utf-8 -*-
from typing import Any, List, Optional, Tuple, cast

from pydantic import validator

import eve
from gtc import common


class Expr(common.Expr):
    # TODO: remove when abstract nodes implemented in eve
    def __init__(self, *args: Any, **kwargs: Any):
        if type(self) is Expr:
            raise TypeError("Cannot instantiate abstract Expr type of numpy IR")
        super().__init__(*args, **kwargs)


class Literal(common.Literal, Expr):
    kind = cast(common.ExprKind, common.ExprKind.SCALAR)
    dtype: common.DataType

    @validator("dtype")
    def is_defined(cls, dtype: common.DataType) -> common.DataType:
        undefined = [common.DataType.AUTO, common.DataType.DEFAULT, common.DataType.INVALID]
        if dtype in undefined:
            raise ValueError("npir.Literal may not have undefined data type.")
        return dtype


class Cast(common.Cast[Expr], Expr):
    dtype: common.DataType
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


class VectorExpression(Expr):
    # TODO: remove when abstract nodes implemented in eve
    kind = cast(common.ExprKind, common.ExprKind.FIELD)

    def __init__(self, *args: Any, **kwargs: Any):
        if type(self) is VectorExpression:
            raise TypeError("Cannot instantiate abstract VectorExpression type of numpy IR")
        super().__init__(*args, **kwargs)


class BroadCast(VectorExpression):
    expr: Expr


class VectorLValue(common.LocNode):
    pass


class FieldSlice(VectorExpression, VectorLValue):
    name: str
    i_offset: AxisOffset
    j_offset: AxisOffset
    k_offset: AxisOffset


class VectorTemp(VectorExpression, VectorLValue):
    name: common.SymbolRef


class VectorArithmetic(common.BinaryOp[VectorExpression], VectorExpression):
    pass


class VectorUnaryOp(common.UnaryOp[VectorExpression], VectorExpression):
    pass


class VectorAssign(common.AssignStmt[VectorLValue, VectorExpression], VectorExpression):
    left: VectorLValue
    right: VectorExpression
    mask: Optional[VectorExpression]


class VerticalPass(common.LocNode):
    body: List[VectorAssign]
    lower: common.AxisBound
    upper: common.AxisBound
    direction: common.LoopOrder


class DomainPadding(eve.Node):
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]


class Computation(common.LocNode):
    field_params: List[str]
    params: List[str]
    vertical_passes: List[VerticalPass]
    domain_padding: DomainPadding


class NativeFuncCall(common.NativeFuncCall[Expr], VectorExpression):
    _dtype_propagation = common.native_func_call_dtype_propagation(strict=True)
