from typing import List, Tuple

import eve
from pydantic import validator

from gt4py.gtc import common, gtir


class Literal(gtir.Literal):
    @validator("dtype")
    def is_defined(cls, dtype):
        undefined = [common.DataType.AUTO, common.DataType.DEFAULT, common.DataType.INVALID]
        if dtype in undefined:
            raise ValueError("npir.Literal may not have undefined data type.")
        return dtype


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
    def i(cls, offset: int, *, parallel=True) -> "AxisOffset":
        return cls.from_int(axis_name=AxisName.I, offset=offset, parallel=parallel)

    @classmethod
    def j(cls, offset: int, *, parallel=True) -> "AxisOffset":
        return cls.from_int(axis_name=AxisName.J, offset=offset, parallel=parallel)

    @classmethod
    def k(cls, offset: int, *, parallel=False) -> "AxisOffset":
        return cls.from_int(axis_name=AxisName.K, offset=offset, parallel=parallel)


class VectorExpression(gtir.LocNode):
    pass


class FieldSlice(VectorExpression):
    name: str
    i_offset: AxisOffset
    j_offset: AxisOffset
    k_offset: AxisOffset


class VectorArithmetic(VectorExpression):
    left: VectorExpression
    right: VectorExpression
    op: common.ArithmeticOperator


class VectorAssign(VectorExpression):
    left: FieldSlice
    right: VectorExpression


class VerticalPass(gtir.LocNode):
    body: List[VectorAssign]
    lower: NumericalOffset
    upper: NumericalOffset
    direction: common.LoopOrder


class DomainPadding(eve.Node):
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]


class Computation(gtir.LocNode):
    field_params: List[str]
    scalar_params: List[str]
    vertical_passes: List[VerticalPass]
    domain_padding: DomainPadding
