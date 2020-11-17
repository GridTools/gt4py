from __future__ import annotations

from typing import List, Tuple, Union

import eve
import numpy
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


class Offset(eve.Node):
    offset: NumericalOffset
    axis_name: str

    @classmethod
    def from_int(cls, *, axis_name: str, offset: int) -> Offset:
        return cls(axis_name=axis_name, offset=NumericalOffset(value=offset))

    @classmethod
    def k(cls, offset: int) -> Offset:
        return cls.from_int(axis_name="k", offset=offset)


class ParallelOffset(Offset):
    @classmethod
    def i(cls, offset: int) -> ParallelOffset:
        return cls.from_int(axis_name="i", offset=offset)

    @classmethod
    def j(cls, offset: int) -> ParallelOffset:
        return cls.from_int(axis_name="j", offset=offset)


class SequentialOffset(Offset):
    pass


class VectorExpression(gtir.LocNode):
    pass


class FieldSlice(VectorExpression):
    name: str
    i_offset: ParallelOffset
    j_offset: ParallelOffset
    k_offset: Offset


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
    i: Tuple[int, int]
    j: Tuple[int, int]
    k: Tuple[int, int]


class Computation(gtir.LocNode):
    field_params: List[str]
    scalar_params: List[str]
    vertical_passes: List[VerticalPass]
    domain_padding: DomainPadding
