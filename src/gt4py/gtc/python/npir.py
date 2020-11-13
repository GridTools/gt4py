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


class ParallelOffset(eve.Node):
    offset: int
    sign: str
    axis_name: str


class SequentialOffset(eve.Node):
    offset: int
    sign: str
    axis_name: str


class VerticalPass(gtir.LocNode):
    body: List[gtir.Expr]
    lower: gtir.AxisBound
    upper: gtir.AxisBound
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
