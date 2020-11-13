from typing import List

import eve

from gt4py.gtc import common, gtir


class VerticalPass(gtir.LocNode):
    body: List[NumpyExpression]
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
