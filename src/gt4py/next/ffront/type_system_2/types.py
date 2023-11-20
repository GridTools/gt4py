from gt4py.next.type_system_2 import types as ts2
from typing import Optional
from gt4py.next import common as func_common


class FieldOperatorType(ts2.Type):
    arguments: list[ts2.FunctionArgument]
    result: Optional[ts2.Type]

    def __init__(self, arguments: list[ts2.FunctionArgument], result: Optional[ts2.Type]):
        self.arguments = arguments
        self.result = result


class ScanOperatorType(ts2.Type):
    dimension: func_common.Dimension
    carry: Optional[ts2.Type]
    arguments: list[ts2.FunctionArgument]
    result: Optional[ts2.Type]

    def __init__(
            self,
            dimension: func_common.Dimension,
            carry: Optional[ts2.Type],
            arguments: list[ts2.FunctionArgument],
            result: Optional[ts2.Type]
    ):
        self.dimension = dimension
        self.carry = carry
        self.arguments = arguments
        self.result = result
