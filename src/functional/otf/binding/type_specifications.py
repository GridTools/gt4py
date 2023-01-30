from dataclasses import dataclass

import functional.type_system.type_specifications as ts
from functional import common as func_common


@dataclass(frozen=True)
class IndexFieldType(ts.DataType):
    axis: func_common.Dimension
    dtype: ts.ScalarType

    def __str__(self):
        return f"IndexField[{self.axis.value}, {self.dtype}]"
