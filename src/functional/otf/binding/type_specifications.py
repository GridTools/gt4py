from dataclasses import dataclass

import functional.type_system.type_specifications as ts
from functional import common as func_common
from functional.type_system.type_specifications import (  # noqa: F401
    CallableType as CallableType,
    DataType as DataType,
    DeferredType as DeferredType,
    DimensionType as DimensionType,
    FieldType as FieldType,
    FunctionType as FunctionType,
    OffsetType as OffsetType,
    ScalarKind as ScalarKind,
    ScalarType as ScalarType,
    TupleType as TupleType,
    TypeSpec as TypeSpec,
    VoidType as VoidType,
)


@dataclass(frozen=True)
class IndexFieldType(ts.DataType):
    axis: func_common.Dimension
    dtype: ts.ScalarType

    def __str__(self):
        return f"IndexField[{self.axis.value}, {self.dtype}]"
