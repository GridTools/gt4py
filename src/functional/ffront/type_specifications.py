from dataclasses import dataclass

from functional import common as func_common
from functional.type_system.type_specifications import (
    TypeSpec,
    DataType,
    CallableType,
    DeferredType,
    VoidType,
    DimensionType,
    OffsetType,
    ScalarKind,
    ScalarType,
    FieldType,
    TupleType,
    CallableType,
    FunctionType
)


@dataclass(frozen=True)
class ProgramType(TypeSpec, CallableType):
    definition: FunctionType


@dataclass(frozen=True)
class FieldOperatorType(TypeSpec, CallableType):
    definition: FunctionType


@dataclass(frozen=True)
class ScanOperatorType(TypeSpec, CallableType):
    axis: func_common.Dimension
    definition: FunctionType
