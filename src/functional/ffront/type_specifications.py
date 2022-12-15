from dataclasses import dataclass

from functional import common as func_common
from functional.type_system.type_specifications import (
    CallableType,
    DataType,
    DeferredType,
    DimensionType,
    FieldType,
    FunctionType,
    OffsetType,
    ScalarKind,
    ScalarType,
    TupleType,
    TypeSpec,
    VoidType,
)


__all__ = [
    "CallableType",
    "DataType",
    "DeferredType",
    "DimensionType",
    "FieldType",
    "FunctionType",
    "OffsetType",
    "ScalarKind",
    "ScalarType",
    "TupleType",
    "TypeSpec",
    "VoidType",
]


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
