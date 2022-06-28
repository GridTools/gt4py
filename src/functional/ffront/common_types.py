from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from eve.type_definitions import IntEnum, StrEnum
from functional import common as func_common


class ScalarKind(IntEnum):
    BOOL = 1
    INT32 = 32
    INT64 = 64
    # Python's "int" type in the Python AST should be mapped to ScalarKind.INT in our ASTs. The size, as
    # determined by numpy, varies by platform. (Size is the same as C's "long" type.)
    INT = INT32 if np.int_ == np.int32 else INT64
    FLOAT32 = 1032
    FLOAT64 = 1064
    DIMENSION = 2001
    STRING = 3001


class Namespace(StrEnum):
    LOCAL = "local"
    CLOSURE = "closure"
    EXTERNAL = "external"


class SymbolType:
    pass


@dataclass(frozen=True)
class DeferredSymbolType(SymbolType):
    """Dummy used to represent a type not yet inferred."""

    constraint: Optional[type[SymbolType] | tuple[type[SymbolType], ...]]


@dataclass(frozen=True)
class SymbolTypeVariable(SymbolType):
    id: str  # noqa A003
    bound: type[SymbolType]


@dataclass(frozen=True)
class VoidType(SymbolType):
    """
    Return type of a function without return values.

    Note: only useful for stateful dialects.
    """


@dataclass(frozen=True)
class DimensionType(SymbolType):
    dim: func_common.Dimension


@dataclass(frozen=True)
class OffsetType(SymbolType):
    source: Optional[func_common.Dimension] = None
    target: Optional[tuple[func_common.Dimension, ...]] = None

    def __str__(self):
        return f"Offset[{self.source}, {self.target}]"


@dataclass(frozen=True)
class DataType(SymbolType):
    ...


@dataclass(frozen=True)
class ScalarType(DataType):
    kind: ScalarKind
    shape: Optional[list[int]] = None

    def __str__(self):
        kind_str = self.kind.name.lower()
        if self.shape is None:
            return kind_str
        return f"{kind_str}{self.shape}"


@dataclass(frozen=True)
class TupleType(DataType):
    types: list[DataType]

    def __str__(self):
        return f"tuple{self.types}"


@dataclass(frozen=True)
class FieldType(DataType):
    dims: list[func_common.Dimension] | Literal[Ellipsis]  # type: ignore[valid-type,misc]
    dtype: ScalarType

    def __str__(self):
        dims = "..." if self.dims is Ellipsis else f"[{', '.join(dim.value for dim in self.dims)}]"
        return f"Field[{dims}, dtype={self.dtype}]"


@dataclass(frozen=True)
class FunctionType(SymbolType):
    args: list[DataType | DeferredSymbolType]
    kwargs: dict[str, DataType | DeferredSymbolType]
    returns: DataType | DeferredSymbolType | VoidType

    def __str__(self):
        arg_strs = [str(arg) for arg in self.args]
        kwarg_strs = [f"{key}: {value}" for key, value in self.kwargs.items()]
        args_str = ", ".join((*arg_strs, *kwarg_strs))
        return f"({args_str}) -> {self.returns}"
