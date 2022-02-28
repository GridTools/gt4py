import typing
from dataclasses import dataclass
from typing import Literal, Optional, Union

from eve.type_definitions import IntEnum, StrEnum
from functional import common as func_common


class ScalarKind(IntEnum):
    BOOL = 1
    INT32 = 32
    INT64 = 64
    FLOAT32 = 1032
    FLOAT64 = 1064


class Namespace(StrEnum):
    LOCAL = "local"
    CLOSURE = "closure"
    EXTERNAL = "external"


class SymbolType:
    @classmethod
    def validate(cls, v):
        if not isinstance(v, cls):
            raise TypeError(f"Value is not a valid `{cls}`")
        return v

    @classmethod
    def __get_validators__(cls):
        yield cls.validate


@dataclass(frozen=True)
class DeferredSymbolType(SymbolType):
    """Dummy used to represent a type not yet inferred."""

    constraint: typing.Optional[typing.Type[SymbolType]]


@dataclass(frozen=True)
class SymbolTypeVariable(SymbolType):
    id: str  # noqa A003
    bound: typing.Type[SymbolType]


@dataclass(frozen=True)
class VoidType(SymbolType):
    """
    Return type of a function without return values.

    Note: only useful for stateful dialects.
    """

    ...


@dataclass(frozen=True)
class OffsetType(SymbolType):
    ...

    def __str__(self):
        return f"Offset[{self.id}]"


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
    dims: Union[list[func_common.Dimension], Literal[Ellipsis]]  # type: ignore[valid-type,misc]
    dtype: ScalarType

    def __str__(self):
        dims = "..." if self.dims is Ellipsis else f"[{', '.join(dim.value for dim in self.dims)}]"
        return f"Field[{dims}, dtype={self.dtype}]"


@dataclass(frozen=True)
class FunctionType(SymbolType):
    args: list[DataType]
    kwargs: dict[str, DataType]
    returns: Union[DataType, VoidType]

    def __str__(self):
        arg_strs = [str(arg) for arg in self.args]
        kwarg_strs = [f"{key}: {value}" for key, value in self.kwargs.items()]
        args_str = ", ".join((*arg_strs, *kwarg_strs))
        return f"({args_str}) -> {self.returns}"
