# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Final, Iterator, Optional, Sequence, TypeVar

from gt4py.eve import (
    datamodels as eve_datamodels,
    extended_typing as xtyping,
    type_definitions as eve_types,
)
from gt4py.next import common


class TypeSpec(eve_datamodels.DataModel, kw_only=False, frozen=True): ...  # type: ignore[call-arg]


class DataType(TypeSpec):
    """
    A base type for all types that represent data storage.

    Derive floating point, integral or field types from this class.
    """


class CallableType(TypeSpec):
    """
    A base type for all types are callable.

    Derive other callable types, such as FunctionType or FieldOperatorType from
    this class.
    """


class DeferredType(TypeSpec):
    """Dummy used to represent a type not yet inferred."""

    constraint: Optional[type[TypeSpec] | tuple[type[TypeSpec], ...]]


class VoidType(TypeSpec):
    """
    Return type of a function without return values.

    Note: only useful for stateful dialects.
    """


class DimensionType(TypeSpec):
    dim: common.Dimension

    def __str__(self) -> str:
        return str(self.dim)


class IndexType(TypeSpec):
    """
    Represents the type of an index into a dimension.
    """

    dim: common.Dimension

    def __str__(self) -> str:
        return f"Index[{self.dim}]"


class OffsetType(TypeSpec):
    # TODO(havogt): replace by ConnectivityType
    source: common.Dimension
    target: tuple[common.Dimension] | tuple[common.Dimension, common.Dimension]

    def __str__(self) -> str:
        return f"Offset[{self.source}, {self.target}]"


class ScalarKind(eve_types.IntEnum):
    BOOL = 1
    INT8 = 2
    UINT8 = 3
    INT16 = 4
    UINT16 = 5
    INT32 = 6
    UINT32 = 7
    INT64 = 8
    UINT64 = 9
    FLOAT32 = 10
    FLOAT64 = 11
    STRING = 12


class ScalarType(DataType):
    kind: ScalarKind
    shape: Optional[list[int]] = None

    def __str__(self) -> str:
        kind_str = self.kind.name.lower()
        if self.shape is None:
            return kind_str
        return f"{kind_str}{self.shape}"


class ListType(DataType):
    """Represents a neighbor list in the ITIR representation.

    Note:
      - not used in the frontend. The concept is represented as Field with local Dimension.
      - `None` is used to describe lists originating from `make_const_list`.
    """

    element_type: DataType
    offset_type: common.Dimension | None


class FieldType(DataType, CallableType):
    dims: list[common.Dimension]
    dtype: ScalarType | ListType

    def __str__(self) -> str:
        dims = "..." if self.dims is Ellipsis else f"[{', '.join(dim.value for dim in self.dims)}]"
        return f"Field[{dims}, {self.dtype}]"

    @eve_datamodels.validator("dims")
    def _dims_validator(
        self, attribute: eve_datamodels.Attribute, dims: list[common.Dimension]
    ) -> None:
        common.check_dims(dims)


class TupleType(DataType):
    # TODO(tehrengruber): Remove `DeferredType` again. This was erroneously
    #  introduced before we checked the annotations at runtime. All attributes of
    #  a type that are types themselves must be concrete.
    types: list[DataType | DimensionType | DeferredType]

    def __str__(self) -> str:
        return f"tuple[{', '.join(map(str, self.types))}]"

    def __iter__(self) -> Iterator[DataType | DimensionType | DeferredType]:
        yield from self.types

    def __len__(self) -> int:
        return len(self.types)


class NamedCollectionType(DataType):
    types: list[DataType | DimensionType | DeferredType]
    keys: list[str]
    #: The original python type. It should be only used in the boundaries between
    #: Python and GT4Py DSL, that is, `type translation` and in constructor/extractor
    #: steps for custom containers.
    #: It uses the "entry-point"-like format required by `pkgutil.resolve_name()`:
    #:   '__module__:__qualname__'
    original_python_type: (
        str  # Format: '__module__:__qualname__' (as required by `pkgutil.resolve_name()`)
    )

    def __getattr__(self, name: str) -> DataType | DimensionType | DeferredType:
        keys = object.__getattribute__(self, "keys")
        if name in keys:
            return self.types[keys.index(name)]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __str__(self) -> str:
        return f"NamedTuple{{{', '.join(f'{k}: {v}' for k, v in zip(self.keys, self.types))}}}"

    def __iter__(self) -> Iterator[DataType | DimensionType | DeferredType]:
        # Note: Unlike `Mapping`s, we iterate the values (not the keys) by default.
        yield from self.types

    def __len__(self) -> int:
        return len(self.types)


CollectionTypeSpecT = TypeVar("CollectionTypeSpecT", TupleType, NamedCollectionType)
CollectionTypeSpec = TupleType | NamedCollectionType
COLLECTION_TYPE_SPECS: Final[tuple[type[CollectionTypeSpec], ...]] = xtyping.get_args(
    CollectionTypeSpec
)


class FunctionType(CallableType):
    pos_only_args: Sequence[TypeSpec]
    pos_or_kw_args: dict[str, TypeSpec]
    kw_only_args: dict[str, TypeSpec]
    returns: TypeSpec

    def __str__(self) -> str:
        arg_strs = [str(arg) for arg in self.pos_only_args]
        kwarg_strs = [f"{key}: {value}" for key, value in self.pos_or_kw_args.items()]
        args_str = ", ".join((*arg_strs, *kwarg_strs))
        return f"({args_str}) -> {self.returns}"


class ConstructorType(CallableType):
    definition: FunctionType

    @property
    def constructed_type(self) -> TypeSpec:
        return self.definition.returns


class DomainType(DataType):
    dims: list[common.Dimension]
