# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Iterator, Optional, Sequence, Union

from gt4py.eve import datamodels as eve_datamodels, type_definitions as eve_types
from gt4py.next import common


class TypeSpec(eve_datamodels.DataModel, kw_only=False, frozen=True): ...  # type: ignore[call-arg]


class DataType(TypeSpec):
    """
    A base type for all types that represent data storage.

    Derive floating point, integral or field types from this class.
    """


class CallableType:
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

    Note: not used in the frontend. The concept is represented as Field with local Dimension.
    """

    element_type: DataType
    # TODO(tehrengruber): make `offset_type` mandatory
    offset_type: Optional[common.Dimension] = None


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


class FunctionType(TypeSpec, CallableType):
    pos_only_args: Sequence[TypeSpec]
    pos_or_kw_args: dict[str, TypeSpec]
    kw_only_args: dict[str, TypeSpec]
    returns: Union[TypeSpec]

    def __str__(self) -> str:
        arg_strs = [str(arg) for arg in self.pos_only_args]
        kwarg_strs = [f"{key}: {value}" for key, value in self.pos_or_kw_args.items()]
        args_str = ", ".join((*arg_strs, *kwarg_strs))
        return f"({args_str}) -> {self.returns}"
