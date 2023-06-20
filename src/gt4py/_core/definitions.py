# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import dataclasses
import enum
import functools
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from gt4py.eve.extended_typing import (
    Final,
    Generic,
    Literal,
    Mapping,
    Type,
    TypeVar,
    Union,
    TypeAlias,
)

# Scalar types supported by GT4Py
bool_ = np.bool_

int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64

uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64

float32 = np.float32
float64 = np.float64

# TODO(egparedes): add support for complex numbers (complex64, complex128)

BOOL_TYPES: Final[tuple[type, ...]] = (bool, bool_)
ScalarBoolType: TypeAlias = Union[bool, bool_]

SINT_TYPES: Final[tuple[type, ...]] = (int8, int16, int32, int64, int)
ScalarSignedIntType: TypeAlias = Union[int8, int16, int32, int64, int]

UINT_TYPES: Final[tuple[type, ...]] = (uint8, uint16, uint32, uint64)
ScalarUnsignedIntType: TypeAlias = Union[uint8, uint16, uint32, uint64]

INT_TYPES: Final[tuple[type, ...]] = (*SINT_TYPES, *UINT_TYPES)
ScalarIntType: TypeAlias = Union[ScalarSignedIntType, ScalarUnsignedIntType]

FLOAT_TYPES: Final[tuple[type, ...]] = (float32, float64, float)
ScalarFloatType: TypeAlias = Union[float32, float64, float]


SCALAR_TYPES: Final[tuple[type, ...]] = (*BOOL_TYPES, *INT_TYPES, *FLOAT_TYPES)
#: Type alias for all scalar types supported by GT4Py
ScalarType: TypeAlias = Union[ScalarBoolType, ScalarIntType, ScalarFloatType]

_SC = TypeVar("_SC", bound=ScalarType)


class DeviceType(enum.IntEnum):
    """The type of the device where a memory buffer is allocated."""

    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10


@dataclasses.dataclass(frozen=True, slots=True)
class Device:
    """
    Representation of a computing device.

    This definition is based on the DLPack device definition. A device is
    described by a pair of `DeviceType` and `device_id`. The `device_id`
    is an integer that is interpreted differently depending on the
    `DeviceType`. For example, for `DeviceType.CPU` it could be the CPU
    core number, for `DeviceType.CUDA` it could be the CUDA device number, etc.
    """

    device_type: DeviceType
    device_id: int

    def __iter__(self) -> Iterable[int]:
        return iter((self.device_type, self.device_id))


class DTypeCode(int, enum.Enum):
    """
    Kind of a specific data type.

    Actual values taken from DLPack reference implementation at:
    https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
    """

    INT = 0
    UINT = 1
    FLOAT = 2
    OPAQUE_POINTER = 3
    COMPLEX = 5  # bfloat16 is defined as BFLOAT = 4 in DLPack


_DTYPECODE_TO_NUMPY_KIND: Final[Mapping[DTypeCode, Literal["i", "u", "f", "c", "V"]]] = {
    DTypeCode.INT: "i",
    DTypeCode.UINT: "u",
    DTypeCode.FLOAT: "f",
    DTypeCode.OPAQUE_POINTER: "V",
    DTypeCode.COMPLEX: "c",
}

_NUMPY_KIND_TO_DTYPECODE: Final[Mapping[Literal["i", "u", "f", "c", "V"], DTypeCode]] = {
    kind: code for code, kind in _DTYPECODE_TO_NUMPY_KIND.items()
}


@dataclasses.dataclass(frozen=True)
class DType(Generic[_SC]):
    """
    Descriptor of data type for field elements.

    This definition is based on DLPack and Array API standards. The data
    type is described by a name and a triple, `DTypeCode`, `bits`, and
    `lanes`, which should be interpreted as packed `lanes` repetitions
    of elements from `type_code` data-category of width `bits`.

    Note:
        Array API standard only requires DTypes to be comparable with `__eq__`.
    """

    name: str
    type_code: DTypeCode
    bits: int
    lanes: int

    def __post_init__(self) -> None:
        if self.bits not in (1, 8, 16, 32, 64):
            raise ValueError(f"Non byte-sized dtypes (bits={self.bits}) are not supported")

    @classmethod
    def from_np_dtype(cls, dtype: np.dtype) -> DType[_SC]:
        bits = dtype.itemsize * 8
        type_code = _NUMPY_KIND_TO_DTYPECODE.get(dtype.kind, None)
        if type_code is None:
            raise ValueError(f"NumPy dtype {dtype} cannot be converted to DType")

        return cls(name=dtype.name, type_code=type_code, bits=dtype.itemsize * 8, lanes=1)

    @functools.cached_property
    def np_dtype(self) -> np.dtype:
        if self.bits % 8 != 0:
            raise RuntimeError(f"NumPy dtype is not supported {self.bits} bits types")
        spec = f"{_DTYPECODE_TO_NUMPY_KIND[self.type_code]}{self.bits // 8}"
        if self.lanes > 1:
            spec = f"({self.lanes},){spec}"
        return np.dtype(spec)

    @functools.cached_property
    def byte_size(self) -> int:
        return (self.bits // 8) * self.lanes

    #: For compatibility with NumPy
    @property
    def dtype(self) -> np.dtype:
        return self.np_dtype

    @functools.cached_property
    def scalar_type(self) -> Type[_SC]:
        return self.np_dtype.type

    def __int__(self) -> int:
        return self.type_code.value * 100000 + self.lanes * 1000 + self.bits

    def __str__(self) -> str:
        return f"<{self.name}>"


DTypeLike = Union[DType, npt.DTypeLike]

BOOL_DTYPE: Final[DType[bool_]] = DType("bool", DTypeCode.UINT, 1, 1)
INT8_DTYPE: Final[DType[int8]] = DType("int8", DTypeCode.INT, 8, 1)
INT16_DTYPE: Final[DType[int16]] = DType("int16", DTypeCode.INT, 16, 1)
INT32_DTYPE: Final[DType[int32]] = DType("int32", DTypeCode.INT, 32, 1)
INT64_DTYPE: Final[DType[int64]] = DType("int64", DTypeCode.INT, 64, 1)
UINT8_DTYPE: Final[DType[uint8]] = DType("uint8", DTypeCode.UINT, 8, 1)
UINT16_DTYPE: Final[DType[uint16]] = DType("uint16", DTypeCode.UINT, 16, 1)
UINT32_DTYPE: Final[DType[uint32]] = DType("uint32", DTypeCode.UINT, 32, 1)
UINT64_DTYPE: Final[DType[uint64]] = DType("uint64", DTypeCode.UINT, 64, 1)
FLOAT32_DTYPE: Final[DType[float32]] = DType("float32", DTypeCode.FLOAT, 32, 1)
FLOAT64_DTYPE: Final[DType[float64]] = DType("float64", DTypeCode.FLOAT, 64, 1)
# TODO(egparedes): add support for complex types
# TODO(egparedes):  complex64: Final[DType[np.complex64]] = DType("complex64", DTypeCode.COMPLEX, 64, 1)
# TODO(egparedes):  complex128: Final[DType[np.complex128]] = DType("complex128", DTypeCode.COMPLEX, 128, 1)
