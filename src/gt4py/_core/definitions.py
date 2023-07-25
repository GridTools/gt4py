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
import numbers
from typing import (
    Any,
    Final,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
    TYPE_CHECKING,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import cupy as cp

    CuPyNDArray = cp.ndarray

    import jax.numpy as jnp

    JaxNDArray = jnp.ndarray


class DeviceType(enum.Enum):
    """The type of the device where a memory buffer is allocated.

    Enum values taken from DLPack reference implementation at:
    https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
    """

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

BoolScalar: TypeAlias = Union[bool_, bool]
BoolT = TypeVar("BoolT", bool_, bool)
BOOL_TYPES: Final[tuple[type, ...]] = cast(tuple[type, ...], BoolScalar.__args__)  # type: ignore[attr-defined]

IntScalar: TypeAlias = Union[int8, int16, int32, int64, int]
IntT = TypeVar("IntT", int8, int16, int32, int64, int)
INT_TYPES: Final[tuple[type, ...]] = cast(tuple[type, ...], IntScalar.__args__)  # type: ignore[attr-defined]

UnsignedIntScalar: TypeAlias = Union[uint8, uint16, uint32, uint64]
UnsignedIntT = TypeVar("UnsignedIntT", uint8, uint16, uint32, uint64)
UINT_TYPES: Final[tuple[type, ...]] = cast(tuple[type, ...], UnsignedIntScalar.__args__)  # type: ignore[attr-defined]

IntegralScalar: TypeAlias = Union[IntScalar, UnsignedIntScalar]
IntegralT = TypeVar("IntegralT", IntScalar, UnsignedIntScalar)
INTEGRAL_TYPES: Final[tuple[type, ...]] = (*INT_TYPES, *UINT_TYPES)

FloatingScalar: TypeAlias = Union[float32, float64, float]
FloatingT = TypeVar("FloatingT", float32, float64, float)
FLOAT_TYPES: Final[tuple[type, ...]] = cast(tuple[type, ...], FloatingScalar.__args__)  # type: ignore[attr-defined]

#: Type alias for all scalar types supported by GT4Py
Scalar: TypeAlias = Union[BoolScalar, IntegralScalar, FloatingScalar]
ScalarT = TypeVar("ScalarT", BoolScalar, IntegralScalar, FloatingScalar)
SCALAR_TYPES: Final[tuple[type, ...]] = (*BOOL_TYPES, *INTEGRAL_TYPES, *FLOAT_TYPES)


class BooleanIntegral(numbers.Integral):
    """Abstract base class for boolean integral types."""

    ...


class UnsignedIntegral(numbers.Integral):
    """Abstract base class for unsigned integral types."""

    ...


def is_boolean_integral_type(integral_type: type) -> TypeGuard[type[BooleanIntegral]]:
    return issubclass(integral_type, BOOL_TYPES)


def is_unsigned_integral_type(integral_type: type) -> TypeGuard[type[UnsignedIntegral]]:
    return issubclass(integral_type, UINT_TYPES)


class DTypeKind(enum.Enum):
    """
    Kind of a specific data type.

    Character codes match the type code for the corresponding kind in the
    array interface protocol (https://numpy.org/doc/stable/reference/arrays.interface.html).
    """

    BOOL = "b"
    INT = "i"
    UINT = "u"
    FLOAT = "f"
    OPAQUE_POINTER = "V"
    COMPLEX = "c"


@overload
def dtype_kind(sc_type: type[BoolScalar]) -> Literal[DTypeKind.BOOL]:  # type: ignore[misc]
    ...


@overload
def dtype_kind(sc_type: type[Union[int8, int16, int32, int64, int]]) -> Literal[DTypeKind.INT]:
    ...


@overload
def dtype_kind(sc_type: type[UnsignedIntScalar]) -> Literal[DTypeKind.UINT]:
    ...


@overload
def dtype_kind(sc_type: type[FloatingScalar]) -> Literal[DTypeKind.FLOAT]:
    ...


@overload
def dtype_kind(sc_type: type[Scalar]) -> DTypeKind:
    ...


def dtype_kind(sc_type: type[Scalar]) -> DTypeKind:
    """Return the data type kind of the given scalar type."""
    if issubclass(sc_type, numbers.Integral):
        if is_boolean_integral_type(sc_type):
            return DTypeKind.BOOL
        elif is_unsigned_integral_type(sc_type):
            return DTypeKind.UINT
        else:
            return DTypeKind.INT
    if issubclass(sc_type, numbers.Real):
        return DTypeKind.FLOAT
    if issubclass(sc_type, numbers.Complex):
        return DTypeKind.COMPLEX

    raise TypeError("Unknown scalar type kind")



# class DType(enum.Enum):
#     """
#     Descriptor of data type for field elements.

#     This definition is based on DLPack and Array API standards. The data
#     type is described by a name and a triple, `DTypeCode`, `bits`, and
#     `lanes`, which should be interpreted as packed `lanes` repetitions
#     of elements from `type_code` data-category of width `bits`.

#     The Array API standard only requires DTypes to be comparable with `__eq__`.

#     Additionally, instances of this class can also be used as valid NumPy
#     `dtype`s definitions due to the `.dtype` attribute.

#     Note:
#         This DType definition is implemented in a non-extensible way on purpose
#         to avoid the complexity of dealing with user-defined data types.
#     """

#     bool_ = scalars.bool_
#     int8 = scalars.int8
#     int16 = scalars.int16
#     int32 = scalars.int32
#     int64 = scalars.int64
#     uint8 = scalars.uint8
#     uint16 = scalars.uint16
#     uint32 = scalars.uint32
#     uint64 = scalars.uint64
#     float32 = scalars.float32
#     float64 = scalars.float64

#     @classmethod
#     def _missing_(cls, value: DTypeLike) -> DType:
#         canonical_type = np.dtype(value).type
#         if canonical_type != value:
#             for member in cls:
#                 if member.value == canonical_type:
#                     return member
#         return None

#     # Definition properties
#     @functools.cached_property
#     def type_code(self) -> DTypeCode:
#         return DTypeCode(self.dtype.kind)

#     @functools.cached_property
#     def bits(self) -> int:
#         assert self.dtype.itemsize % self.lanes == 0
#         return 8 * (self.dtype.itemsize // self.lanes)

#     @functools.cached_property
#     def lanes(self) -> int:
#         shape = self.dtype.shape or (1,)
#         return math.prod(shape)

#     # Convenience functions
#     @functools.cached_property
#     def scalar_type(self) -> type:
#         return self.value

#     @functools.cached_property
#     def byte_size(self) -> int:
#         return self.dtype.itemsize

#     # NumPy compatibility functions
#     @functools.cached_property
#     def dtype(self) -> np.dtype:
#         return np.dtype(self.value)

#     @classmethod
#     def from_np_dtype(cls, np_dtype: npt.DTypeLike) -> DType:
#         return cls(np.dtype(np_dtype).type)


# DTypeLike = Union[DType, npt.DTypeLike]


@dataclasses.dataclass(frozen=True)
class DType(Generic[ScalarT]):
    """
    Descriptor of data type for NDArrayObject elements.

    This definition is based on DLPack and Array API standards. The Array API
    standard only requires DTypes to be comparable with `__eq__`.

    Additionally, instances of this class can also be used as valid NumPy
    `dtype`s definitions due to the `.dtype` attribute.
    """

    scalar_type: type[ScalarT]
    subshape: tuple[int, ...] = dataclasses.field(default=())

    @functools.cached_property
    def kind(self) -> DTypeKind:
        return dtype_kind(self.scalar_type)

    @property
    def subndim(self) -> int:
        return len(self.subshape)

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype corresponding to this DType."""
        return np.dtype(self.scalar_type)


@dataclasses.dataclass(frozen=True)
class IntegerDType(DType[IntegralT]):
    pass


@dataclasses.dataclass(frozen=True)
class UnsignedIntDType(DType[UnsignedIntT]):
    pass


@dataclasses.dataclass(frozen=True)
class UInt8DType(UnsignedIntDType[int8]):
    scalar_type: Final[type[uint8]] = dataclasses.field(default=uint8, init=False)


@dataclasses.dataclass(frozen=True)
class UInt16DType(UnsignedIntDType[int16]):
    scalar_type: Final[type[uint16]] = dataclasses.field(default=uint16, init=False)


@dataclasses.dataclass(frozen=True)
class UInt32DType(UnsignedIntDType[int32]):
    scalar_type: Final[type[uint32]] = dataclasses.field(default=uint32, init=False)


@dataclasses.dataclass(frozen=True)
class UInt64DType(UnsignedIntDType[int64]):
    scalar_type: Final[type[uint64]] = dataclasses.field(default=uint64, init=False)


@dataclasses.dataclass(frozen=True)
class SignedIntDType(DType[IntT]):
    pass


@dataclasses.dataclass(frozen=True)
class Int8DType(SignedIntDType[int8]):
    scalar_type: Final[type[int8]] = dataclasses.field(default=int8, init=False)


@dataclasses.dataclass(frozen=True)
class Int16DType(SignedIntDType[int16]):
    scalar_type: Final[type[int16]] = dataclasses.field(default=int16, init=False)


@dataclasses.dataclass(frozen=True)
class Int32DType(SignedIntDType[int32]):
    scalar_type: Final[type[int32]] = dataclasses.field(default=int32, init=False)


@dataclasses.dataclass(frozen=True)
class Int64DType(SignedIntDType[int64]):
    scalar_type: Final[type[int64]] = dataclasses.field(default=int64, init=False)


@dataclasses.dataclass(frozen=True)
class FloatingDType(DType[FloatingT]):
    pass


@dataclasses.dataclass(frozen=True)
class Float32DType(FloatingDType[float32]):
    scalar_type: Final[type[float32]] = dataclasses.field(default=float32, init=False)


@dataclasses.dataclass(frozen=True)
class Float64DType(FloatingDType[float64]):
    scalar_type: Final[type[float64]] = dataclasses.field(default=float64, init=False)


SliceLike = Union[int, tuple[int, ...], None, slice, "NDArrayObject"]

NDArrayObject = Union[npt.NDArray, CuPyNDArray, JaxNDArray, "NDArrayObjectProto"]


class NDArrayObjectProto(Protocol):
    @property
    def ndim(self) -> int:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def dtype(self) -> Any:
        ...

    def __getitem__(self, item: SliceLike) -> NDArrayObject:
        ...

    def __abs__(self) -> NDArrayObject:
        ...

    def __neg__(self) -> NDArrayObject:
        ...

    def __add__(self, other: NDArrayObject | Scalar) -> NDArrayObject:
        ...

    def __radd__(self, other: NDArrayObject | Scalar) -> NDArrayObject:
        ...

    def __sub__(self, other: NDArrayObject | Scalar) -> NDArrayObject:
        ...

    def __rsub__(self, other: NDArrayObject | Scalar) -> NDArrayObject:
        ...

    def __mul__(self, other: NDArrayObject | Scalar) -> NDArrayObject:
        ...

    def __rmul__(self, other: NDArrayObject | Scalar) -> NDArrayObject:
        ...

    def __floordiv__(self, other: NDArrayObject | Scalar) -> NDArrayObject:
        ...

    def __rfloordiv__(self, other: NDArrayObject | Scalar) -> NDArrayObject:
        ...

    def __truediv__(self, other: NDArrayObject | Scalar) -> NDArrayObject:
        ...

    def __rtruediv__(self, other: NDArrayObject | Scalar) -> NDArrayObject:
        ...

    def __pow__(self, other: NDArrayObject | Scalar) -> NDArrayObject:
        ...
