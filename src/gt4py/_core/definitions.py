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
    Iterator,
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


# -- Device representation --
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

    def __iter__(self) -> Iterator[DeviceType | int]:
        yield self.device_type
        yield self.device_id


# -- Scalar types supported by GT4Py --
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
IntT = TypeVar("IntT", bound=Union[int8, int16, int32, int64, int])
INT_TYPES: Final[tuple[type, ...]] = cast(tuple[type, ...], IntScalar.__args__)  # type: ignore[attr-defined]

UnsignedIntScalar: TypeAlias = Union[uint8, uint16, uint32, uint64]
UnsignedIntT = TypeVar("UnsignedIntT", bound=Union[uint8, uint16, uint32, uint64])
UINT_TYPES: Final[tuple[type, ...]] = cast(tuple[type, ...], UnsignedIntScalar.__args__)  # type: ignore[attr-defined]

IntegralScalar: TypeAlias = Union[IntScalar, UnsignedIntScalar]
IntegralT = TypeVar("IntegralT", bound=Union[IntScalar, UnsignedIntScalar])
INTEGRAL_TYPES: Final[tuple[type, ...]] = (*INT_TYPES, *UINT_TYPES)

FloatingScalar: TypeAlias = Union[float32, float64, float]
FloatingT = TypeVar("FloatingT", bound=Union[float32, float64, float])
FLOAT_TYPES: Final[tuple[type, ...]] = cast(tuple[type, ...], FloatingScalar.__args__)  # type: ignore[attr-defined]

#: Type alias for all scalar types supported by GT4Py
Scalar: TypeAlias = Union[BoolScalar, IntegralScalar, FloatingScalar]
ScalarT = TypeVar("ScalarT", bound=Union[BoolScalar, IntegralScalar, FloatingScalar])
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


# -- Data type descriptors --
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
def dtype_kind(sc_type: type[BoolT]) -> Literal[DTypeKind.BOOL]:  # type: ignore[misc]
    ...


@overload
def dtype_kind(sc_type: type[IntT]) -> Literal[DTypeKind.INT]:
    ...


@overload
def dtype_kind(sc_type: type[UnsignedIntT]) -> Literal[DTypeKind.UINT]:
    ...


@overload
def dtype_kind(sc_type: type[FloatingT]) -> Literal[DTypeKind.FLOAT]:
    ...


@overload
def dtype_kind(sc_type: type[ScalarT]) -> DTypeKind:
    ...


def dtype_kind(sc_type: type[ScalarT]) -> DTypeKind:
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


@dataclasses.dataclass(frozen=True)
class DType(Generic[ScalarT]):
    """
    Descriptor of data type for Field elements.

    This definition is based on DLPack and Array API standards. The Array API
    standard only requires DTypes to be comparable with `__eq__`.

    Additionally, instances of this class can also be used as valid NumPy
    `dtype`s definitions due to the `.dtype` attribute.
    """

    #     This definition is based on DLPack and Array API standards. The data
    #     type is described by a name and a triple, `DTypeCode`, `bits`, and
    #     `lanes`, which should be interpreted as packed `lanes` repetitions
    #     of elements from `type_code` data-category of width `bits`.
    #     Note:
    #         This DType definition is implemented in a non-extensible way on purpose
    #         to avoid the complexity of dealing with user-defined data types.

    scalar_type: type[ScalarT]
    subshape: tuple[int, ...] = dataclasses.field(default=())

    @functools.cached_property
    def kind(self) -> DTypeKind:
        return dtype_kind(self.scalar_type)

    @functools.cached_property
    def dtype(self) -> np.dtype:
        """NumPy dtype corresponding to this DType."""
        return np.dtype(self.scalar_type)

    @property
    def byte_size(self) -> int:
        return self.dtype.itemsize

    @property
    def subndim(self) -> int:
        return len(self.subshape)


#     @functools.cached_property
#     def bits(self) -> int:
#         assert self.dtype.itemsize % self.lanes == 0
#         return 8 * (self.dtype.itemsize // self.lanes)

#     @functools.cached_property
#     def lanes(self) -> int:
#         shape = self.dtype.shape or (1,)
#         return math.prod(shape)


@dataclasses.dataclass(frozen=True)
class IntegerDType(DType[IntegralT]):
    pass


@dataclasses.dataclass(frozen=True)
class UnsignedIntDType(DType[UnsignedIntT]):
    pass


@dataclasses.dataclass(frozen=True)
class UInt8DType(UnsignedIntDType[uint8]):
    scalar_type: Final[type[uint8]] = dataclasses.field(default=uint8, init=False)


@dataclasses.dataclass(frozen=True)
class UInt16DType(UnsignedIntDType[uint16]):
    scalar_type: Final[type[uint16]] = dataclasses.field(default=uint16, init=False)


@dataclasses.dataclass(frozen=True)
class UInt32DType(UnsignedIntDType[uint32]):
    scalar_type: Final[type[uint32]] = dataclasses.field(default=uint32, init=False)


@dataclasses.dataclass(frozen=True)
class UInt64DType(UnsignedIntDType[uint64]):
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


DTypeLike = Union[DType, npt.DTypeLike]


def dtype(dtype_like: DTypeLike) -> DType:
    """Return the DType corresponding to the given dtype-like object."""
    return dtype_like if isinstance(dtype_like, DType) else DType(np.dtype(dtype_like).type)


# -- NDArrays and slices --
SliceLike = Union[int, tuple[int, ...], None, slice, "NDArrayObject"]


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

    def __getitem__(self, item: SliceLike) -> NDArrayObjectProto:
        ...

    def __abs__(self) -> NDArrayObjectProto:
        ...

    def __neg__(self) -> NDArrayObjectProto:
        ...

    def __add__(self, other: NDArrayObject | Scalar) -> NDArrayObjectProto:
        ...

    def __radd__(self, other: NDArrayObject | Scalar) -> NDArrayObjectProto:
        ...

    def __sub__(self, other: NDArrayObject | Scalar) -> NDArrayObjectProto:
        ...

    def __rsub__(self, other: NDArrayObject | Scalar) -> NDArrayObjectProto:
        ...

    def __mul__(self, other: NDArrayObject | Scalar) -> NDArrayObjectProto:
        ...

    def __rmul__(self, other: NDArrayObject | Scalar) -> NDArrayObjectProto:
        ...

    def __floordiv__(self, other: NDArrayObject | Scalar) -> NDArrayObjectProto:
        ...

    def __rfloordiv__(self, other: NDArrayObject | Scalar) -> NDArrayObjectProto:
        ...

    def __truediv__(self, other: NDArrayObject | Scalar) -> NDArrayObjectProto:
        ...

    def __rtruediv__(self, other: NDArrayObject | Scalar) -> NDArrayObjectProto:
        ...

    def __pow__(self, other: NDArrayObject | Scalar) -> NDArrayObjectProto:
        ...


NDArrayObject = Union[npt.NDArray, "CuPyNDArray", "JaxNDArray", NDArrayObjectProto]
NDArrayObjectT = TypeVar(
    "NDArrayObjectT", npt.NDArray, "CuPyNDArray", "JaxNDArray", NDArrayObjectProto
)
