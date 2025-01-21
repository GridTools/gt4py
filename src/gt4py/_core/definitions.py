# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import collections.abc
import dataclasses
import enum
import functools
import math
import numbers


try:
    import ml_dtypes
except ModuleNotFoundError:
    ml_dtypes = None

import numpy as np
import numpy.typing as npt

import gt4py.eve as eve
from gt4py.eve.extended_typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Iterator,
    Literal,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
    cast,
    overload,
)


if TYPE_CHECKING:
    import cupy as cp

    CuPyNDArray: TypeAlias = cp.ndarray

    import jax.numpy as jnp

    JaxNDArray: TypeAlias = jnp.ndarray


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

float16 = np.float16
if ml_dtypes:
    bfloat16 = ml_dtypes.bfloat16
float32 = np.float32
float64 = np.float64

BoolScalar: TypeAlias = Union[bool_, bool]
BoolT = TypeVar("BoolT", bound=BoolScalar)
BOOL_TYPES: Final[Tuple[type, ...]] = cast(
    Tuple[type, ...],
    BoolScalar.__args__,  # type: ignore[attr-defined]
)


IntScalar: TypeAlias = Union[int8, int16, int32, int64, int]
IntT = TypeVar("IntT", bound=IntScalar)
INT_TYPES: Final[Tuple[type, ...]] = cast(
    Tuple[type, ...],
    IntScalar.__args__,  # type: ignore[attr-defined]
)


UnsignedIntScalar: TypeAlias = Union[uint8, uint16, uint32, uint64]
UnsignedIntT = TypeVar("UnsignedIntT", bound=UnsignedIntScalar)
UINT_TYPES: Final[Tuple[type, ...]] = cast(
    Tuple[type, ...],
    UnsignedIntScalar.__args__,  # type: ignore[attr-defined]
)


IntegralScalar: TypeAlias = Union[IntScalar, UnsignedIntScalar]
IntegralT = TypeVar("IntegralT", bound=IntegralScalar)
INTEGRAL_TYPES: Final[Tuple[type, ...]] = (*INT_TYPES, *UINT_TYPES)

if ml_dtypes:
    FloatingScalar: TypeAlias = Union[float16, ml_dtypes.bfloat16, float32, float64, float]
else:
    FloatingScalar: TypeAlias = Union[float16, float32, float64, float]
FloatingT = TypeVar("FloatingT", bound=FloatingScalar)
FLOAT_TYPES: Final[Tuple[type, ...]] = cast(
    Tuple[type, ...],
    FloatingScalar.__args__,  # type: ignore[attr-defined]
)


#: Type alias for all scalar types supported by GT4Py
Scalar: TypeAlias = Union[BoolScalar, IntegralScalar, FloatingScalar]
ScalarT = TypeVar("ScalarT", bound=Scalar)
SCALAR_TYPES: Final[tuple[type, ...]] = (*BOOL_TYPES, *INTEGRAL_TYPES, *FLOAT_TYPES)


def is_scalar_type(t: Any) -> TypeGuard[Scalar]:
    return isinstance(t, SCALAR_TYPES)


class BooleanIntegral(numbers.Integral):
    """Abstract base class for boolean integral types."""

    ...


class PositiveIntegral(numbers.Integral):
    """Abstract base class representing positive integral numbers.."""

    ...


def is_boolean_integral_type(integral_type: type) -> TypeGuard[Type[BooleanIntegral]]:
    return issubclass(integral_type, BOOL_TYPES)


def is_positive_integral_type(integral_type: type) -> TypeGuard[Type[PositiveIntegral]]:
    return issubclass(integral_type, UINT_TYPES)


TensorShape: TypeAlias = Sequence[
    int
]  # TODO(egparedes) figure out if PositiveIntegral can be made to work


def is_valid_tensor_shape(value: Sequence[IntegralScalar]) -> TypeGuard[TensorShape]:
    return isinstance(value, collections.abc.Sequence) and all(
        isinstance(v, numbers.Integral) and v > 0 for v in value
    )


# -- Data type descriptors --
class DTypeKind(eve.StrEnum):
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
def dtype_kind(
    sc_type: Type[IntT] | Type[BoolT],  # mypy doesn't distinguish IntT and BoolT
) -> Literal[DTypeKind.INT, DTypeKind.BOOL]: ...


@overload
def dtype_kind(sc_type: Type[UnsignedIntT]) -> Literal[DTypeKind.UINT]: ...


@overload
def dtype_kind(sc_type: Type[FloatingT]) -> Literal[DTypeKind.FLOAT]: ...


@overload
def dtype_kind(sc_type: Type[ScalarT]) -> DTypeKind: ...


def dtype_kind(sc_type: Type[ScalarT]) -> DTypeKind:
    """Return the data type kind of the given scalar type."""
    if issubclass(sc_type, numbers.Integral):
        if is_boolean_integral_type(sc_type):
            return DTypeKind.BOOL
        elif is_positive_integral_type(sc_type):
            return DTypeKind.UINT
        else:
            return DTypeKind.INT
    if issubclass(sc_type, numbers.Real):
        return DTypeKind.FLOAT
    if issubclass(sc_type, numbers.Complex):
        return DTypeKind.COMPLEX

    raise TypeError("Unknown scalar type kind.")


@dataclasses.dataclass(frozen=True)
class DType(Generic[ScalarT]):
    """
    Descriptor of data type for Field elements.

    This definition is based on DLPack and Array API standards. The data type
    should be interpreted as packed `lanes` repetitions of elements from
    `kind` data-category of `bit_width` width.

    The Array API standard only requires DTypes to be comparable with `__eq__`.

    Additionally, instances of this class can also be used as valid NumPy
    `dtype`s definitions due to the `.dtype` attribute.
    """

    scalar_type: Type[ScalarT]
    tensor_shape: TensorShape = dataclasses.field(default=())

    def __post_init__(self) -> None:
        if not isinstance(self.scalar_type, type):
            raise TypeError(f"Invalid scalar type '{self.scalar_type}'")
        if not is_valid_tensor_shape(self.tensor_shape):
            raise TypeError(f"Invalid tensor shape '{self.tensor_shape}'")

    @functools.cached_property
    def kind(self) -> DTypeKind:
        return dtype_kind(self.scalar_type)

    @functools.cached_property
    def dtype(self) -> np.dtype:
        """The NumPy dtype corresponding to this DType."""
        return np.dtype(f"{self.tensor_shape}{np.dtype(self.scalar_type).name}")

    @functools.cached_property
    def byte_size(self) -> int:
        return np.dtype(self.scalar_type).itemsize * self.lanes

    @property
    def bit_width(self) -> int:
        return 8 * np.dtype(self.scalar_type).itemsize

    @property
    def lanes(self) -> int:
        return math.prod(self.tensor_shape or (1,))

    @property
    def subndim(self) -> int:
        return len(self.tensor_shape)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, DType)
            and self.scalar_type == other.scalar_type
            and self.tensor_shape == other.tensor_shape
        )

    def __hash__(self) -> int:
        return hash((self.scalar_type, self.tensor_shape))


@dataclasses.dataclass(frozen=True)
class IntegerDType(DType[IntegralT]):
    pass


@dataclasses.dataclass(frozen=True)
class UnsignedIntDType(DType[UnsignedIntT]):
    pass


@dataclasses.dataclass(frozen=True)
class UInt8DType(UnsignedIntDType[uint8]):
    scalar_type: Final[Type[uint8]] = dataclasses.field(default=uint8, init=False)


@dataclasses.dataclass(frozen=True)
class UInt16DType(UnsignedIntDType[uint16]):
    scalar_type: Final[Type[uint16]] = dataclasses.field(default=uint16, init=False)


@dataclasses.dataclass(frozen=True)
class UInt32DType(UnsignedIntDType[uint32]):
    scalar_type: Final[Type[uint32]] = dataclasses.field(default=uint32, init=False)


@dataclasses.dataclass(frozen=True)
class UInt64DType(UnsignedIntDType[uint64]):
    scalar_type: Final[Type[uint64]] = dataclasses.field(default=uint64, init=False)


@dataclasses.dataclass(frozen=True)
class SignedIntDType(DType[IntT]):
    pass


@dataclasses.dataclass(frozen=True)
class Int8DType(SignedIntDType[int8]):
    scalar_type: Final[Type[int8]] = dataclasses.field(default=int8, init=False)


@dataclasses.dataclass(frozen=True)
class Int16DType(SignedIntDType[int16]):
    scalar_type: Final[Type[int16]] = dataclasses.field(default=int16, init=False)


@dataclasses.dataclass(frozen=True)
class Int32DType(SignedIntDType[int32]):
    scalar_type: Final[Type[int32]] = dataclasses.field(default=int32, init=False)


@dataclasses.dataclass(frozen=True)
class Int64DType(SignedIntDType[int64]):
    scalar_type: Final[Type[int64]] = dataclasses.field(default=int64, init=False)


@dataclasses.dataclass(frozen=True)
class FloatingDType(DType[FloatingT]):
    pass


@dataclasses.dataclass(frozen=True)  # TODO
class Float16DType(FloatingDType[float16]):
    scalar_type: Final[Type[float16]] = dataclasses.field(default=float16, init=False)


if ml_dtypes:

    @dataclasses.dataclass(frozen=True)  # TODO
    class BFloat16DType(FloatingDType[ml_dtypes.bfloat16]):
        scalar_type: Final[Type[ml_dtypes.bfloat16]] = dataclasses.field(
            default=ml_dtypes.bfloat16, init=False
        )


@dataclasses.dataclass(frozen=True)
class Float32DType(FloatingDType[float32]):
    scalar_type: Final[Type[float32]] = dataclasses.field(default=float32, init=False)


@dataclasses.dataclass(frozen=True)
class Float64DType(FloatingDType[float64]):
    scalar_type: Final[Type[float64]] = dataclasses.field(default=float64, init=False)


@dataclasses.dataclass(frozen=True)
class BoolDType(DType[bool_]):
    scalar_type: Final[Type[bool_]] = dataclasses.field(default=bool_, init=False)


DTypeLike = Union[DType, npt.DTypeLike]


def dtype(dtype_like: DTypeLike) -> DType:
    """Return the DType corresponding to the given dtype-like object."""
    return dtype_like if isinstance(dtype_like, DType) else DType(np.dtype(dtype_like).type)


# -- Custom protocols  --
class GTDimsInterface(Protocol):
    """
    A `GTDimsInterface` is an object providing the `__gt_dims__` property, naming the buffer dimensions.

    In `gt4py.cartesian` the allowed values are `"I"`, `"J"` and `"K"` with the established semantics.

    See :ref:`cartesian-arrays-dimension-mapping` for details.
    """

    @property
    def __gt_dims__(self) -> Tuple[str, ...]: ...


class GTOriginInterface(Protocol):
    """
    A `GTOriginInterface` is an object providing `__gt_origin__`, describing the origin of a buffer.

    See :ref:`cartesian-arrays-default-origin` for details.
    """

    @property
    def __gt_origin__(self) -> Tuple[int, ...]: ...


# -- Device representation --
class DeviceType(enum.IntEnum):
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
    CUDA_MANAGED = 13
    ONE_API = 14


CPUDeviceTyping: TypeAlias = Literal[DeviceType.CPU]
CUDADeviceTyping: TypeAlias = Literal[DeviceType.CUDA]
CPUPinnedDeviceTyping: TypeAlias = Literal[DeviceType.CPU_PINNED]
OpenCLDeviceTyping: TypeAlias = Literal[DeviceType.OPENCL]
VulkanDeviceTyping: TypeAlias = Literal[DeviceType.VULKAN]
MetalDeviceTyping: TypeAlias = Literal[DeviceType.METAL]
VPIDeviceTyping: TypeAlias = Literal[DeviceType.VPI]
ROCMDeviceTyping: TypeAlias = Literal[DeviceType.ROCM]
CUDAManagedDeviceTyping: TypeAlias = Literal[DeviceType.CUDA_MANAGED]
OneApiDeviceTyping: TypeAlias = Literal[DeviceType.ONE_API]


DeviceTypeT = TypeVar(
    "DeviceTypeT",
    CPUDeviceTyping,
    CUDADeviceTyping,
    CPUPinnedDeviceTyping,
    OpenCLDeviceTyping,
    VulkanDeviceTyping,
    MetalDeviceTyping,
    VPIDeviceTyping,
    ROCMDeviceTyping,
)


@dataclasses.dataclass(frozen=True)
class Device(Generic[DeviceTypeT]):
    """
    Representation of a computing device.

    This definition is based on the DLPack device definition. A device is
    described by a pair of `DeviceType` and `device_id`. The `device_id`
    is an integer that is interpreted differently depending on the
    `DeviceType`. For example, for `DeviceType.CPU` it could be the CPU
    core number, for `DeviceType.CUDA` it could be the CUDA device number, etc.
    """

    device_type: DeviceTypeT
    device_id: int

    def __iter__(self) -> Iterator[DeviceTypeT | int]:
        yield self.device_type
        yield self.device_id


# -- NDArrays and slices --
SliceLike = Union[int, Tuple[int, ...], None, slice, "NDArrayObject"]


class NDArrayObject(Protocol):
    @property
    def ndim(self) -> int: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def strides(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...

    @property
    def itemsize(self) -> int: ...

    def item(self) -> Any: ...

    def astype(self, dtype: npt.DTypeLike) -> NDArrayObject: ...

    def any(self) -> bool: ...

    def __getitem__(self, item: Any) -> NDArrayObject: ...

    def __abs__(self) -> NDArrayObject: ...

    def __neg__(self) -> NDArrayObject: ...

    def __add__(self, other: NDArrayObject | Scalar) -> NDArrayObject: ...

    def __radd__(self, other: Any) -> NDArrayObject: ...

    def __sub__(self, other: NDArrayObject | Scalar) -> NDArrayObject: ...

    def __rsub__(self, other: Any) -> NDArrayObject: ...

    def __mul__(self, other: NDArrayObject | Scalar) -> NDArrayObject: ...

    def __rmul__(self, other: Any) -> NDArrayObject: ...

    def __floordiv__(self, other: NDArrayObject | Scalar) -> NDArrayObject: ...

    def __rfloordiv__(self, other: Any) -> NDArrayObject: ...

    def __truediv__(self, other: NDArrayObject | Scalar) -> NDArrayObject: ...

    def __rtruediv__(self, other: Any) -> NDArrayObject: ...

    def __pow__(self, other: NDArrayObject | Scalar) -> NDArrayObject: ...

    def __eq__(self, other: NDArrayObject | Scalar) -> NDArrayObject:  # type: ignore[override] # mypy wants to return `bool`
        ...

    def __ne__(self, other: NDArrayObject | Scalar) -> NDArrayObject:  # type: ignore[override] # mypy wants to return `bool`
        ...

    def __gt__(self, other: NDArrayObject | Scalar) -> NDArrayObject:  # type: ignore[misc] # Forward operator is not callable
        ...

    def __ge__(self, other: NDArrayObject | Scalar) -> NDArrayObject:  # type: ignore[misc] # Forward operator is not callable
        ...

    def __lt__(self, other: NDArrayObject | Scalar) -> NDArrayObject:  # type: ignore[misc] # Forward operator is not callable
        ...

    def __le__(self, other: NDArrayObject | Scalar) -> NDArrayObject:  # type: ignore[misc] # Forward operator is not callable
        ...

    def __and__(self, other: NDArrayObject | Scalar) -> NDArrayObject: ...

    def __or__(self, other: NDArrayObject | Scalar) -> NDArrayObject: ...

    def __xor__(self, other: NDArrayObject | Scalar) -> NDArrayObject: ...
