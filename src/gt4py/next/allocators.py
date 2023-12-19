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

import abc
import dataclasses

import numpy as np

import gt4py._core.definitions as core_defs
import gt4py.next.common as common
import gt4py.storage.allocators as core_allocators
from gt4py.eve.extended_typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Literal,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
    TypeGuard,
    cast,
)


try:
    import cupy as cp
except ImportError:
    cp = None


CUPY_DEVICE: Final[Literal[None, core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM]] = (
    None
    if not cp
    else (core_defs.DeviceType.ROCM if cp.cuda.get_hipcc_path() else core_defs.DeviceType.CUDA)
)


FieldLayoutMapper: TypeAlias = Callable[
    [Sequence[common.Dimension]], core_allocators.BufferLayoutMap
]


class FieldBufferAllocatorProtocol(Protocol[core_defs.DeviceTypeT]):
    """Protocol for buffer allocators used to allocate memory for fields with a given domain."""

    @property
    @abc.abstractmethod
    def __gt_device_type__(self) -> core_defs.DeviceTypeT:
        ...

    @abc.abstractmethod
    def __gt_allocate__(
        self,
        domain: common.Domain,
        dtype: core_defs.DType[core_defs.ScalarT],
        device_id: int = 0,
        aligned_index: Optional[Sequence[common.NamedIndex]] = None,  # absolute position
    ) -> core_allocators.TensorBuffer[core_defs.DeviceTypeT, core_defs.ScalarT]:
        ...


def is_field_allocator(obj: Any) -> TypeGuard[FieldBufferAllocatorProtocol]:
    return hasattr(obj, "__gt_device_type__") and hasattr(obj, "__gt_allocate__")


def is_field_allocator_for(
    obj: Any, device: core_defs.DeviceTypeT
) -> TypeGuard[FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]]:
    return is_field_allocator(obj) and obj.__gt_device_type__ is device


class FieldBufferAllocatorFactoryProtocol(Protocol[core_defs.DeviceTypeT]):
    """Protocol for device-specific buffer allocator factories for fields."""

    @property
    @abc.abstractmethod
    def __gt_allocator__(self) -> FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]:
        ...


def is_field_allocator_factory(obj: Any) -> TypeGuard[FieldBufferAllocatorFactoryProtocol]:
    return hasattr(obj, "__gt_allocator__")


def is_field_allocator_factory_for(
    obj: Any, device: core_defs.DeviceTypeT
) -> TypeGuard[FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]]:
    return is_field_allocator_factory(obj) and obj.__gt_allocator__.__gt_device_type__ is device


FieldBufferAllocationUtil = (
    FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]
    | FieldBufferAllocatorFactoryProtocol[core_defs.DeviceTypeT]
)


def is_field_allocation_tool(obj: Any) -> TypeGuard[FieldBufferAllocationUtil]:
    return is_field_allocator(obj) or is_field_allocator_factory(obj)


def get_allocator(
    obj: Any,
    *,
    default: Optional[FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]] = None,
    strict: bool = False,
) -> Optional[FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]]:
    """
    Return a field-buffer-allocator from an object assumed to be an allocator or an allocator factory.

    A default allocator can be provided as fallback in case `obj` is neither an allocator nor a factory.

    Arguments:
        obj: The allocator or allocator factory.
        default: Fallback allocator.
        strict: If `True`, raise an exception if there is no way to get a valid allocator
            from `obj` or `default`.

    Returns:
        A field buffer allocator.

    Raises:
        TypeError: If `obj` is neither a field allocator nor a field allocator factory and no default
            is provided in `strict` mode.
    """
    if is_field_allocator(obj):
        return obj
    elif is_field_allocator_factory(obj):
        return obj.__gt_allocator__
    elif not strict or is_field_allocator(default):
        return default
    else:
        raise TypeError(
            f"Object '{obj}' is neither a field allocator nor a field allocator factory."
        )


@dataclasses.dataclass(frozen=True)
class BaseFieldBufferAllocator(FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]):
    """Parametrizable field buffer allocator base class."""

    device_type: core_defs.DeviceTypeT
    array_ns: core_allocators.ValidNumPyLikeAllocationNS
    layout_mapper: FieldLayoutMapper
    byte_alignment: int

    @property
    def __gt_device_type__(self) -> core_defs.DeviceTypeT:
        return self.device_type

    def __gt_allocate__(
        self,
        domain: common.Domain,
        dtype: core_defs.DType[core_defs.ScalarT],
        device_id: int = 0,
        aligned_index: Optional[Sequence[common.NamedIndex]] = None,  # absolute position
    ) -> core_allocators.TensorBuffer[core_defs.DeviceTypeT, core_defs.ScalarT]:
        shape = domain.shape
        layout_map = self.layout_mapper(domain.dims)
        # TODO(egparedes): add support for non-empty aligned index values
        assert aligned_index is None

        return core_allocators.NDArrayBufferAllocator(self.device_type, self.array_ns).allocate(
            shape, dtype, device_id, layout_map, self.byte_alignment, aligned_index
        )


if TYPE_CHECKING:
    __TensorFieldAllocatorAsFieldAllocatorInterfaceT: type[
        FieldBufferAllocatorProtocol
    ] = BaseFieldBufferAllocator


def horizontal_first_layout_mapper(
    dims: Sequence[common.Dimension],
) -> core_allocators.BufferLayoutMap:
    """Map dimensions to a buffer layout making horizonal dims change the slowest (i.e. larger strides)."""

    def pos_of_kind(kind: common.DimensionKind) -> list[int]:
        return [i for i, dim in enumerate(dims) if dim.kind == kind]

    horizontals = pos_of_kind(common.DimensionKind.HORIZONTAL)
    verticals = pos_of_kind(common.DimensionKind.VERTICAL)
    locals_ = pos_of_kind(common.DimensionKind.LOCAL)

    layout_map = [0] * len(dims)
    for i, pos in enumerate(horizontals + verticals + locals_):
        layout_map[pos] = len(dims) - 1 - i

    valid_layout_map = tuple(layout_map)
    assert core_allocators.is_valid_layout_map(valid_layout_map)

    return valid_layout_map


if TYPE_CHECKING:
    __horizontal_first_layout_mapper: FieldLayoutMapper = horizontal_first_layout_mapper


#: Registry of default allocators for each device type.
device_allocators: dict[core_defs.DeviceType, FieldBufferAllocatorProtocol] = {}


assert core_allocators.is_valid_nplike_allocation_ns(np)
np_alloc_ns: core_allocators.ValidNumPyLikeAllocationNS = np  # Just for static type checking


class StandardCPUFieldBufferAllocator(BaseFieldBufferAllocator[core_defs.CPUDeviceTyping]):
    """A field buffer allocator for CPU devices that uses a horizontal-first layout mapper and 64-byte alignment."""

    def __init__(self) -> None:
        super().__init__(
            device_type=core_defs.DeviceType.CPU,
            array_ns=np_alloc_ns,
            layout_mapper=horizontal_first_layout_mapper,
            byte_alignment=64,
        )


device_allocators[core_defs.DeviceType.CPU] = StandardCPUFieldBufferAllocator()

assert is_field_allocator(device_allocators[core_defs.DeviceType.CPU])


@dataclasses.dataclass(frozen=True)
class InvalidFieldBufferAllocator(FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]):
    """A field buffer allocator that always raises an exception."""

    device_type: core_defs.DeviceTypeT
    exception: Exception

    @property
    def __gt_device_type__(self) -> core_defs.DeviceTypeT:
        return self.device_type

    def __gt_allocate__(
        self,
        domain: common.Domain,
        dtype: core_defs.DType[core_defs.ScalarT],
        device_id: int = 0,
        aligned_index: Optional[Sequence[common.NamedIndex]] = None,  # absolute position
    ) -> core_allocators.TensorBuffer[core_defs.DeviceTypeT, core_defs.ScalarT]:
        raise self.exception


if CUPY_DEVICE is not None:
    cp_alloc_ns: core_allocators.ValidNumPyLikeAllocationNS = cp  # Just for static type checking
    assert core_allocators.is_valid_nplike_allocation_ns(cp_alloc_ns)

    if CUPY_DEVICE is core_defs.DeviceType.CUDA:

        class CUDAFieldBufferAllocator(BaseFieldBufferAllocator[core_defs.CUDADeviceTyping]):
            def __init__(self) -> None:
                super().__init__(
                    device_type=core_defs.DeviceType.CUDA,
                    array_ns=cp_alloc_ns,
                    layout_mapper=horizontal_first_layout_mapper,
                    byte_alignment=128,
                )

        device_allocators[core_defs.DeviceType.CUDA] = CUDAFieldBufferAllocator()

    else:

        class ROCMFieldBufferAllocator(BaseFieldBufferAllocator[core_defs.ROCMDeviceTyping]):
            def __init__(self) -> None:
                super().__init__(
                    device_type=core_defs.DeviceType.ROCM,
                    array_ns=cp_alloc_ns,
                    layout_mapper=horizontal_first_layout_mapper,
                    byte_alignment=128,
                )

        device_allocators[core_defs.DeviceType.ROCM] = ROCMFieldBufferAllocator()

else:

    class InvalidGPUFielBufferAllocator(InvalidFieldBufferAllocator[core_defs.CUDADeviceTyping]):
        def __init__(self) -> None:
            super().__init__(
                device_type=core_defs.DeviceType.CUDA,
                exception=RuntimeError("Missing `cupy` dependency for GPU allocation"),
            )


StandardGPUFieldBufferAllocator: Final[type[FieldBufferAllocatorProtocol]] = cast(
    type[FieldBufferAllocatorProtocol],
    type(device_allocators[CUPY_DEVICE]) if CUPY_DEVICE else InvalidGPUFielBufferAllocator,
)


def allocate(
    domain: common.DomainLike,
    dtype: core_defs.DType[core_defs.ScalarT],
    *,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    allocator: Optional[FieldBufferAllocationUtil] = None,
    device: Optional[core_defs.Device] = None,
) -> core_allocators.TensorBuffer:
    """
    Allocate a TensorBuffer for the given domain and device or allocator.

    The arguments `device` and `allocator` are mutually exclusive.
    If `device` is specified, the corresponding default allocator
    (defined in :data:`device_allocators`) is used.

    Arguments:
        domain: The domain which should be backed by the allocated tensor buffer.
        dtype: Data type.
        aligned_index: N-dimensional index of the first aligned element
        allocator: The allocator to use for the allocation.
        device: The device to allocate the tensor buffer on (using the default
            allocator for this kind of device from :data:`device_allocators`).

    Returns:
        The allocated tensor buffer.

    Raises:
        ValueError
            If illegal or inconsistent arguments are specified.

    """
    if device is None and allocator is None:
        raise ValueError("No 'device' or 'allocator' specified.")
    actual_allocator = get_allocator(allocator)
    if actual_allocator is None:
        assert device is not None  # for mypy
        actual_allocator = device_allocators[device.device_type]
    elif device is None:
        device = core_defs.Device(actual_allocator.__gt_device_type__, 0)
    elif device.device_type != actual_allocator.__gt_device_type__:
        raise ValueError(f"Device '{device}' and allocator '{actual_allocator}' are incompatible.")

    return actual_allocator.__gt_allocate__(
        domain=common.domain(domain),
        dtype=dtype,
        device_id=device.device_id,
        aligned_index=aligned_index,
    )
