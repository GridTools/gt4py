# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# TODO custom_layout_allocators ?

import abc
import dataclasses
import functools

import gt4py._core.definitions as core_defs
import gt4py.next.common as common
import gt4py.storage.allocators as core_allocators
from gt4py.eve.extended_typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
    TypeIs,
    cast,
)


FieldLayoutMapper: TypeAlias = Callable[
    [Sequence[common.Dimension]], core_allocators.BufferLayoutMap
]


class FieldBufferAllocatorProtocol(Protocol[core_defs.DeviceTypeT]):
    """Protocol for buffer allocators used to allocate memory for fields with a given domain."""

    @property
    @abc.abstractmethod
    def __gt_device_type__(self) -> core_defs.DeviceTypeT: ...

    @abc.abstractmethod
    def __gt_allocate__(
        self,
        domain: common.Domain,
        dtype: core_defs.DType[core_defs.ScalarT],
        device_id: int = 0,
        aligned_index: Optional[Sequence[common.NamedIndex]] = None,  # absolute position
    ) -> core_allocators.TensorBuffer[core_defs.DeviceTypeT, core_defs.ScalarT]: ...


def is_field_allocator(obj: Any) -> TypeIs[FieldBufferAllocatorProtocol]:
    return hasattr(obj, "__gt_device_type__") and hasattr(obj, "__gt_allocate__")


def is_field_allocator_for(
    obj: Any, device: core_defs.DeviceTypeT
) -> TypeIs[FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]]:
    return is_field_allocator(obj) and obj.__gt_device_type__ is device


class FieldBufferAllocatorFactoryProtocol(Protocol[core_defs.DeviceTypeT]):
    """Protocol for device-specific buffer allocator factories for fields."""

    @property
    @abc.abstractmethod
    def __gt_allocator__(self) -> FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]: ...


def is_field_allocator_factory(obj: Any) -> TypeIs[FieldBufferAllocatorFactoryProtocol]:
    return hasattr(obj, "__gt_allocator__")


def is_field_allocator_factory_for(
    obj: Any, device: core_defs.DeviceTypeT
) -> TypeIs[FieldBufferAllocatorFactoryProtocol[core_defs.DeviceTypeT]]:
    return is_field_allocator_factory(obj) and obj.__gt_allocator__.__gt_device_type__ is device


FieldBufferAllocationUtil = (
    FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]
    | FieldBufferAllocatorFactoryProtocol[core_defs.DeviceTypeT]
)


def is_field_allocation_tool(obj: Any) -> TypeIs[FieldBufferAllocationUtil]:
    return is_field_allocator(obj) or is_field_allocator_factory(obj)


def is_field_allocation_tool_for(
    obj: Any, device: core_defs.DeviceTypeT
) -> TypeIs[FieldBufferAllocationUtil]:
    return is_field_allocator_for(obj, device) or is_field_allocator_factory_for(obj, device)


@dataclasses.dataclass(frozen=True)
class BaseFieldBufferAllocator(FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]):
    """Parametrizable field buffer allocator base class."""

    device_type: core_defs.DeviceTypeT
    array_utils: core_allocators.ArrayUtils
    layout_mapper: FieldLayoutMapper
    byte_alignment: int

    @functools.cached_property
    def buffer_allocator(self) -> core_allocators.BufferAllocator:
        return core_allocators.NDArrayBufferAllocator(self.device_type, self.array_utils)

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
        # TODO implement the alignment and add tests for both Cell and Edge in the aligned_index Sequence
        assert aligned_index is None

        return self.buffer_allocator.allocate(
            shape, dtype, device_id, layout_map, self.byte_alignment, aligned_index
        )


if TYPE_CHECKING:
    __TensorFieldAllocatorAsFieldAllocatorInterfaceT: type[FieldBufferAllocatorProtocol] = (
        BaseFieldBufferAllocator
    )


def horizontal_first_layout_mapper(
    dims: Sequence[common.Dimension],
) -> core_allocators.BufferLayoutMap:
    """Map dimensions to a buffer layout making horizontal dims change the slowest (i.e. larger strides)."""

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


class StandardCPUFieldBufferAllocator(BaseFieldBufferAllocator[core_defs.CPUDeviceTyping]):
    """A field buffer allocator for CPU devices that uses a horizontal-first layout mapper and 64-byte alignment."""

    def __init__(self) -> None:
        super().__init__(
            device_type=core_defs.DeviceType.CPU,
            array_utils=core_allocators.numpy_array_utils,
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


if core_defs.CUPY_DEVICE_TYPE is not None:
    assert isinstance(core_allocators.cupy_array_utils, core_allocators.ArrayUtils)
    cupy_array_utils = core_allocators.cupy_array_utils

    if core_defs.CUPY_DEVICE_TYPE is core_defs.DeviceType.CUDA:

        class CUDAFieldBufferAllocator(BaseFieldBufferAllocator[core_defs.CUDADeviceTyping]):
            def __init__(self) -> None:
                super().__init__(
                    device_type=core_defs.DeviceType.CUDA,
                    array_utils=cupy_array_utils,
                    layout_mapper=horizontal_first_layout_mapper,
                    byte_alignment=128,
                )

        device_allocators[core_defs.DeviceType.CUDA] = CUDAFieldBufferAllocator()

    else:

        class ROCMFieldBufferAllocator(BaseFieldBufferAllocator[core_defs.ROCMDeviceTyping]):
            def __init__(self) -> None:
                super().__init__(
                    device_type=core_defs.DeviceType.ROCM,
                    array_utils=cupy_array_utils,
                    layout_mapper=horizontal_first_layout_mapper,
                    byte_alignment=128,
                )

        device_allocators[core_defs.DeviceType.ROCM] = ROCMFieldBufferAllocator()

else:

    class InvalidGPUFieldBufferAllocator(InvalidFieldBufferAllocator[core_defs.CUDADeviceTyping]):
        def __init__(self) -> None:
            super().__init__(
                device_type=core_defs.DeviceType.CUDA,
                exception=RuntimeError("Missing `cupy` dependency for GPU allocation"),
            )


StandardGPUFieldBufferAllocator: Final[type[FieldBufferAllocatorProtocol]] = cast(
    type[FieldBufferAllocatorProtocol],
    type(device_allocators[core_defs.CUPY_DEVICE_TYPE])
    if core_defs.CUPY_DEVICE_TYPE
    else InvalidGPUFieldBufferAllocator,
)
