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

import abc
import dataclasses
import functools
import math
import operator

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.eve.extended_typing import (
    Any,
    Generic,
    NewType,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
    TypeGuard,
    TypeVar,
    TYPE_CHECKING,
)

try:
    import cupy as cp
except ImportError:
    cp = None


_ScalarT = TypeVar("_ScalarT", bound=core_defs.Scalar)
_NDBufferT = TypeVar(
    "_NDBufferT", xtyping.ArrayInterface, xtyping.CUDAArrayInterface, xtyping.DLPackBuffer
)

#: Tuple of positive integers encoding a tensor shape.
BufferShape: TypeAlias = Tuple[core_defs.UnsignedIntegral, ...]


#: Tuple of positive integers encoding a permutation of the dimensions.
BufferLayoutMap = NewType("BufferLayoutMap", Tuple[core_defs.UnsignedIntegral, ...])


def is_valid_shape(value: tuple[core_defs.IntegralScalar, ...]) -> TypeGuard[BufferShape]:
    return isinstance(value, tuple) and all(isinstance(v, int) and v > 0 for v in value)


def is_valid_layout_map(value: tuple[core_defs.IntegralScalar, ...]) -> TypeGuard[BufferLayoutMap]:
    dims = list(range(len(value)))
    return (
        isinstance(value, tuple) and len(value) == len(set(value)) and all(i in dims for i in value)
    )


@dataclasses.dataclass(frozen=True, slots=True)
class TensorBuffer(Generic[_NDBufferT, _ScalarT]):
    """
    N-dimensional (tensor-like) memory buffer.

    The actual class of the stored buffer and ndarray instances is
    represented in the `NDBufferT` parameter and might be any n-dimensional
    buffer-like class with a compatible buffer interface (e.g. NumPy
    or CuPy `ndarray`.)

    Attributes:
        buffer: Raw allocated buffer.
        memory_address: Memory address of the buffer.
        device: Device where the buffer is allocated.
        dtype: Data type descriptor.
        shape: Tuple with lengths of the corresponding tensor dimensions.
        strides: Tuple with sizes (in bytes) of the steps in each dimension.
        layout_map: Tuple with the order of the dimensions in the buffer.
            layout_map[i] = j means that the i-th dimension of the tensor
            corresponds to the j-th dimension in the buffer.
        byte_offset: Offset (in bytes) from the beginning of the buffer to
            the first valid element.
        byte_alignment: Alignment (in bytes) of the first valid element.
        aligned_index: N-dimensional index of the first aligned element.
        ndarray: N-dimensional tensor view of the allocated buffer.
    """

    buffer: _NDBufferT = dataclasses.field(hash=False)
    memory_address: int
    device: core_defs.Device
    dtype: core_defs.DType[_ScalarT]
    shape: BufferShape
    strides: Tuple[int, ...]
    layout_map: BufferLayoutMap
    byte_offset: int
    byte_alignment: int
    aligned_index: Tuple[int, ...]
    ndarray: _NDBufferT = dataclasses.field(hash=False)

    @property
    def ndim(self):
        """Order of the tensor (`len(tensor_buffer.shape)`)."""
        return len(self.shape)

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        try:
            return self.ndarray.__array__(dtype=dtype)
        except (AttributeError, TypeError):
            raise TypeError("Cannot export tensor buffer as NumPy array.")

    @property
    def __cuda_array_interface__(self) -> xtyping.CUDAArrayInterfaceTypedDict:
        try:
            return self.ndarray.__cuda_array_interface__
        except (AttributeError, TypeError):
            raise TypeError("Cannot export tensor buffer to CUDA array interface.")

    def __dlpack__(self) -> xtyping.PyCapsule:
        try:
            return self.ndarray.__dlpack__()
        except (AttributeError, TypeError):
            raise TypeError("Cannot export tensor buffer to DLPack.")

    def __dlpack_device__(self) -> xtyping.DLPackDevice:
        try:
            return self.ndarray.__dlpack_device__()
        except (AttributeError, TypeError):
            raise TypeError("Cannot extract DLPack device from tensor buffer.")


#: Registry of allocators for each device type.
device_allocators: dict[core_defs.DeviceType, BufferAllocator] = {}


class BufferAllocator(Protocol[_NDBufferT]):
    """Protocol for buffer allocators."""

    @property
    def device_type(self) -> core_defs.DeviceType:
        ...

    def allocate(
        self,
        shape: Sequence[int],
        dtype: core_defs.DType[_ScalarT],
        layout_map: Sequence[int],
        device: core_defs.Device,
        byte_alignment: Optional[int] = None,
        aligned_index: Sequence[int] = None,
    ) -> TensorBuffer[_NDBufferT, _ScalarT]:
        """
        Allocate a TensorBuffer with the given shape, layout and alignment settings.

        Args:
            device: Device where the buffer is allocated.
            dtype: Data type descriptor.
            shape: Tensor dimensions.
            layout_map: layout of the dimensions in the buffer.
                layout_map[i] = j means that the i-th dimension of the tensor
                corresponds to the j-th dimension of the buffer.
            byte_alignment: Alignment (in bytes) of the first valid element.
            aligned_index: N-dimensional index of the first aligned element.
        """
        ...


@dataclasses.dataclass(frozen=True, init=False)
class _BaseNDArrayBufferAllocator(Generic[_NDBufferT]):
    """Base class for buffer allocators using NumPy-like modules."""

    def allocate(
        self,
        shape: Sequence[int],
        dtype: core_defs.DType[_ScalarT],
        layout_map: Sequence[int],
        device: core_defs.Device,
        byte_alignment: Optional[int] = None,
        aligned_index: Sequence[int] = None,
    ) -> TensorBuffer[_NDBufferT, _ScalarT]:
        if not is_valid_shape(shape):
            raise ValueError(f"Invalid shape {shape}")
        ndim = len(shape)
        if len(layout_map) != ndim or not is_valid_layout_map(layout_map):
            raise ValueError(f"Invalid layout_map {layout_map} for shape {shape}")

        # Compute number of items inside an aligned block
        item_size = dtype.byte_size
        if math.gcd(byte_alignment, item_size) not in (byte_alignment, item_size):
            raise ValueError(
                f"Incompatible 'byte_alignment' ({byte_alignment}) and 'dtype' size ({item_size})"
            )
        items_per_aligned_block = (byte_alignment // item_size) or 1

        # Compute the padding required in the contiguous dimension to get aligned blocks
        dims_layout = [layout_map.index(i) for i in range(len(shape))]
        padded_shape: Sequence[int] = list(shape)
        padded_shape[dims_layout[-1]] = (
            math.ceil(shape[dims_layout[-1]] / items_per_aligned_block) * items_per_aligned_block
        )
        padded_shape = tuple(shape)
        total_length = item_size * functools.reduce(operator.mul, padded_shape) + (
            byte_alignment - 1
        )

        # Compute strides
        strides = [item_size] * len(shape)
        accumulator = item_size
        for i in range(len(shape) - 2, -1, -1):
            accumulator = strides[dims_layout[i]] = accumulator * padded_shape[dims_layout[i + 1]]

        # Allocate total size
        buffer, memory_address = self._raw_alloc(total_length, device)

        # Compute final byte offset to align the requested buffer index
        aligned_index = tuple(aligned_index or ([0] * len(shape)))
        aligned_index_offset = (
            items_per_aligned_block
            * (int(math.ceil(aligned_index[dims_layout[-1]] / items_per_aligned_block)))
            - aligned_index[dims_layout[-1]]
        ) * item_size

        allocation_mismatch = (memory_address % byte_alignment) // item_size
        byte_offset = item_size * (
            (aligned_index_offset - allocation_mismatch) % items_per_aligned_block
        )

        # Create shaped view from buffer
        ndarray = self._tensorize(
            buffer, dtype, shape, padded_shape, item_size, strides, byte_offset
        )

        return TensorBuffer(
            buffer=buffer,
            memory_address=memory_address,
            device=device,
            dtype=dtype,
            shape=shape,
            strides=strides,
            layout_map=layout_map,
            byte_offset=byte_offset,
            byte_alignment=byte_alignment,
            aligned_index=aligned_index,
            ndarray=ndarray,
        )

    @abc.abstractmethod
    def _raw_alloc(self, length: int, device: core_defs.Device) -> tuple[_NDBufferT, int]:
        pass

    @abc.abstractmethod
    def _tensorize(
        self,
        buffer: _NDBufferT,
        dtype: core_defs.DType[_ScalarT],
        shape: BufferShape,
        allocated_shape: BufferShape,
        item_size: int,
        strides: Sequence[int],
        byte_offset: int,
    ) -> _NDBufferT:
        pass


if TYPE_CHECKING:

    class _NumPyLikeModule(Protocol[_NDBufferT]):
        class _NumPyLibModule(Protocol):
            class _NumPyLibStridesModule(Protocol):
                def as_strided(ndarray: _NDBufferT, **kwargs: Any) -> _NDBufferT:
                    ...

            stride_tricks: _NumPyLibStridesModule

        lib: _NumPyLibModule

        def empty(self, shape: BufferShape, dtype: np.dtype) -> np.ndarray:
            ...

        def byte_bounds(self, ndarray: _NDBufferT) -> tuple[int, int]:
            ...


@dataclasses.dataclass(frozen=True)
class NumPyLikeArrayBufferAllocator(_BaseNDArrayBufferAllocator[_NDBufferT]):
    device_type: core_defs.DeviceType
    xp: _NumPyLikeModule[_NDBufferT]

    def _raw_alloc(self, length: int, device: core_defs.Device) -> tuple[_NDBufferT, int]:
        if device.device_type != core_defs.DeviceType.CPU or device.device_id != 0:
            raise ValueError(f"Unsupported device {device} for memory allocation")

        buffer = self.xp.empty(shape=(length,), dtype=np.uint8)
        return buffer, self.xp.byte_bounds(buffer)[0]

    def _tensorize(
        self,
        buffer: _NDBufferT,
        dtype: core_defs.DType[_ScalarT],
        shape: BufferShape,
        allocated_shape: BufferShape,
        item_size: int,
        strides: Sequence[int],
        byte_offset: int,
    ) -> _NDBufferT:
        aligned_buffer = buffer[byte_offset : byte_offset + math.prod(allocated_shape) * item_size]
        flat_ndarray = aligned_buffer.view(dtype=np.dtype(dtype))
        tensor_view = self.xp.lib.stride_tricks.as_strided(
            flat_ndarray, shape=allocated_shape, strides=strides
        )
        if len(shape) and shape != allocated_shape:
            shape_slices = tuple(slice(0, s, None) for s in shape)
            tensor_view = tensor_view[shape_slices]

        return tensor_view


device_allocators[core_defs.DeviceType.CPU] = NumPyLikeArrayBufferAllocator(
    device_type=core_defs.DeviceType.CPU,
    xp=np,
)

if cp:
    device_allocators[core_defs.DeviceType.CUDA] = NumPyLikeArrayBufferAllocator(
        device_type=core_defs.DeviceType.CPU,
        xp=cp,
    )


def allocate(
    shape: Sequence[int],
    dtype: core_defs.DType[_ScalarT],
    layout_map: BufferLayoutMap,
    *,
    byte_alignment: Optional[int] = None,
    aligned_index: Sequence[int] = None,
    device: core_defs.Device = None,
    allocator: BufferAllocator = None,
) -> TensorBuffer:
    """Allocate a TensorBuffer with the given settings on the given device."""

    if device is None and allocator is None:
        raise ValueError("No 'device' or 'allocator' specified")
    device = device or allocator.device_type
    allocator = allocator or device_allocators[device.device_type]
    if device.device_type != allocator.device_type:
        raise ValueError(f"Device {device} and allocator {allocator} are incompatible")

    return allocator.allocate(
        shape=shape,
        dtype=dtype,
        layout_map=layout_map,
        byte_alignment=byte_alignment,
        aligned_index=aligned_index,
        device=device,
    )
