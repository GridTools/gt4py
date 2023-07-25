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
import types

import numpy as np

from gt4py._core import scalars
from gt4py._core import definitions
from gt4py.eve import extended_typing as xtyping
from gt4py.eve.extended_typing import (
    Any,
    Generic,
    NewType,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeGuard,
    TypeVar,
    Union,
)

try:
    import cupy as cp
except ImportError:
    cp = None


_ScalarT = TypeVar("_ScalarT", bound=scalars.ScalarType)
_NDBufferT = TypeVar(
    "_NDBufferT",
    bound=Union[xtyping.ArrayInterface, xtyping.CUDAArrayInterface, xtyping.DLPackBuffer],
)

#: Tuple of positive integers encoding a tensor shape.
Shape = NewType("Shape", Tuple[int, ...])

#: Tuple of positive integers encoding a permutation of the dimensions.
LayoutMap = NewType("LayoutMap", Tuple[int, ...])


def is_valid_shape(value: tuple[int, ...]) -> TypeGuard[Shape]:
    return isinstance(value, tuple) and all(isinstance(v, int) and v > 0 for v in value)


def is_valid_layout_map(value: tuple[int, ...]) -> TypeGuard[LayoutMap]:
    dims = list(range(len(value)))
    return (
        isinstance(value, tuple) and len(value) == len(set(value)) and all(i in dims for i in value)
    )


@dataclasses.dataclass(frozen=True, slots=True)
class TensorBuffer(Generic[_NDBufferT, _ScalarT]):
    """
    N-dimensional (tensor-like) memory buffer.

    The actual class of the stored buffer and ndarray instances is
    represented in the `NDBufferT` parameter and might be any ndarray-like
    class from compatible array libraries (e.g. NumPy, CuPy, etc.)

    Attributes:
        buffer: Raw allocated buffer.
        memory_address: Memory address of the buffer.
        device: Device where the buffer is allocated.
        dtype: Data type descriptor.
        shape: Tuple with lengths of the corresponding tensor dimensions.
        ndim: Order of the tensor (`len(tensor_buffer.shape)`).
        strides: Tuple with sizes (in bytes) of the steps in each dimension.
        layout_map: Tuple with the order of the dimensions in the buffer.
            layout_map[i] = j means that the i-th dimension of the tensor
            corresponds to the j-th dimension of the buffer.
        byte_offset: Offset (in bytes) from the beginning of the buffer to
            the first valid element.
        byte_alignment: Alignment (in bytes) of the first valid element.
        aligned_index: N-dimensional index of the first aligned element.
        ndarray: N-dimensional tensor view of the allocated buffer.
    """

    buffer: _NDBufferT = dataclasses.field(hash=False)
    memory_address: int
    device: definitions.Device
    dtype: definitions.DType[_ScalarT]
    shape: Shape
    strides: tuple[int, ...]
    layout_map: LayoutMap
    byte_offset: int
    byte_alignment: int
    aligned_index: Tuple[int, ...]
    ndarray: _NDBufferT = dataclasses.field(hash=False)

    @property
    def ndim(self):
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

    def __dlpack_device__(self) -> xtyping.DLDevice:
        try:
            return self.ndarray.__dlpack_device__()
        except (AttributeError, TypeError):
            raise TypeError("Cannot extract DLPack device from tensor buffer.")


class BufferAllocator(Protocol[_NDBufferT]):
    device_type: definitions.DeviceType

    @abc.abstractmethod
    def allocate(
        self,
        device: definitions.Device,
        dtype: definitions.DType[_ScalarT],
        shape: Sequence[int],
        layout_map: Sequence[int],
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


ALLOCATORS: dict[definitions.DeviceType, BufferAllocator] = {}


@dataclasses.dataclass(frozen=True)
class BaseNDArrayBufferAllocator(BufferAllocator[_NDBufferT]):
    def allocate(
        self,
        device: definitions.Device,
        dtype: definitions.DType[_ScalarT],
        shape: Sequence[int],
        layout_map: Sequence[int],
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

        # Compute final byte offset to align the requested index
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
            buffer, shape, dtype, padded_shape, strides, byte_offset, item_size
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
    def _raw_alloc(self, length: int, device: definitions.Device) -> tuple[_NDBufferT, int]:
        pass

    @abc.abstractmethod
    def _tensorize(
        self,
        buffer: _NDBufferT,
        shape: Shape,
        dtype: definitions.DType[_ScalarT],
        allocated_shape: Shape,
        strides: Sequence[int],
        byte_offset: int,
        item_size: int,
    ) -> _NDBufferT:
        pass


class _NumPyLikeModule(Protocol):
    def empty(self, shape: Shape, dtype: np.dtype) -> np.ndarray:
        ...

    lib: Any


@dataclasses.dataclass(frozen=True)
class NumPyLikeArrayBufferAllocator(BaseNDArrayBufferAllocator[_NDBufferT]):
    device_type: definitions.DeviceType
    np_module: _NumPyLikeModule

    def _raw_alloc(self, length: int, device: definitions.Device) -> tuple[_NDBufferT, int]:
        nnp = self.np_module
        if device.device_type != definitions.DeviceType.CPU or device.device_id != 0:
            raise ValueError(f"Unsupported device {device} for memory allocation")

        buffer = nnp.empty(shape=(length,), dtype=np.uint8)
        return buffer, buffer.ctypes.data

    def _tensorize(
        self,
        buffer: _NDBufferT,
        shape: Shape,
        dtype: definitions.DType[_ScalarT],
        allocated_shape: Shape,
        strides: Sequence[int],
        byte_offset: int,
        item_size: int,
    ) -> _NDBufferT:
        nnp = self.np_module
        aligned_buffer = buffer[byte_offset : byte_offset + nnp.prod(allocated_shape) * item_size]
        flat_ndarray = aligned_buffer.view(dtype=np.dtype(dtype))
        tensor_view = nnp.lib.stride_tricks.as_strided(
            flat_ndarray, shape=allocated_shape, strides=strides
        )
        if len(shape) and shape != allocated_shape:
            shape_slices = tuple(slice(0, s, None) for s in shape)
            tensor_view = tensor_view[shape_slices]

        return tensor_view


ALLOCATORS[definitions.DeviceType.CPU] = NumPyLikeArrayBufferAllocator(
    device_type=definitions.DeviceType.CPU,
    np_module=np,
)

if cp:
    ALLOCATORS[definitions.DeviceType.CUDA] = NumPyLikeArrayBufferAllocator(
        device_type=definitions.DeviceType.CPU,
        np_module=cp,
    )


def allocate(
    shape: Sequence[int],
    dtype: definitions.DType[_ScalarT],
    layout_map: LayoutMap,
    *,
    device: definitions.Device = None,
    allocator: BufferAllocator = None,
    byte_alignment: Optional[int] = None,
    aligned_index: Sequence[int] = None,
) -> TensorBuffer:
    if device:
        if allocator:
            if device.type != allocator.device_type:
                raise ValueError(f"Device {device} and allocator {allocator} are incompatible")
        else:
            allocator = ALLOCATORS[device.device_type]
    elif allocator:
        device = allocator.device_type
    else:
        raise ValueError("No 'device' or 'allocator' specified")

    if not is_valid_shape(shape):
        raise ValueError(f"Invalid shape {shape}")

    return allocator.allocate(
        device=device,
        dtype=dtype,
        shape=shape,
        layout_map=layout_map,
        byte_alignment=byte_alignment,
        aligned_index=aligned_index,
    )
