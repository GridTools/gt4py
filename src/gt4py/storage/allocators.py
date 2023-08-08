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
import collections.abc
import dataclasses
import functools
import math
import operator

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.eve.extended_typing import (
    TYPE_CHECKING,
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
    Union,
    cast,
)


try:
    import cupy as cp
except ImportError:
    cp = None


_ScalarT = TypeVar("_ScalarT", bound=core_defs.Scalar)


_NDBuffer: TypeAlias = Union[
    xtyping.ArrayInterface,
    xtyping.CUDAArrayInterface,
    xtyping.DLPackBuffer,
]


#: Tuple of positive integers encoding a permutation of the dimensions.
BufferLayoutMap = NewType("BufferLayoutMap", Sequence[core_defs.PositiveIntegral])


def is_valid_layout_map(value: Sequence[Any]) -> TypeGuard[BufferLayoutMap]:
    dims = list(range(len(value)))
    return (
        isinstance(value, collections.abc.Sequence)
        and len(value) == len(set(value))
        and all(i in dims for i in value)
    )


@dataclasses.dataclass(frozen=True)
class TensorBuffer(Generic[core_defs.NDArrayObjectT, _ScalarT]):
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

    buffer: _NDBuffer = dataclasses.field(hash=False)
    memory_address: int
    device: core_defs.Device
    dtype: core_defs.DType[_ScalarT]
    shape: core_defs.TensorShape
    strides: Tuple[int, ...]
    layout_map: BufferLayoutMap
    byte_offset: int
    byte_alignment: int
    aligned_index: Tuple[int, ...]
    ndarray: core_defs.NDArrayObjectT = dataclasses.field(hash=False)

    @property
    def ndim(self):
        """Order of the tensor (`len(tensor_buffer.shape)`)."""
        return len(self.shape)

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        if not hasattr(self.ndarray, "__array__"):
            raise TypeError("Cannot export tensor buffer as NumPy array.")

        return self.ndarray.__array__(dtype=dtype)  # type: ignore[call-overload] # TODO(egparades): figure out the mypy fix

    @property
    def __cuda_array_interface__(self) -> xtyping.CUDAArrayInterfaceTypedDict:
        if not hasattr(self.ndarray, "__cuda_array_interface__"):
            raise TypeError("Cannot export tensor buffer to CUDA array interface.")
        return self.ndarray.__cuda_array_interface__

    def __dlpack__(self) -> xtyping.PyCapsule:
        if not hasattr(self.ndarray, "__dlpack__"):
            raise TypeError("Cannot export tensor buffer to DLPack.")
        return self.ndarray.__dlpack__()

    def __dlpack_device__(self) -> xtyping.DLPackDevice:
        if not hasattr(self.ndarray, "__dlpack_device__"):
            raise TypeError("Cannot extract DLPack device from tensor buffer.")
        return self.ndarray.__dlpack_device__()


class BufferAllocator(Protocol[core_defs.NDArrayObjectT]):
    """Protocol for buffer allocators."""

    @property
    def device_type(self) -> core_defs.DeviceType:
        ...

    def allocate(
        self,
        shape: Sequence[core_defs.IntegralScalar],
        dtype: core_defs.DType[_ScalarT],
        layout_map: BufferLayoutMap,
        device: core_defs.Device,
        byte_alignment: int,
        aligned_index: Optional[Sequence[int]] = None,
    ) -> TensorBuffer[core_defs.NDArrayObjectT, _ScalarT]:
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
class _BaseNDArrayBufferAllocator(abc.ABC, Generic[core_defs.NDArrayObjectT]):
    """Base class for buffer allocators using NumPy-like modules."""

    def allocate(
        self,
        shape: Sequence[core_defs.IntegralScalar],
        dtype: core_defs.DType[_ScalarT],
        layout_map: BufferLayoutMap,
        device: core_defs.Device,
        byte_alignment: int,
        aligned_index: Optional[Sequence[int]] = None,
    ) -> TensorBuffer[core_defs.NDArrayObjectT, _ScalarT]:
        if not core_defs.is_valid_tensor_shape(shape):
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
        padded_shape_lst = list(shape)
        if ndim > 0:
            padded_shape_lst[dims_layout[-1]] = (
                math.ceil(shape[dims_layout[-1]] / items_per_aligned_block)
                * items_per_aligned_block
            )
        padded_shape = tuple(padded_shape_lst)
        assert core_defs.is_valid_tensor_shape(padded_shape)
        total_length = item_size * functools.reduce(operator.mul, padded_shape, 1) + (
            byte_alignment - 1
        )  # the worst case for misalignment is `byte_alignment - 1`

        # Compute strides
        strides_lst = [item_size] * len(shape)
        accumulator = item_size
        for i in range(len(shape) - 2, -1, -1):
            accumulator = strides_lst[dims_layout[i]] = (
                accumulator * padded_shape[dims_layout[i + 1]]
            )
        strides = tuple(strides_lst)

        # Allocate total size
        buffer = self.raw_alloc(total_length, device)
        memory_address = self.array_ns.byte_bounds(buffer)[0]

        # Compute final byte offset to align the requested buffer index
        aligned_index = tuple(aligned_index or ([0] * len(shape)))
        aligned_index_offset = (
            (
                items_per_aligned_block
                * (int(math.ceil(aligned_index[dims_layout[-1]] / items_per_aligned_block)))
                - aligned_index[dims_layout[-1]]
            )
            * item_size
            if ndim > 0
            else 0
        )

        allocation_mismatch_offset = (
            byte_alignment - memory_address % byte_alignment
        ) % byte_alignment
        byte_offset = (aligned_index_offset + allocation_mismatch_offset) % byte_alignment

        # Create shaped view from buffer
        ndarray = self.tensorize(
            buffer, dtype, shape, padded_shape, item_size, strides, byte_offset
        )

        if device.device_type == core_defs.DeviceType.ROCM:
            # until we can rely on dlpack
            ndarray.__hip_array_interface__ = {  # type: ignore[attr-defined]
                "shape": ndarray.shape,  # type: ignore[union-attr]
                "typestr": ndarray.dtype.descr[0][1],  # type: ignore[union-attr]
                "descr": ndarray.dtype.descr,  # type: ignore[union-attr]
                "stream": 1,
                "version": 3,
                "strides": ndarray.strides,  # type: ignore[union-attr, attr-defined]
                "data": (ndarray.data.ptr, False),  # type: ignore[union-attr, attr-defined]
            }

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

    @property
    @abc.abstractmethod
    def array_ns(self) -> _NumPyLikeNamespace[core_defs.NDArrayObjectT]:
        pass

    @abc.abstractmethod
    def raw_alloc(self, length: int, device: core_defs.Device) -> _NDBuffer:
        pass

    @abc.abstractmethod
    def tensorize(
        self,
        buffer: _NDBuffer,
        dtype: core_defs.DType[_ScalarT],
        shape: core_defs.TensorShape,
        allocated_shape: core_defs.TensorShape,
        item_size: int,
        strides: Sequence[int],
        byte_offset: int,
    ) -> core_defs.NDArrayObjectT:
        pass


if TYPE_CHECKING:

    class _NumPyLikeNamespace(Protocol[core_defs.NDArrayObjectT]):
        class _NumPyLibModule(Protocol):
            class _NumPyLibStridesModule(Protocol):
                def as_strided(
                    self, ndarray: core_defs.NDArrayObjectT, **kwargs: Any
                ) -> core_defs.NDArrayObjectT:
                    ...

            stride_tricks: _NumPyLibStridesModule

        lib: _NumPyLibModule

        def empty(self, shape: core_defs.TensorShape, dtype: np.dtype) -> core_defs.NDArrayObjectT:
            ...

        def byte_bounds(self, ndarray: _NDBuffer) -> tuple[int, int]:
            ...


@dataclasses.dataclass(frozen=True)
class NumPyLikeArrayBufferAllocator(_BaseNDArrayBufferAllocator[core_defs.NDArrayObjectT]):
    device_type: core_defs.DeviceType
    array_ns_ref: _NumPyLikeNamespace[core_defs.NDArrayObjectT]

    @property
    def array_ns(self) -> _NumPyLikeNamespace[core_defs.NDArrayObjectT]:
        return self.array_ns_ref

    def raw_alloc(self, length: int, device: core_defs.Device) -> _NDBuffer:
        if device.device_type != core_defs.DeviceType.CPU and device.device_id != 0:
            raise ValueError(f"Unsupported device {device} for memory allocation")

        shape = (length,)
        assert core_defs.is_valid_tensor_shape(shape)  # for mypy
        return cast(
            _NDBuffer, self.array_ns.empty(shape=shape, dtype=np.dtype(np.uint8))
        )  # TODO(havogt): figure out how we type this properly

    def tensorize(
        self,
        buffer: _NDBuffer,
        dtype: core_defs.DType[_ScalarT],
        shape: core_defs.TensorShape,
        allocated_shape: core_defs.TensorShape,
        item_size: int,
        strides: Sequence[int],
        byte_offset: int,
    ) -> core_defs.NDArrayObjectT:
        aligned_buffer = buffer[byte_offset : byte_offset + math.prod(allocated_shape) * item_size]  # type: ignore[index] # TODO(egparedes): should we extend `_NDBuffer`s to cover __getitem__?
        flat_ndarray = aligned_buffer.view(dtype=np.dtype(dtype))
        tensor_view = self.array_ns.lib.stride_tricks.as_strided(
            flat_ndarray, shape=allocated_shape, strides=strides
        )
        if len(shape) and shape != allocated_shape:
            shape_slices = tuple(slice(0, s, None) for s in shape)
            tensor_view = tensor_view[shape_slices]

        return tensor_view


#: Registry of allocators for each device type.
device_allocators: dict[core_defs.DeviceType, BufferAllocator] = {}

device_allocators[core_defs.DeviceType.CPU] = NumPyLikeArrayBufferAllocator(
    device_type=core_defs.DeviceType.CPU,
    array_ns_ref=cast(_NumPyLikeNamespace, np) if TYPE_CHECKING else np,
)

if cp:
    device_allocators[core_defs.DeviceType.CUDA] = NumPyLikeArrayBufferAllocator(
        device_type=core_defs.DeviceType.CUDA,
        array_ns_ref=cp,
    )
    device_allocators[core_defs.DeviceType.ROCM] = NumPyLikeArrayBufferAllocator(
        device_type=core_defs.DeviceType.ROCM,
        array_ns_ref=cp,
    )


def allocate(
    shape: Sequence[core_defs.IntegralScalar],
    dtype: core_defs.DType[_ScalarT],
    layout_map: BufferLayoutMap,
    *,
    byte_alignment: int,
    aligned_index: Optional[Sequence[int]] = None,
    device: Optional[core_defs.Device] = None,
    allocator: Optional[BufferAllocator] = None,
) -> TensorBuffer:
    """Allocate a TensorBuffer with the given settings on the given device."""
    if device is None and allocator is None:
        raise ValueError("No 'device' or 'allocator' specified")
    if device is None:
        assert allocator is not None  # for mypy
        device = core_defs.Device(allocator.device_type, 0)
    assert device is not None  # for mypy
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
