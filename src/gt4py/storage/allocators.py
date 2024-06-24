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

"""Memory buffer allocation and management."""

from __future__ import annotations

import abc
import collections.abc
import dataclasses
import functools
import math
import operator

import numpy as np
import numpy.typing as npt

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.eve.extended_typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Literal,
    NewType,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeAlias,
    TypeGuard,
    Union,
    cast,
)


try:
    import cupy as cp
except ImportError:
    cp = None


_NDBuffer: TypeAlias = Union[
    # TODO: add `xtyping.Buffer` once we update typing_extensions
    xtyping.ArrayInterface,
    xtyping.CUDAArrayInterface,
    xtyping.DLPackBuffer,
]

#: Tuple of positive integers encoding a permutation of the dimensions, such that
#: layout_map[i] = j means that the i-th dimension of the tensor corresponds
#: to the j-th dimension in the (C-layout) buffer.
BufferLayoutMap = NewType("BufferLayoutMap", Sequence[core_defs.PositiveIntegral])


def is_valid_layout_map(value: Sequence[Any]) -> TypeGuard[BufferLayoutMap]:
    dims = list(range(len(value)))
    return (
        isinstance(value, collections.abc.Sequence)
        and len(value) == len(set(value))
        and all(i in dims for i in value)
    )


@dataclasses.dataclass(frozen=True)
class TensorBuffer(Generic[core_defs.DeviceTypeLiteralT, core_defs.ScalarT]):
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
        layout_map: Tuple with the order of the dimensions in the buffer
            layout_map[i] = j means that the i-th dimension of the tensor
            corresponds to the j-th dimension in the (C-layout) buffer.
        byte_offset: Offset (in bytes) from the beginning of the buffer to
            the first valid element.
        byte_alignment: Alignment (in bytes) of the first valid element.
        aligned_index: N-dimensional index of the first aligned element.
        ndarray: N-dimensional tensor view of the allocated buffer.
    """

    buffer: _NDBuffer = dataclasses.field(hash=False)
    memory_address: int
    device: core_defs.Device[core_defs.DeviceTypeLiteralT]
    dtype: core_defs.DType[core_defs.ScalarT]
    shape: core_defs.TensorShape
    strides: Tuple[int, ...]
    layout_map: BufferLayoutMap
    byte_offset: int
    byte_alignment: int
    aligned_index: Tuple[int, ...]
    ndarray: core_defs.NDArrayObject = dataclasses.field(hash=False)

    @property
    def ndim(self):
        """Order of the tensor (`len(tensor_buffer.shape)`)."""
        return len(self.shape)

    def __array__(self, dtype: Optional[npt.DTypeLike] = None, /) -> np.ndarray:
        if not xtyping.supports_array(self.ndarray):
            raise TypeError("Cannot export tensor buffer as NumPy array.")

        return self.ndarray.__array__(dtype)

    @property
    def __array_interface__(self) -> dict[str, Any]:
        if not xtyping.supports_array_interface(self.ndarray):
            raise TypeError("Cannot export tensor buffer to NumPy array interface.")

        return self.ndarray.__array_interface__

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        if not xtyping.supports_cuda_array_interface(self.ndarray):
            raise TypeError("Cannot export tensor buffer to CUDA array interface.")

        return self.ndarray.__cuda_array_interface__

    def __dlpack__(self, *, stream: Optional[int] = None) -> Any:
        if not hasattr(self.ndarray, "__dlpack__"):
            raise TypeError("Cannot export tensor buffer to DLPack.")
        return self.ndarray.__dlpack__(stream=stream)  # type: ignore[call-arg,arg-type]  # stream is not always supported

    def __dlpack_device__(self) -> xtyping.DLPackDevice:
        if not hasattr(self.ndarray, "__dlpack_device__"):
            raise TypeError("Cannot extract DLPack device from tensor buffer.")
        return self.ndarray.__dlpack_device__()


if TYPE_CHECKING:
    # TensorBuffer should be compatible with all the expected buffer interfaces
    __TensorBufferAsArrayInterfaceT: Type[xtyping.ArrayInterface] = TensorBuffer
    __TensorBufferAsCUDAArrayInterfaceT: Type[xtyping.CUDAArrayInterface] = TensorBuffer
    __TensorBufferAsDLPackBufferT: Type[xtyping.DLPackBuffer] = TensorBuffer


class BufferAllocator(Protocol[core_defs.DeviceTypeLiteralT]):
    """Protocol for tensor buffer allocators."""

    @property
    def device_type(self) -> core_defs.DeviceTypeLiteralT: ...

    def allocate(
        self,
        shape: Sequence[core_defs.IntegralScalar],
        dtype: core_defs.DType[core_defs.ScalarT],
        device_id: int,
        layout_map: BufferLayoutMap,
        byte_alignment: int,
        aligned_index: Optional[Sequence[int]] = None,
    ) -> TensorBuffer[core_defs.DeviceTypeLiteralT, core_defs.ScalarT]:
        """
        Allocate a TensorBuffer with the given shape, layout and alignment settings.

        Args:
            shape: Tensor dimensions.
            dtype: Data type descriptor.
            layout_map: layout of the dimensions in a buffer with C-layout (contiguous dimension is last).
              layout_map[i] = j means that the i-th dimension of the tensor
              corresponds to the j-th dimension of the buffer.
            device_id: Id of the device of `device_type` where the buffer is allocated.
            byte_alignment: Alignment (in bytes) of the first valid element.
            aligned_index: N-dimensional index of the first aligned element.
        """
        ...


class MemoryResourceHandler(Protocol[core_defs.DeviceTypeLiteralT]):
    """Interface for memory resource handlers."""

    @property
    @abc.abstractmethod
    def device_type(self) -> core_defs.DeviceTypeLiteralT: ...

    @abc.abstractmethod
    def address_of(self, buffer: _NDBuffer) -> int:
        """Get the address of a buffer."""

    @abc.abstractmethod
    def malloc(self, byte_size: int, device_id: int) -> _NDBuffer:
        """Allocate memory on a specific device.

        Args:
            byte_size: The size of the memory to allocate in bytes.
            device_id: The ID of the device.

        Returns:
            The allocated buffer.
        """

    @abc.abstractmethod
    def tensorize(
        self,
        buffer: _NDBuffer,
        dtype: core_defs.DType[core_defs.ScalarT],
        shape: core_defs.TensorShape,
        padded_shape: core_defs.TensorShape,
        strides: Sequence[int],
        byte_offset: int,
    ) -> core_defs.NDArrayObject:
        """Create an NDArray object from a buffer.

        Args:
            buffer: The buffer.
            dtype: The data type of the NDArray.
            shape: The shape of the NDArray.
            padded_shape: The actually allocated shape of the buffer (including padding).
            strides: The strides to be used in the result NDArray.
            byte_offset: The initial padding in the buffer to be ignored in the result NDArray.

        Returns:
            The created NDArray object.
        """


class NumPyMemoryResourceHandler(MemoryResourceHandler[Literal[core_defs.DeviceType.CPU]]):
    """Memory resource handler for allocating and managing memory on CPU using NumPy."""

    device_type: Final = core_defs.DeviceType.CPU
    array_ns: Final = np

    if hasattr(np, "byte_bounds"):

        @classmethod
        def address_of(cls, buffer: _NDBuffer) -> int:
            assert isinstance(buffer, np.ndarray)
            return np.byte_bounds(buffer)[0]  # noqa: NPY201 # np.byte_bounds moved in NumPy >= 2.0

    else:
        assert hasattr(np.lib.array_utils, "byte_bounds")  # type: ignore [attr-defined]  # Only NumPy >= 2.0

        @classmethod
        def address_of(cls, buffer: _NDBuffer) -> int:
            return np.lib.array_utils.byte_bounds(buffer)[0]  # type: ignore [attr-defined]  # Only NumPy >= 2.0

    @classmethod
    def malloc(cls, byte_size: int, device_id: int) -> _NDBuffer:
        if device_id != 0:
            raise ValueError(f"Unsupported device Id {device_id} for CPU memory allocation")
        return np.empty(shape=(byte_size,), dtype=np.uint8)

    @classmethod
    def tensorize(
        cls,
        buffer: _NDBuffer,
        dtype: core_defs.DType[core_defs.ScalarT],
        shape: core_defs.TensorShape,
        allocated_shape: core_defs.TensorShape,
        strides: Sequence[int],
        byte_offset: int,
    ) -> core_defs.NDArrayObject:
        assert isinstance(buffer, np.ndarray)
        item_size = dtype.byte_size
        aligned_buffer = buffer[byte_offset : byte_offset + math.prod(allocated_shape) * item_size]
        flat_ndarray = aligned_buffer.view(dtype=np.dtype(dtype))
        tensor_view = np.lib.stride_tricks.as_strided(
            flat_ndarray, shape=allocated_shape, strides=strides
        )
        if len(shape) and shape != allocated_shape:
            shape_slices = tuple(slice(0, s, None) for s in shape)
            tensor_view = tensor_view[shape_slices]

        return cast(core_defs.NDArrayObject, tensor_view)


@dataclasses.dataclass(frozen=True)
class NDArrayBufferAllocator(Generic[core_defs.DeviceTypeLiteralT]):
    """BufferAllocator implementation using NDArray objects from tensor libraries."""

    resource_handler: MemoryResourceHandler[core_defs.DeviceTypeLiteralT]

    @property
    def device_type(self) -> core_defs.DeviceTypeLiteralT:
        return self.resource_handler.device_type

    def allocate(
        self,
        shape: Sequence[core_defs.IntegralScalar],
        dtype: core_defs.DType[core_defs.ScalarT],
        device_id: int,
        layout_map: BufferLayoutMap,
        byte_alignment: int,
        aligned_index: Optional[Sequence[int]] = None,
    ) -> TensorBuffer[core_defs.DeviceTypeLiteralT, core_defs.ScalarT]:
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
        byte_size = item_size * functools.reduce(operator.mul, padded_shape, 1) + (
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
        buffer = self.resource_handler.malloc(byte_size, device_id)
        memory_address = self.resource_handler.address_of(buffer)

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
        ndarray = self.resource_handler.tensorize(
            buffer, dtype, shape, padded_shape, strides, byte_offset
        )

        if self.device_type == core_defs.DeviceType.ROCM:
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
            device=core_defs.Device(self.device_type, device_id),
            dtype=dtype,
            shape=shape,
            strides=strides,
            layout_map=layout_map,
            byte_offset=byte_offset,
            byte_alignment=byte_alignment,
            aligned_index=aligned_index,
            ndarray=ndarray,
        )


numpy_buffer_allocator = NDArrayBufferAllocator[core_defs.CPUDeviceTypeLiteral](
    resource_handler=NumPyMemoryResourceHandler
)

# -- CuPy / GPU allocation --
cupy_buffer_allocator: Union[
    None,
    NDArrayBufferAllocator[core_defs.CUDADeviceTypeLiteral],
    NDArrayBufferAllocator[core_defs.ROCMDeviceTypeLiteral],
] = None

CuPyDeviceType: Literal[None, core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM] = None
CuPyMemoryResourceHandler: Union[None, type[MemoryResourceHandler]] = None

if cp:

    class _CuPyMemoryResourceHandler(MemoryResourceHandler[core_defs.DeviceTypeLiteralT]):
        """Memory resource handler for allocating and managing memory on GPU using CuPy."""

        array_ns: Final = cp

        if hasattr(cp, "byte_bounds"):

            @classmethod
            def address_of(cls, buffer: _NDBuffer) -> int:
                assert isinstance(buffer, cp.ndarray)
                return cp.byte_bounds(buffer)[0]

        else:
            assert hasattr(cp.lib.array_utils, "byte_bounds")

            @classmethod
            def address_of(cls, buffer: _NDBuffer) -> int:
                return cp.lib.array_utils.byte_bounds(buffer)[0]

        @classmethod
        def malloc(cls, byte_size: int, device_id: int) -> _NDBuffer:
            with cp.cuda.Device(device_id):
                return cp.empty(shape=(byte_size,), dtype=np.uint8)

        @classmethod
        def tensorize(
            cls,
            buffer: _NDBuffer,
            dtype: core_defs.DType[core_defs.ScalarT],
            shape: core_defs.TensorShape,
            allocated_shape: core_defs.TensorShape,
            strides: Sequence[int],
            byte_offset: int,
        ) -> core_defs.NDArrayObject:
            assert isinstance(buffer, cp.ndarray)
            item_size = dtype.byte_size
            aligned_buffer = buffer[
                byte_offset : byte_offset + math.prod(allocated_shape) * item_size
            ]
            flat_ndarray = aligned_buffer.view(dtype=cp.dtype(dtype))
            tensor_view = cp.lib.stride_tricks.as_strided(
                flat_ndarray, shape=allocated_shape, strides=strides
            )
            if len(shape) and shape != allocated_shape:
                shape_slices = tuple(slice(0, s, None) for s in shape)
                tensor_view = tensor_view[shape_slices]

            return tensor_view

    if not cp.cuda.runtime.is_hip:

        class CuPyMemoryResourceHandler(
            _CuPyMemoryResourceHandler[core_defs.ROCMDeviceTypeLiteral]
        ):  # type: ignore[no-redef]
            """Memory resource handler for allocating and managing memory on ROCm GPUs using CuPy."""

            device_type: Final = core_defs.DeviceType.ROCM

        CuPyDeviceType = core_defs.DeviceType.ROCM

    else:

        class CuPyMemoryResourceHandler(
            _CuPyMemoryResourceHandler[core_defs.CUDADeviceTypeLiteral]
        ):  # type: ignore[no-redef]
            """Memory resource handler for allocating and managing memory on CUDA GPUs using CuPy."""

            device_type: Final = core_defs.DeviceType.CUDA

        CuPyDeviceType = core_defs.DeviceType.CUDA

    cupy_buffer_allocator = cast(
        Union[
            NDArrayBufferAllocator[core_defs.CUDADeviceTypeLiteral],
            NDArrayBufferAllocator[core_defs.ROCMDeviceTypeLiteral],
        ],
        NDArrayBufferAllocator[Literal[core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM]](  # type: ignore[type-var]
            resource_handler=CuPyMemoryResourceHandler  # type: ignore[arg-type]
        ),
    )
