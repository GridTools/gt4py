# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import collections.abc
import dataclasses
import functools
import math
import operator
import types

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.eve.extended_typing import (
    Any,
    Callable,
    Generic,
    NewType,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
    TypeGuard,
    Union,
)


try:
    import cupy as cp
except ImportError:
    cp = None


_NDBuffer: TypeAlias = Union[
    # TODO: add `xtyping.Buffer` once we update typing_extensions
    xtyping.ArrayInterface, xtyping.CUDAArrayInterface, xtyping.DLPackBuffer
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


class BufferAllocator(Protocol[core_defs.DeviceTypeT]):
    """Protocol for buffer allocators."""

    @property
    def device_type(self) -> core_defs.DeviceTypeT: ...

    def allocate(
        self,
        shape: Sequence[core_defs.IntegralScalar],
        dtype: core_defs.DType[core_defs.ScalarT],
        device_id: int,
        layout_map: BufferLayoutMap,
        byte_alignment: int,
        aligned_index: Optional[Sequence[int]] = None,
    ) -> core_defs.MutableNDArrayObject:
        """
        Allocate an NDArrayObject with the given shape, layout and alignment settings.

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


@dataclasses.dataclass(frozen=True, init=False)
class _BaseNDArrayBufferAllocator(abc.ABC, Generic[core_defs.DeviceTypeT]):
    """Base class for buffer allocators using NumPy-like modules."""

    @property
    @abc.abstractmethod
    def device_type(self) -> core_defs.DeviceTypeT:
        pass

    def allocate(
        self,
        shape: Sequence[core_defs.IntegralScalar],
        dtype: core_defs.DType[core_defs.ScalarT],
        device_id: int,
        layout_map: BufferLayoutMap,
        byte_alignment: int,
        aligned_index: Optional[Sequence[int]] = None,
    ) -> core_defs.MutableNDArrayObject:
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
        buffer = self.malloc(total_length, device_id)
        memory_address = self.array_utils.byte_bounds(buffer)[0]

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

        return self.tensorize(buffer, dtype, shape, padded_shape, item_size, strides, byte_offset)

    @property
    @abc.abstractmethod
    def array_utils(self) -> ArrayUtils:
        pass

    @abc.abstractmethod
    def malloc(self, length: int, device_id: int) -> _NDBuffer:
        pass

    @abc.abstractmethod
    def tensorize(
        self,
        buffer: _NDBuffer,
        dtype: core_defs.DType[core_defs.ScalarT],
        shape: core_defs.TensorShape,
        allocated_shape: core_defs.TensorShape,
        item_size: int,
        strides: Sequence[int],
        byte_offset: int,
    ) -> core_defs.MutableNDArrayObject:
        """Create shaped view from buffer."""
        pass


@dataclasses.dataclass
class ArrayUtils:
    array_ns: types.ModuleType
    empty: Callable[..., _NDBuffer]
    byte_bounds: Callable[[_NDBuffer], Tuple[int, int]]
    as_strided: Callable[..., core_defs.MutableNDArrayObject]


numpy_array_utils = ArrayUtils(
    array_ns=np,
    empty=np.empty,
    byte_bounds=np.byte_bounds if hasattr(np, "byte_bounds") else np.lib.array_utils.byte_bounds,  # type: ignore  # noqa: NPY201
    as_strided=np.lib.stride_tricks.as_strided,  # type: ignore[arg-type]  # as_strided signature is just a sketch
)

cupy_array_utils = None

if cp is not None:
    cupy_array_utils = ArrayUtils(
        array_ns=cp,
        empty=cp.empty,
        byte_bounds=cp.byte_bounds
        if hasattr(cp, "byte_bounds")
        else cp.lib.array_utils.byte_bounds,
        as_strided=cp.lib.stride_tricks.as_strided,  # type: ignore[arg-type]  # as_strided signature is just a sketch
    )


@dataclasses.dataclass(frozen=True, init=False)
class NDArrayBufferAllocator(_BaseNDArrayBufferAllocator[core_defs.DeviceTypeT]):
    _device_type: core_defs.DeviceTypeT
    _array_utils: ArrayUtils

    def __init__(self, device_type: core_defs.DeviceTypeT, array_utils: ArrayUtils):
        object.__setattr__(self, "_device_type", device_type)
        object.__setattr__(self, "_array_utils", array_utils)

    @property
    def device_type(self) -> core_defs.DeviceTypeT:
        return self._device_type

    @property
    def array_utils(self) -> ArrayUtils:
        return self._array_utils

    def malloc(self, length: int, device_id: int) -> _NDBuffer:
        if self.device_type == core_defs.DeviceType.CPU and device_id != 0:
            raise ValueError(f"Unsupported device ID {device_id} for CPU memory allocation")

        shape = (length,)
        assert core_defs.is_valid_tensor_shape(shape)  # for mypy
        out = self._array_utils.empty(shape=tuple(shape), dtype=np.dtype(np.uint8))
        return out

    def tensorize(
        self,
        buffer: _NDBuffer,
        dtype: core_defs.DType[core_defs.ScalarT],
        shape: core_defs.TensorShape,
        allocated_shape: core_defs.TensorShape,
        item_size: int,
        strides: Sequence[int],
        byte_offset: int,
    ) -> core_defs.MutableNDArrayObject:
        aligned_buffer = buffer[byte_offset : byte_offset + math.prod(allocated_shape) * item_size]  # type: ignore[index] # TODO(egparedes): should we extend `_NDBuffer`s to cover __getitem__?
        flat_ndarray = aligned_buffer.view(dtype=np.dtype(dtype))
        tensor_view = self._array_utils.as_strided(
            flat_ndarray, shape=allocated_shape, strides=strides
        )
        if len(shape) and shape != allocated_shape:
            shape_slices = tuple(slice(0, s, None) for s in shape)
            tensor_view = tensor_view[shape_slices]

        return tensor_view
