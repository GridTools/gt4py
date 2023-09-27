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
from typing import Optional, Sequence, cast

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.storage import allocators as core_allocators


class FieldAllocatorInterface:
    @property
    def __gt_device_type__(self) -> core_defs.DeviceType:
        raise NotImplementedError()

    @abc.abstractmethod
    def __gt_allocate__(
        self,
        domain: common.Domain,
        dtype: core_defs.DType,
        device_id: int = 0,
        aligned_index: Optional[Sequence[common.NamedIndex]] = None,  # absolute position
    ) -> core_defs.NDArrayObject:
        raise NotImplementedError()


def _horizontal_first_layout_map(
    dims: Sequence[common.Dimension],
) -> core_allocators.BufferLayoutMap:
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


class DefaultCPUAllocator(FieldAllocatorInterface):
    """
    Allocates a buffer a NumPy buffer for CPU.

    Stride 1 dimension will be the first horizontal dimension.
    """

    def __gt_allocate__(
        self,
        domain: common.Domain,
        dtype: core_defs.DType,
        device_id: int = 0,
        aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    ) -> core_defs.NDArrayObject:
        shape = domain.shape
        layout_map = _horizontal_first_layout_map(domain.dims)
        byte_alignment = 64  # TODO check
        assert aligned_index is None  # TODO

        return (
            core_allocators.NDArrayBufferAllocator(core_defs.DeviceType.CPU, np)
            .allocate(shape, dtype, layout_map, device_id, byte_alignment, aligned_index)
            .ndarray
        )


class DefaultCUDAAllocator(FieldAllocatorInterface):
    def __gt_allocate__(
        self,
        domain: common.Domain,
        dtype: core_defs.DType,
        device_id: int = 0,
        aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    ) -> core_defs.NDArrayObject:
        try:
            import cupy as cp
        except ImportError:
            raise RuntimeError("CuPy is not available.")
        shape = domain.shape
        layout_map = _horizontal_first_layout_map(domain.dims)
        byte_alignment = 128  # TODO double check
        assert aligned_index is None  # TODO

        return (
            core_allocators.NDArrayBufferAllocator(core_defs.DeviceType.CUDA, cp)
            .allocate(shape, dtype, layout_map, device_id, byte_alignment, aligned_index)
            .ndarray
        )
