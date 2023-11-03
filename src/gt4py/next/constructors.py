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

from collections.abc import Sequence
from typing import Optional

import gt4py._core.definitions as core_defs
import gt4py.eve.extended_typing as xtyping
import gt4py.next.allocators as next_allocators
import gt4py.next.common as common
import gt4py.next.embedded.nd_array_field as nd_array_field
import gt4py.storage.cartesian.utils as storage_utils


def empty(
    domain: common.DomainLike,
    dtype: core_defs.DTypeLike = core_defs.Float64DType(()),
    *,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    allocator: Optional[next_allocators.FieldBufferAllocationTool] = None,
    device: Optional[core_defs.Device] = None,
) -> nd_array_field.NdArrayField:
    """Allocate an array of uninitialized (undefined) values with performance-optimal strides and alignment.

    !!!TODO!!!

    Parameters
    ----------
        shape : `Sequence` of `int`
            The shape of the resulting `ndarray`
        dtype :  DTypeLike, optional
            The dtype of the resulting `ndarray`

    Keyword Arguments
    -----------------
        backend : `str`
            The target backend for which the allocation is optimized.
        aligned_index: `Sequence` of `int`, optional
            Indicate the index of the resulting array that most commonly corresponds to the origin of the compute
            domain. If not passed, it is aligned at the data origin.
        dimensions: `Sequence` of `str`, optional
            Indicate the semantic meaning of the dimensions in the provided array. Only used for determining optimal
            strides, the information is not stored.

    Returns
    -------
        NumPy or CuPy ndarray
            With uninitialized values, padded and aligned to provide optimal performance for the given `backend` and
            `aligned_index`

    Raises
    -------
        TypeError
            If arguments of an unexpected type are specified.
        ValueError
            If illegal or inconsistent arguments are specified.
    """
    dtype = core_defs.dtype(dtype)
    buffer = next_allocators.allocate(
        domain, dtype, aligned_index=aligned_index, allocator=allocator, device=device
    )
    res = common.field(buffer.ndarray, domain=domain)
    assert common.is_mutable_field(res)
    assert isinstance(res, nd_array_field.NdArrayField)
    return res


def zeros(
    domain: common.DomainLike,
    dtype: core_defs.DTypeLike = core_defs.Float64DType(()),
    *,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    allocator: Optional[next_allocators.FieldBufferAllocatorProtocol] = None,
    device: Optional[core_defs.Device] = None,
) -> nd_array_field.NdArrayField:
    field = empty(
        domain=domain,
        dtype=dtype,
        aligned_index=aligned_index,
        allocator=allocator,
        device=device,
    )
    field[...] = field.dtype.scalar_type(0)
    return field


def ones(
    domain: common.DomainLike,
    dtype: core_defs.DTypeLike = core_defs.Float64DType(()),
    *,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    allocator: Optional[next_allocators.FieldBufferAllocatorProtocol] = None,
    device: Optional[core_defs.Device] = None,
) -> nd_array_field.NdArrayField:
    field = empty(
        domain=domain,
        dtype=dtype,
        aligned_index=aligned_index,
        allocator=allocator,
        device=device,
    )
    field[...] = field.dtype.scalar_type(1)
    return field


def full(
    domain: common.DomainLike,
    fill_value: core_defs.Scalar,
    dtype: Optional[core_defs.DTypeLike] = None,
    *,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    allocator: Optional[next_allocators.FieldBufferAllocatorProtocol] = None,
    device: Optional[core_defs.Device] = None,
) -> nd_array_field.NdArrayField:
    field = empty(
        domain=domain,
        dtype=dtype if dtype is not None else core_defs.dtype(type(fill_value)),
        aligned_index=aligned_index,
        allocator=allocator,
        device=device,
    )
    field[...] = field.dtype.scalar_type(fill_value)
    return field


def asfield(
    domain: common.DomainLike | Sequence[common.Dimension],
    data: core_defs.NDArrayObject,
    dtype: Optional[core_defs.DTypeLike] = None,
    *,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    allocator: Optional[next_allocators.FieldBufferAllocatorProtocol] = None,
    device: Optional[core_defs.Device] = None,
    # copy=False, TODO
) -> nd_array_field.NdArrayField:
    if isinstance(domain, Sequence) and all(isinstance(dim, common.Dimension) for dim in domain):
        if len(domain) != data.ndim:
            raise ValueError(
                f"Cannot construct `Field` from array of shape `{data.shape}` and domain `{domain}` "
            )
        actual_domain = common.domain(tuple(zip(domain, data.shape)))
    else:
        actual_domain = common.domain(domain)

    # TODO make sure we don't reallocate if its in correct layout and device
    shape = storage_utils.asarray(data).shape
    if shape != actual_domain.shape:
        raise ValueError(f"Cannot construct `Field` from array of shape `{shape}` ")
    if dtype is None:
        dtype = storage_utils.asarray(data).dtype
    dtype = core_defs.dtype(dtype)
    assert dtype.tensor_shape == ()  # TODO

    if allocator is device is None and xtyping.supports_dlpack(data):
        device = core_defs.Device(*data.__dlpack_device__())

    field = empty(
        domain=actual_domain,
        dtype=dtype,
        aligned_index=aligned_index,
        allocator=allocator,
        device=device,
    )

    field[...] = field.array_ns.asarray(data)

    return field