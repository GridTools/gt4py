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

from collections.abc import Sequence
from typing import Optional

import gt4py._core.definitions as core_defs
import gt4py.eve as eve
import gt4py.eve.extended_typing as xtyping
import gt4py.next.allocators as next_allocators
import gt4py.next.common as common
import gt4py.next.embedded.nd_array_field as nd_array_field
import gt4py.storage.cartesian.utils as storage_utils


@eve.utils.with_fluid_partial
def empty(
    domain: common.DomainLike,
    dtype: core_defs.DTypeLike = core_defs.Float64DType(()),
    *,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    allocator: Optional[next_allocators.FieldBufferAllocationTool] = None,
    device: Optional[core_defs.Device] = None,
) -> nd_array_field.NdArrayField:
    """Allocate Field of uninitialized (undefined) values with performance-optimal strides and alignment.

    Parameters
    ----------
        domain : `DomainLike`
            Mapping (or sequence of tuples) of `Dimension` to a range or range-like object.
            Determines the shape of the resulting field buffer.
        dtype : `DTypeLike`, optional
            The dtype of the resulting field buffer. Defaults to float64.

    Keyword Arguments
    -----------------
        aligned_index : `Sequence` of `int`, optional
            Indicate the index of the resulting array that most commonly corresponds to the origin of the compute
            domain. If not passed, it is byte-aligned at the data origin.
        allocator : `FieldBufferAllocationTool`
            An allocator, which knows how to optimize the memory layout for a given device. Or an object, which can provide an allocator. A `gtx.program_processors.otf_compile_executor.OTFBackend` is the most convenient choice in most use cases.
            Required if `device` is `None`. If both are valid, `allocator` will be chosen over the default device allocator.
        device : `Device`
            The device (CPU, type of accelerator) to optimize the memory layout for.
            Required if `allocator` is `None` and will cause the default device allocator to be used in that case.

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

    Examples
    --------
    Initialize a field in one dimension with a backend and a range domain:

    >>> from gt4py import next as gtx
    >>> from gt4py.next.program_processors.runners import roundtrip
    >>> IDim = gtx.Dimension("I")
    >>> a = gtx.empty({IDim: range(3, 10)}, allocator=roundtrip.backend)
    >>> a.shape
    (7,)

    Initialize with a device and an integer domain. It works like a shape with named dimensions:

    >>> from gt4py._core import definitions as core_defs
    >>> JDim = gtx.Dimension("J")
    >>> b = gtx.empty({IDim: 3, JDim: 3}, int, device=core_defs.Device(core_defs.DeviceType.CPU, 0))
    >>> b.ndarray
    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])
    >>> b.shape
    (3, 3)
    """
    dtype = core_defs.dtype(dtype)
    buffer = next_allocators.allocate(
        domain, dtype, aligned_index=aligned_index, allocator=allocator, device=device
    )
    res = common.field(buffer.ndarray, domain=domain)
    assert common.is_mutable_field(res)
    assert isinstance(res, nd_array_field.NdArrayField)
    return res


@eve.utils.with_fluid_partial
def zeros(
    domain: common.DomainLike,
    dtype: core_defs.DTypeLike = core_defs.Float64DType(()),
    *,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    allocator: Optional[next_allocators.FieldBufferAllocatorProtocol] = None,
    device: Optional[core_defs.Device] = None,
) -> nd_array_field.NdArrayField:
    """Create a Field containing all zeros with performance-optimal strides and alignment.

    Parameters
    ----------
    Same as `empty`

    Examples
    --------
    >>> from gt4py import next as gtx
    >>> from gt4py.next.program_processors.runners import roundtrip
    >>> IDim = gtx.Dimension("I")
    >>> gtx.zeros({IDim: range(3, 10)}, allocator=roundtrip.backend).ndarray
    array([0., 0., 0., 0., 0., 0., 0.])
    """
    field = empty(
        domain=domain,
        dtype=dtype,
        aligned_index=aligned_index,
        allocator=allocator,
        device=device,
    )
    field[...] = field.dtype.scalar_type(0)
    return field


@eve.utils.with_fluid_partial
def ones(
    domain: common.DomainLike,
    dtype: core_defs.DTypeLike = core_defs.Float64DType(()),
    *,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    allocator: Optional[next_allocators.FieldBufferAllocatorProtocol] = None,
    device: Optional[core_defs.Device] = None,
) -> nd_array_field.NdArrayField:
    """Create a Field containing all ones with performance-optimal strides and alignment.

    Parameters
    ----------
    Same as `empty`

    Examples
    --------
    >>> from gt4py import next as gtx
    >>> from gt4py.next.program_processors.runners import roundtrip
    >>> IDim = gtx.Dimension("I")
    >>> gtx.ones({IDim: range(3, 10)}, allocator=roundtrip.backend).ndarray
    array([1., 1., 1., 1., 1., 1., 1.])
    """
    field = empty(
        domain=domain,
        dtype=dtype,
        aligned_index=aligned_index,
        allocator=allocator,
        device=device,
    )
    field[...] = field.dtype.scalar_type(1)
    return field


@eve.utils.with_fluid_partial
def full(
    domain: common.DomainLike,
    fill_value: core_defs.Scalar,
    dtype: Optional[core_defs.DTypeLike] = None,
    *,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    allocator: Optional[next_allocators.FieldBufferAllocatorProtocol] = None,
    device: Optional[core_defs.Device] = None,
) -> nd_array_field.NdArrayField:
    """Create a Field where all values are set to `fill_value` with performance-optimal strides and alignment.

    Parameters
    ----------
        fill_value : `Scalar`
            Each point in the field will be initialized to this value.
        dtype : `DTypeLike`, optional
            The dtype of the resulting field buffer. Defaults to the dtype of `fill_value`.

    Refer to `empty` for the rest of the parameters.

    Examples
    --------
    >>> from gt4py import next as gtx
    >>> from gt4py.next.program_processors.runners import roundtrip
    >>> IDim = gtx.Dimension("I")
    >>> gtx.full({IDim: 3}, 5, allocator=roundtrip.backend).ndarray
    array([5, 5, 5])
    """
    field = empty(
        domain=domain,
        dtype=dtype if dtype is not None else core_defs.dtype(type(fill_value)),
        aligned_index=aligned_index,
        allocator=allocator,
        device=device,
    )
    field[...] = field.dtype.scalar_type(fill_value)
    return field


@eve.utils.with_fluid_partial
def as_field(
    domain: common.DomainLike | Sequence[common.Dimension],
    data: core_defs.NDArrayObject,
    dtype: Optional[core_defs.DTypeLike] = None,
    *,
    origin: Optional[dict[common.Dimension, int]] = None,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    allocator: Optional[next_allocators.FieldBufferAllocatorProtocol] = None,
    device: Optional[core_defs.Device] = None,
    # copy=False, TODO
) -> nd_array_field.NdArrayField:
    """Create a Field with performance-optimal strides and alignments from an array-like object.

    Parameters
    ----------
        domain : `DomainLike` | `Sequence[Dimension]`
            In addition to the values allowed in `empty`, can also just be a sequence of dimensions. By default, the sizes of each dimension will then be taken from the shape of `data`.
        data : `NDArrayObject`
            Array like data object to initialize the field with
        dtype: `DTypeLike`, optional
            The data type of the resulting field. Defaults to the same as `data`.

    Refer to `empty` for the rest of the parameters.

    Keyword Arguments
    -----------------
        origin : `dict[Dimension, int]`, optional
            Only allowed if `domain` is a sequence of dimensions. The indicated index in `data` will be the zero point of the resulting field.
        allocator : `gtx.allocators.FieldBufferAllocationTool`, optional
            Fully optional, in contrast to `empty`.
        device : `Device`, optional
            Fully optional, in contrast to `empty`, defaults to the same device as `data`.

    Refer to `empty` for the rest of the keyword arguments.

    Examples
    --------
    >>> import numpy as np
    >>> from gt4py import next as gtx
    >>> IDim = gtx.Dimension("I")
    >>> xdata = np.array([1, 2, 3])

    Automatic domain from just dimensions:

    >>> a = gtx.as_field([IDim], xdata)
    >>> a.ndarray
    array([1, 2, 3])
    >>> a.domain.ranges[0]
    UnitRange(0, 3)

    Shifted domain using origin:

    >>> b = gtx.as_field([IDim], xdata, origin={IDim: 1})
    >>> b.domain.ranges[0]
    UnitRange(-1, 2)

    Equivalent domain fully specified:

    >>> gtx.as_field({IDim: range(-1, 2)}, xdata).domain.ranges[0]
    UnitRange(-1, 2)
    """
    if isinstance(domain, Sequence) and all(isinstance(dim, common.Dimension) for dim in domain):
        if len(domain) != data.ndim:
            raise ValueError(
                f"Cannot construct `Field` from array of shape `{data.shape}` and domain `{domain}` "
            )
        if origin:
            if set(origin.keys()) - set(domain):
                raise ValueError(
                    f"Origin keys {set(origin.keys()) - set(domain)} not in domain {domain}"
                )
        else:
            origin = {}
        actual_domain = common.domain(
            [
                (d, (-(start_offset := origin.get(d, 0)), s - start_offset))
                for d, s in zip(domain, data.shape)
            ]
        )
    else:
        if origin:
            raise ValueError(f"Cannot specify origin for domain {domain}")
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
