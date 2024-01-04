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

from collections.abc import Mapping, Sequence
from typing import Optional, cast

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
    allocator: Optional[next_allocators.FieldBufferAllocationUtil] = None,
    device: Optional[core_defs.Device] = None,
) -> nd_array_field.NdArrayField:
    """Create a `Field` of uninitialized (undefined) values using the given (or device-default) allocator.

    This function supports partial binding of arguments, see :class:`eve.utils.partial` for details.

    Arguments:
        domain: Definition of the domain of the field (which fix the shape of the allocated field buffer).
            See :class:`gt4py.next.common.Domain` for details.
        dtype: Definition of the data type of the field. Defaults to `float64`.

    Keyword Arguments:
        aligned_index: Index in the definition domain which should be used as reference
            point for memory aligment computations. It can be set to the most common origin
            of computations in this domain (if known) for performance reasons.
        allocator: The allocator or allocator factory (e.g. backend) used for memory buffer
            allocation, which knows how to optimize the memory layout for a given device.
            Required if `device` is `None`. If both are valid, `allocator` will be chosen over
            the default device allocator.
        device: The device (CPU, type of accelerator) to optimize the memory layout for.
            Required if `allocator` is `None` and will cause the default device allocator
            to be used in that case.

    Returns:
        A field, backed by a buffer with memory layout as specified by allocator and alignment requirements.

    Raises:
        ValueError
            If illegal or inconsistent arguments are specified.

    Examples:
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
        >>> b.shape
        (3, 3)
    """
    dtype = core_defs.dtype(dtype)
    if allocator is None and device is None:
        device = core_defs.Device(core_defs.DeviceType.CPU, device_id=0)
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
    """Create a Field containing all zeros using the given (or device-default) allocator.

    This function supports partial binding of arguments, see :class:`eve.utils.partial` for details.
    See :func:`empty` for further details about the meaning of the arguments.

    Examples:
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
    """Create a Field containing all ones using the given (or device-default) allocator.

    This function supports partial binding of arguments, see :class:`eve.utils.partial` for details.
    See :func:`empty` for further details about the meaning of the arguments.

    Examples:
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
    """Create a Field where all values are set to `fill_value` using the given (or device-default) allocator.

    This function supports partial binding of arguments, see :class:`eve.utils.partial` for details.
    See :func:`empty` for further details about the meaning of the arguments.

    Arguments:
        domain: Definition of the domain of the field (and consequently of the shape of the allocated field buffer).
        fill_value: Each point in the field will be initialized to this value.
        dtype: Definition of the data type of the field. Defaults to the dtype of `fill_value`.

    Examples:
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
    origin: Optional[Mapping[common.Dimension, int]] = None,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    allocator: Optional[next_allocators.FieldBufferAllocatorProtocol] = None,
    device: Optional[core_defs.Device] = None,
    # copy=False, TODO
) -> nd_array_field.NdArrayField:
    """Create a Field from an array-like object using the given (or device-default) allocator.

    This function supports partial binding of arguments, see :class:`eve.utils.partial` for details.
    See :func:`empty` for further details about the meaning of the extra keyword arguments.

    Arguments:
        domain: Definition of the domain of the field (and consequently of the shape of the allocated field buffer).
            In addition to the values allowed in `empty`, it can also just be a sequence of dimensions,
            in which case the sizes of each dimension will then be taken from the shape of `data`.
        data: Array like data object to initialize the field with
        dtype: Definition of the data type of the field. Defaults to the same as `data`.

    Keyword Arguments:
        origin: Only allowed if `domain` is a sequence of dimensions. The indicated index in `data`
            will be the zero point of the resulting field.
        allocator: Fully optional, in contrast to `empty`.
        device: Fully optional, in contrast to `empty`, defaults to the same device as `data`.

    Examples:
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
        domain = cast(Sequence[common.Dimension], domain)
        if len(domain) != data.ndim:
            raise ValueError(
                f"Cannot construct 'Field' from array of shape '{data.shape}' and domain '{domain}'."
            )
        if origin:
            domain_dims = set(domain)
            if unknown_dims := set(origin.keys()) - domain_dims:
                raise ValueError(f"Origin keys {unknown_dims} not in domain {domain}.")
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
        actual_domain = common.domain(cast(common.DomainLike, domain))

    # TODO(egparedes): allow zero-copy construction (no reallocation) if buffer has
    #   already the correct layout and device.
    shape = storage_utils.asarray(data).shape
    if shape != actual_domain.shape:
        raise ValueError(f"Cannot construct 'Field' from array of shape '{shape}'.")
    if dtype is None:
        dtype = storage_utils.asarray(data).dtype
    dtype = core_defs.dtype(dtype)
    assert dtype.tensor_shape == ()  # TODO

    if (allocator is None) and (device is None) and xtyping.supports_dlpack(data):
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


@eve.utils.with_fluid_partial
def as_connectivity(
    domain: common.DomainLike | Sequence[common.Dimension],
    codomain: common.Dimension,
    data: core_defs.NDArrayObject,
    dtype: Optional[core_defs.DType] = None,
    *,
    allocator: Optional[next_allocators.FieldBufferAllocatorProtocol] = None,
    device: Optional[core_defs.Device] = None,
    # copy=False, TODO
) -> common.ConnectivityField:
    """
    Construct a connectivity field from the given domain, codomain, and data.

    Arguments:
        domain: The domain of the connectivity field. It can be either a `common.DomainLike` object or a
            sequence of `common.Dimension` objects.
        codomain: The codomain dimension of the connectivity field.
        data: The data used to construct the connectivity field.
        dtype: The data type of the connectivity field. If not provided, it will be inferred from the data.
        allocator: The allocator used to allocate the buffer for the connectivity field. If not provided,
            a default allocator will be used.
        device: The device on which the connectivity field will be allocated. If not provided, the default
            device will be used.

    Returns:
        The constructed connectivity field.

    Raises:
        ValueError: If the domain or codomain is invalid, or if the shape of the data does not match the domain shape.
    """
    if isinstance(domain, Sequence) and all(isinstance(dim, common.Dimension) for dim in domain):
        domain = cast(Sequence[common.Dimension], domain)
        if len(domain) != data.ndim:
            raise ValueError(
                f"Cannot construct 'Field' from array of shape '{data.shape}' and domain '{domain}'."
            )
        actual_domain = common.domain([(d, (0, s)) for d, s in zip(domain, data.shape)])
    else:
        actual_domain = common.domain(cast(common.DomainLike, domain))

    if not isinstance(codomain, common.Dimension):
        raise ValueError(f"Invalid codomain dimension '{codomain}'.")

    # TODO(egparedes): allow zero-copy construction (no reallocation) if buffer has
    #   already the correct layout and device.
    shape = storage_utils.asarray(data).shape
    if shape != actual_domain.shape:
        raise ValueError(f"Cannot construct 'Field' from array of shape '{shape}'.")
    if dtype is None:
        dtype = storage_utils.asarray(data).dtype
    dtype = core_defs.dtype(dtype)
    assert dtype.tensor_shape == ()  # TODO

    if (allocator is None) and (device is None) and xtyping.supports_dlpack(data):
        device = core_defs.Device(*data.__dlpack_device__())
    buffer = next_allocators.allocate(actual_domain, dtype, allocator=allocator, device=device)
    # TODO(havogt): consider addin MutableNDArrayObject
    buffer.ndarray[...] = storage_utils.asarray(data)  # type: ignore[index]
    connectivity_field = common.connectivity(
        buffer.ndarray, codomain=codomain, domain=actual_domain
    )
    assert isinstance(connectivity_field, nd_array_field.NdArrayConnectivityField)

    return connectivity_field
