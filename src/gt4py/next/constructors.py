# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import dataclasses
import functools
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, cast

import gt4py.eve as eve
import gt4py.next.common as common
import gt4py.next.custom_layout_allocators as next_allocators
import gt4py.next.embedded.nd_array_field as nd_array_field
from gt4py._core import (
    definitions as core_defs,
    ndarray_utils as core_ndarray_utils,
    types as core_types,
)
from gt4py.eve import extended_typing as xtyping


"""
Field construction API.

The user-facing API consists of the functions 'empty', 'zeros', 'ones', 'full' and 'as_field',
or the 'FieldConstructor' class for more advanced use cases.

These functions create GT4Py 'Field's backed by arrays created using a specified allocator, which can be either an array namespace
(e.g. 'numpy', 'cupy') or a GT4Py field buffer allocator (e.g. a backend).

This module deals with 3 concepts:
- allocating with a backing array API or 'FieldBufferAllocator'
- translating from absolute (domain) to relative (shape) indexing, e.g. for handling of 'aligned_index'
- translating from GT4Py dtypes and devices to the format expected by the backing array API.
"""

# Type to be used by the end-user
Allocator: TypeAlias = core_ndarray_utils.ArrayNamespace | next_allocators.FieldBufferAllocationUtil
"""Type for field memory allocators.

Accepts either:
- An array namespace following the Array API standard (e.g. ``numpy``, ``cupy``, ``jax.numpy``),
  which will be used directly for array creation.
- A GT4Py field buffer allocator or allocator factory (e.g. a backend), which controls
  memory layout and alignment.
"""

DEFAULT_DEVICE: core_defs.Device = core_defs.Device(core_defs.DeviceType.CPU, 0)
DEFAULT_DTYPE: core_defs.DType = core_defs.Float64DType(())


class FieldConstructor:
    """
    Public-facing field construction API.

    Delegates array creation to the composed `_array_constructor` and wraps the result in a Field.
    """

    _array_constructor: _FieldArrayConstructionNamespace

    def __init__(
        self,
        allocator: Allocator | None = None,
        aligned_index: Sequence[common.NamedIndex] | None = None,
        device: core_defs.Device | None = None,
    ):
        if allocator is None:
            if device is None:
                device = DEFAULT_DEVICE
            # Currently we use optimized CPU layout, but possibly we could also use standard numpy allocation
            allocator = next_allocators.device_allocators[device.device_type]

        if core_ndarray_utils.is_array_namespace(allocator):
            if aligned_index is not None:
                import warnings

                warnings.warn(
                    "`aligned_index` is not supported when using an array namespace allocator and will be ignored.",
                    stacklevel=2,
                )
            translated_device = (
                core_ndarray_utils.get_device_translator(allocator)(device)
                if device is not None
                else None
            )
            self._array_constructor = _ArrayApiCreationNamespace(
                array_ns=allocator, device=translated_device
            )
        elif next_allocators.is_field_allocation_tool(allocator):
            if next_allocators.is_field_allocator_factory(allocator):
                allocator = allocator.__gt_allocator__
            if device is not None and allocator.__gt_device_type__ != device.device_type:
                raise ValueError(
                    f"Allocator {allocator} is for device type {allocator.__gt_device_type__}, but device type {device.device_type} was specified."
                )
            self._array_constructor = _FieldBufferCreationNamespace(
                allocator=allocator,
                device=device,
                aligned_index=aligned_index,
            )
        else:
            raise ValueError(f"Invalid field allocator: {allocator}.")

    class _NdArrayConstructor(Protocol):
        def __call__(
            self, domain: common.Domain, *, dtype: core_defs.DType
        ) -> core_defs.NDArrayObject: ...

    def _construct_from_array(
        self,
        domain: common.DomainLike,
        dtype: core_defs.DTypeLike,
        ndarray_constructor: _NdArrayConstructor,
    ) -> nd_array_field.NdArrayField:
        domain = common.domain(domain)
        dtype = core_defs.dtype(dtype)
        res = common._field(ndarray_constructor(domain, dtype=dtype), domain=domain)
        assert isinstance(res, nd_array_field.NdArrayField)
        return res

    def empty(
        self,
        domain: common.DomainLike,
        *,
        dtype: core_defs.DTypeLike = DEFAULT_DTYPE,
    ) -> nd_array_field.NdArrayField:
        """Create a `Field` of uninitialized values. See :func:`empty` for details."""
        return self._construct_from_array(domain, dtype, self._array_constructor.empty)

    def zeros(
        self,
        domain: common.DomainLike,
        *,
        dtype: core_defs.DTypeLike = DEFAULT_DTYPE,
    ) -> nd_array_field.NdArrayField:
        """Create a `Field` containing all zeros. See :func:`zeros` for details."""
        return self._construct_from_array(domain, dtype, self._array_constructor.zeros)

    def ones(
        self,
        domain: common.DomainLike,
        *,
        dtype: core_defs.DTypeLike = DEFAULT_DTYPE,
    ) -> nd_array_field.NdArrayField:
        """Create a `Field` containing all ones. See :func:`ones` for details."""
        return self._construct_from_array(domain, dtype, self._array_constructor.ones)

    def full(
        self,
        domain: common.DomainLike,
        fill_value: core_defs.Scalar,
        *,
        dtype: core_defs.DTypeLike | None = None,
    ) -> nd_array_field.NdArrayField:
        """Create a `Field` filled with `fill_value`. See :func:`full` for details."""
        dtype = dtype if dtype is not None else type(fill_value)
        return self._construct_from_array(
            domain,
            dtype,
            lambda domain, dtype: self._array_constructor.full(
                domain, fill_value=fill_value, dtype=dtype
            ),
        )

    def as_field(
        self,
        domain: common.DomainLike | Sequence[common.Dimension],
        data: core_defs.NDArrayObject,
        *,
        dtype: core_defs.DTypeLike | None = None,
        origin: Mapping[common.Dimension, int] | None = None,
    ) -> nd_array_field.NdArrayField:
        """Create a `Field` from an array-like object. See :func:`as_field` for details."""
        if isinstance(domain, Sequence) and all(
            isinstance(dim, common.Dimension) for dim in domain
        ):
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
                raise ValueError(f"Cannot specify origin for a concrete domain {domain}")
            actual_domain = common.domain(cast(common.DomainLike, domain))

        if data.shape != actual_domain.shape:
            raise ValueError(f"Cannot construct 'Field' from array of shape '{data.shape}'.")

        if dtype is None:
            dtype = core_defs.dtype(data.dtype)

        return self._construct_from_array(
            actual_domain,
            dtype,
            lambda domain, dtype: self._array_constructor.asarray(domain, data, dtype=dtype),
        )


class _FieldArrayConstructionNamespace(abc.ABC):
    """
    Abstract interface for array creation operations on a specific device/backend.

    The translation from DomainLike and DTypeLike is already done in the public interface.
    """

    @abc.abstractmethod
    def empty(
        self, domain: common.Domain, *, dtype: core_defs.DType
    ) -> core_defs.NDArrayObject: ...

    @abc.abstractmethod
    def zeros(
        self, domain: common.Domain, *, dtype: core_defs.DType
    ) -> core_defs.NDArrayObject: ...

    @abc.abstractmethod
    def ones(self, domain: common.Domain, *, dtype: core_defs.DType) -> core_defs.NDArrayObject: ...

    @abc.abstractmethod
    def full(
        self,
        domain: common.Domain,
        fill_value: core_defs.Scalar,
        *,
        dtype: core_defs.DType,
    ) -> core_defs.NDArrayObject: ...

    @abc.abstractmethod
    def asarray(
        self,
        domain: common.Domain,
        data: core_defs.NDArrayObject,
        *,
        dtype: core_defs.DType,
    ) -> core_defs.NDArrayObject: ...


_ANS = TypeVar("_ANS", bound=core_ndarray_utils.ArrayNamespace)


@dataclasses.dataclass(frozen=True)
class _ArrayApiCreationNamespace(_FieldArrayConstructionNamespace, Generic[_ANS]):
    array_ns: _ANS
    # device in the format expected by the array namespace
    device: Any = None

    def _to_array_ns_dtype(self, dtype: core_defs.DType) -> Any:
        return getattr(self.array_ns, core_types.type_to_name[dtype.scalar_type])

    def empty(
        self,
        domain: common.Domain,
        *,
        dtype: core_defs.DType,
    ) -> core_defs.NDArrayObject:
        return self.array_ns.empty(
            domain.shape, dtype=self._to_array_ns_dtype(dtype), device=self.device
        )

    def zeros(
        self,
        domain: common.Domain,
        *,
        dtype: core_defs.DType,
    ) -> core_defs.NDArrayObject:
        return self.array_ns.zeros(
            domain.shape, dtype=self._to_array_ns_dtype(dtype), device=self.device
        )

    def ones(
        self,
        domain: common.Domain,
        *,
        dtype: core_defs.DType,
    ) -> core_defs.NDArrayObject:
        return self.array_ns.ones(
            domain.shape, dtype=self._to_array_ns_dtype(dtype), device=self.device
        )

    def full(
        self,
        domain: common.Domain,
        fill_value: core_defs.Scalar,
        *,
        dtype: core_defs.DType,
    ) -> core_defs.NDArrayObject:
        return self.array_ns.full(
            domain.shape,
            fill_value=fill_value,
            dtype=self._to_array_ns_dtype(dtype),
            device=self.device,
        )

    def asarray(
        self,
        domain: common.Domain,
        data: core_defs.NDArrayObject,
        *,
        dtype: core_defs.DType,
    ) -> core_defs.NDArrayObject:
        arr = self.array_ns.asarray(
            data, dtype=self._to_array_ns_dtype(dtype), device=self.device, copy=True
        )
        assert domain.shape == arr.shape, (
            f"pre-condition of `asarray` not met: {data.shape=} and {domain.shape} need to agree."
        )
        return arr


@dataclasses.dataclass(frozen=True)
class _FieldBufferCreationNamespace(_FieldArrayConstructionNamespace):
    allocator: next_allocators.FieldBufferAllocatorProtocol
    device: core_defs.Device | None = None
    aligned_index: Sequence[common.NamedIndex] | None = None

    @functools.cached_property
    def device_id(self) -> int:
        return self.device.device_id if self.device is not None else 0

    def empty(self, domain: common.Domain, *, dtype: core_defs.DType) -> core_defs.NDArrayObject:
        return self.allocator.__gt_allocate__(
            domain=domain,
            dtype=dtype,
            aligned_index=self.aligned_index,
            device_id=self.device_id,
        ).ndarray

    def zeros(self, domain: common.Domain, *, dtype: core_defs.DType) -> core_defs.NDArrayObject:
        return self.full(domain, fill_value=0, dtype=dtype)

    def ones(self, domain: common.Domain, *, dtype: core_defs.DType) -> core_defs.NDArrayObject:
        return self.full(domain, fill_value=1, dtype=dtype)

    def full(
        self,
        domain: common.Domain,
        fill_value: core_defs.Scalar,
        *,
        dtype: core_defs.DType,
    ) -> core_defs.NDArrayObject:
        arr = self.empty(domain, dtype=dtype)
        arr[...] = fill_value  # type: ignore[index] # `NDArrayObject` typing is not complete
        return arr

    def asarray(
        self,
        domain: common.Domain,
        data: core_defs.NDArrayObject,
        *,
        dtype: core_defs.DType,
    ) -> core_defs.NDArrayObject:
        arr = self.empty(domain, dtype=dtype)

        arr[...] = core_ndarray_utils.array_namespace(arr).asarray(data)  # type: ignore[index] # `NDArrayObject` typing is not complete
        return arr


@eve.utils.optional_lru_cache
def _field_constructor(
    allocator: Allocator | None,
    aligned_index: Sequence[common.NamedIndex] | None = None,
    device: core_defs.Device | None = None,
) -> FieldConstructor:
    return FieldConstructor(allocator, aligned_index=aligned_index, device=device)


@eve.utils.with_fluid_partial
def empty(
    domain: common.DomainLike,
    dtype: core_defs.DTypeLike = DEFAULT_DTYPE,
    *,
    aligned_index: Sequence[common.NamedIndex] | None = None,
    allocator: Allocator | None = None,
    device: core_defs.Device | None = None,
) -> nd_array_field.NdArrayField:
    """Create a `Field` of uninitialized (undefined) values using the given (or device-default) allocator.

    This function supports partial binding of arguments, see :class:`eve.utils.partial` for details.

    Arguments:
        domain: Definition of the domain of the field (which fix the shape of the allocated field buffer).
            See :class:`gt4py.next.common.Domain` for details.
        dtype: Definition of the data type of the field. Defaults to `float64`.

    Keyword Arguments:
        aligned_index: Index in the definition domain which should be used as reference
            point for memory alignment computations. It can be set to the most common origin
            of computations in this domain (if known) for performance reasons.
        allocator: An array namespace (e.g. ``numpy``, ``cupy``) or an allocator /
            allocator factory (e.g. backend) used for memory buffer allocation.
            If neither `allocator` nor `device` is given, a default CPU allocator is used.
            If both are given, `allocator` takes precedence over the default device allocator.
        device: The device (CPU, type of accelerator) to optimize the memory layout for.
            If neither `allocator` nor `device` is given, a default CPU device is used.
            When only `device` is given, the default allocator for that device type is used.

    Returns:
        A field, backed by a buffer with memory layout as specified by allocator and alignment requirements.

    Raises:
        ValueError
            If illegal or inconsistent arguments are specified.

    Examples:
        Initialize a field in one dimension with a backend and a range domain:

        >>> from gt4py import next as gtx
        >>> IDim = gtx.Dimension("I")
        >>> a = gtx.empty({IDim: range(3, 10)}, allocator=gtx.itir_python)
        >>> a.shape
        (7,)

        Initialize a field in one dimension from an array namespace:

        >>> import numpy as np
        >>> from gt4py import next as gtx
        >>> IDim = gtx.Dimension("I")
        >>> a = gtx.empty({IDim: range(3, 10)}, allocator=np)
        >>> a.shape
        (7,)

        Initialize with a device and an integer domain. It works like a shape with named dimensions:

        >>> from gt4py._core import definitions as core_defs
        >>> JDim = gtx.Dimension("J")
        >>> b = gtx.empty(
        ...     {IDim: 3, JDim: 3}, int, device=core_defs.Device(core_defs.DeviceType.CPU, 0)
        ... )
        >>> b.shape
        (3, 3)
    """
    return _field_constructor(allocator, aligned_index=aligned_index, device=device).empty(
        domain, dtype=dtype
    )


@eve.utils.with_fluid_partial
def zeros(
    domain: common.DomainLike,
    dtype: core_defs.DTypeLike = DEFAULT_DTYPE,
    *,
    aligned_index: Sequence[common.NamedIndex] | None = None,
    allocator: Allocator | None = None,
    device: core_defs.Device | None = None,
) -> nd_array_field.NdArrayField:
    """Create a Field containing all zeros using the given (or device-default) allocator.

    This function supports partial binding of arguments, see :class:`eve.utils.partial` for details.
    See :func:`empty` for further details about the meaning of the arguments.

    Examples:
        >>> from gt4py import next as gtx
        >>> IDim = gtx.Dimension("I")
        >>> gtx.zeros({IDim: range(3, 10)}, allocator=gtx.itir_python).ndarray
        array([0., 0., 0., 0., 0., 0., 0.])
    """
    return _field_constructor(allocator, aligned_index=aligned_index, device=device).zeros(
        domain, dtype=dtype
    )


@eve.utils.with_fluid_partial
def ones(
    domain: common.DomainLike,
    dtype: core_defs.DTypeLike = DEFAULT_DTYPE,
    *,
    aligned_index: Sequence[common.NamedIndex] | None = None,
    allocator: Allocator | None = None,
    device: core_defs.Device | None = None,
) -> nd_array_field.NdArrayField:
    """Create a Field containing all ones using the given (or device-default) allocator.

    This function supports partial binding of arguments, see :class:`eve.utils.partial` for details.
    See :func:`empty` for further details about the meaning of the arguments.

    Examples:
        >>> from gt4py import next as gtx
        >>> IDim = gtx.Dimension("I")
        >>> gtx.ones({IDim: range(3, 10)}, allocator=gtx.itir_python).ndarray
        array([1., 1., 1., 1., 1., 1., 1.])
    """
    return _field_constructor(allocator, aligned_index=aligned_index, device=device).ones(
        domain, dtype=dtype
    )


@eve.utils.with_fluid_partial
def full(
    domain: common.DomainLike,
    fill_value: core_defs.Scalar,
    dtype: core_defs.DTypeLike | None = None,
    *,
    aligned_index: Sequence[common.NamedIndex] | None = None,
    allocator: Allocator | None = None,
    device: core_defs.Device | None = None,
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
        >>> IDim = gtx.Dimension("I")
        >>> gtx.full({IDim: 3}, 5, allocator=gtx.itir_python).ndarray
        array([5, 5, 5])
    """
    return _field_constructor(allocator, aligned_index=aligned_index, device=device).full(
        domain, fill_value=fill_value, dtype=dtype
    )


@eve.utils.with_fluid_partial
def as_field(
    domain: common.DomainLike | Sequence[common.Dimension],
    data: core_defs.NDArrayObject,
    dtype: core_defs.DTypeLike | None = None,
    *,
    origin: Mapping[common.Dimension, int] | None = None,
    aligned_index: Sequence[common.NamedIndex] | None = None,
    allocator: Allocator | None = None,
    device: core_defs.Device | None = None,
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
        allocator: An array namespace (e.g. ``numpy``) or an allocator / allocator factory (e.g. backend).
            If not given, defaults to the device of `data` (if available) or the default CPU allocator.
        device: The device to optimize memory layout for. If not given, defaults to the device
            of `data` (if available) or the default CPU device.

    Note: we do not support a 'copy' argument as we want to avoid creating fields aliasing other data/fields.

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
    if allocator is None and device is None and xtyping.supports_dlpack(data):
        # allocate for the device of the input data if no explicit allocator or device is given
        device = core_defs.Device(*data.__dlpack_device__())
    return _field_constructor(allocator, aligned_index=aligned_index, device=device).as_field(
        domain=domain, data=data, dtype=dtype, origin=origin
    )


@eve.utils.with_fluid_partial
def as_connectivity(
    domain: common.DomainLike | Sequence[common.Dimension],
    codomain: common.Dimension,
    data: core_defs.NDArrayObject,
    dtype: core_defs.DTypeLike | None = None,
    *,
    origin: Mapping[common.Dimension, int] | None = None,
    aligned_index: Sequence[common.NamedIndex] | None = None,
    allocator: Allocator | None = None,
    device: core_defs.Device | None = None,
    skip_value: core_defs.IntegralScalar | eve.NothingType | None = eve.NOTHING,
) -> common.Connectivity:
    """
    Construct a `Connectivity` from the given domain, codomain, and data.

    Arguments:
        domain: The domain of the connectivity. It can be either a `common.DomainLike` object or a
            sequence of `common.Dimension` objects.
        codomain: The codomain dimension of the connectivity.
        data: The data used to construct the connectivity field.
        dtype: The data type of the connectivity. If not provided, it will be inferred from the data.

    Keyword Arguments:
        origin: Only allowed if `domain` is a sequence of dimensions. The indicated index in `data`
            will be the zero point of the resulting connectivity.
        aligned_index: Index in the definition domain which should be used as reference
            point for memory alignment computations.
        allocator: An array namespace or an allocator / allocator factory (e.g. backend).
            If not provided, defaults to the device of `data` or the default CPU allocator.
        device: The device on which the connectivity will be allocated. If not provided,
            defaults to the device of `data` or the default CPU device.
        skip_value: The value that signals missing entries in the neighbor table. Defaults to the default
            skip value if it is found in data, otherwise to `None` (= no skip value).

    Returns:
        The constructed connectivity field.

    Raises:
        ValueError: If the domain or codomain is invalid, or if the shape of the data does not match the domain shape.

    Examples:
        >>> import numpy as np
        >>> from gt4py import next as gtx
        >>> Vertex = gtx.Dimension("Vertex")
        >>> Edge = gtx.Dimension("Edge")
        >>> V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
        >>> data = np.array([[0, 1], [1, 2], [2, 0]])
        >>> conn = gtx.as_connectivity([Vertex, V2EDim], Edge, data)
        >>> conn.ndarray
        array([[0, 1],
               [1, 2],
               [2, 0]])
        >>> conn.domain
        Domain(dims=(Dimension(value='Vertex', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='V2E', kind=<DimensionKind.LOCAL: 'local'>)), ranges=(UnitRange(0, 3), UnitRange(0, 2)))
    """
    if skip_value is eve.NOTHING:
        skip_value = (
            common._DEFAULT_SKIP_VALUE if (data == common._DEFAULT_SKIP_VALUE).any() else None
        )

    field = as_field(
        domain=domain,
        data=data,
        dtype=dtype,
        origin=origin,
        aligned_index=aligned_index,
        allocator=allocator,
        device=device,
    )

    assert skip_value is not eve.NOTHING  # for mypy
    connectivity_field = common._connectivity(
        field.ndarray,
        codomain=codomain,
        domain=field.domain,
        skip_value=cast(core_defs.IntegralScalar | None, skip_value),  # see assert above
    )
    assert isinstance(connectivity_field, nd_array_field.NdArrayConnectivityField)

    return connectivity_field
