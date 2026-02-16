# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Generic, Optional, Protocol, TypeAlias, TypeGuard, TypeVar, cast

import numpy as np

import gt4py.eve as eve
import gt4py.eve.extended_typing as xtyping
import gt4py.next.allocators as next_allocators
import gt4py.next.common as common
import gt4py.next.embedded.nd_array_field as nd_array_field
import gt4py.storage.cartesian.utils as storage_utils
from gt4py._core import definitions as core_defs


# TODO for Monday:
# the current proposal breaks one of our other ideas, we wanted to store the allocator on the field
# to be able to propagate the allocation strategy in array operations, but that requires the aligned_index
# to be stored in absolute (NamedIndex) format, because we want to use it to align domain points, not buffer points.
# But then we can't compute the relative (int) aligned index from just the shape passed to e.g. empty.
# Of course we can still save the (FieldAllocator, aligned_index) tuple on the field not the resolved one.


# Notes (3 concepts): we need
# - translation of dtype/device to target namespace
# - absolute (domain) to relative (shape), includes aligned index
# - backing array api OR FieldBufferAllocator


# TODO: move to core_defs?
class ArrayAPICreationNamespace(Protocol):
    """
    Subset of the Array API standard namespace with functions relevant for construction.
    """

    def empty(
        self, shape: tuple[int, ...], *, dtype: Any | None = None, device: Any | None = None
    ) -> core_defs.NDArrayObject: ...

    def zeros(
        self, shape: tuple[int, ...], *, dtype: Any | None = None, device: Any | None = None
    ) -> core_defs.NDArrayObject: ...

    def ones(
        self, shape: tuple[int, ...], *, dtype: Any | None = None, device: Any | None = None
    ) -> core_defs.NDArrayObject: ...

    def full(
        self,
        shape: tuple[int, ...],
        fill_value: Any,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
    ) -> core_defs.NDArrayObject: ...

    def asarray(
        self,
        obj: Any,
        /,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
        copy: bool | None = None,
    ) -> core_defs.NDArrayObject: ...

    @property
    def bool(self) -> Any: ...

    @property
    def int8(self) -> Any: ...

    @property
    def int16(self) -> Any: ...

    @property
    def int32(self) -> Any: ...

    @property
    def int64(self) -> Any: ...

    @property
    def uint8(self) -> Any: ...

    @property
    def uint16(self) -> Any: ...

    @property
    def uint32(self) -> Any: ...

    @property
    def uint64(self) -> Any: ...

    @property
    def float32(self) -> Any: ...

    @property
    def float64(self) -> Any: ...


def is_array_api_creation_namespace(obj: object) -> TypeGuard[ArrayAPICreationNamespace]:
    """Check whether `obj` is an array namespace.

    An array namespace is any module that follows the Array API standard
    (https://data-apis.org/array-api/latest/).
    """

    # TODO extend
    return (
        hasattr(obj, "empty")
        and hasattr(obj, "zeros")
        and hasattr(obj, "ones")
        and hasattr(obj, "full")
        and hasattr(obj, "asarray")
    )


_ANS = TypeVar("_ANS", bound=ArrayAPICreationNamespace)


@dataclasses.dataclass(frozen=True)
class ArrayNamespaceWrapper(Generic[_ANS]):
    array_ns: _ANS

    def _to_array_ns_dtype(self, dtype: core_defs.DTypeLike | None) -> Any:
        if dtype is None:
            return None
        return getattr(self.array_ns, core_defs.dtype_to_name(dtype.scalar_type))

    def _to_array_ns_device(self, device: core_defs.Device | None) -> Any:
        if device is None:
            return None
        return get_device_translator(self.array_ns)(device)

    def empty(
        self,
        shape: tuple[int, ...],
        *,
        dtype: core_defs.DType | None = None,
        device: core_defs.Device | None = None,
    ) -> core_defs.NDArrayObject:
        return self.array_ns.empty(
            shape, dtype=self._to_array_ns_dtype(dtype), device=self._to_array_ns_device(device)
        )

    def zeros(
        self,
        shape: tuple[int, ...],
        *,
        dtype: core_defs.DType | None = None,
        device: core_defs.Device | None = None,
    ) -> core_defs.NDArrayObject:
        return self.array_ns.zeros(
            shape, dtype=self._to_array_ns_dtype(dtype), device=self._to_array_ns_device(device)
        )

    def ones(
        self,
        shape: tuple[int, ...],
        *,
        dtype: core_defs.DType | None = None,
        device: core_defs.Device | None = None,
    ) -> core_defs.NDArrayObject:
        return self.array_ns.ones(
            shape, dtype=self._to_array_ns_dtype(dtype), device=self._to_array_ns_device(device)
        )

    def full(
        self,
        shape: tuple[int, ...],
        fill_value: core_defs.Scalar,
        *,
        dtype: core_defs.DType | None = None,
        device: core_defs.Device | None = None,
    ) -> core_defs.NDArrayObject:
        return self.array_ns.full(
            shape,
            fill_value,
            dtype=self._to_array_ns_dtype(dtype),
            device=self._to_array_ns_device(device),
        )

    def asarray(
        self,
        obj: Any,
        *,
        dtype: core_defs.DType | None = None,
        device: core_defs.Device | None = None,
        copy: bool | None = None,
    ) -> core_defs.NDArrayObject:
        return self.array_ns.asarray(
            obj,
            dtype=self._to_array_ns_dtype(dtype),
            device=self._to_array_ns_device(device),
            copy=copy,
        )

    @property
    def bool(self) -> _ANS.bool:
        return self.array_ns.bool

    # etc


def _get_array_namespace(arr) -> ArrayAPICreationNamespace:
    # TODO use array api compatibility layer?
    if isinstance(arr, np.ndarray):
        return np
    elif core_defs.cp is not None and isinstance(arr, core_defs.cp.ndarray):
        return core_defs.cp
    else:
        raise ValueError(f"Could not determine array namespace for array of type {type(arr)}.")


@dataclasses.dataclass(frozen=True)
class FieldBufferAllocatorWrapper:
    allocator: next_allocators.FieldBufferAllocatorProtocol
    dims: Sequence[common.Dimension]
    aligned_index: (
        Sequence[int] | None
    )  # TODO the translation from Sequence[NamedIndex] and Domain needs to be done outside!

    def empty(
        self,
        shape: tuple[int, ...],
        *,
        dtype: core_defs.DTypeLike | None = None,
        device: core_defs.Device | None = None,
    ) -> core_defs.NDArrayObject:
        assert dtype is not None
        return self.allocator.__gt_allocate__(
            dims=self.dims,
            shape=shape,
            dtype=core_defs.dtype(dtype),
            aligned_index=self.aligned_index,
            device_id=device.device_id if device is not None else 0,
        ).ndarray

    def zeros(
        self,
        shape: tuple[int, ...],
        *,
        dtype: core_defs.DTypeLike | None = None,
        device: core_defs.Device | None = None,
    ) -> core_defs.NDArrayObject:
        assert dtype is not None
        arr = self.empty(shape, dtype=dtype, device=device)
        arr[...] = 0
        return arr

    def ones(
        self,
        shape: tuple[int, ...],
        *,
        dtype: core_defs.DTypeLike | None = None,
        device: core_defs.Device | None = None,
    ) -> core_defs.NDArrayObject:
        assert dtype is not None
        arr = self.empty(shape, dtype=dtype, device=device)
        arr[...] = 1
        return arr

    def full(
        self,
        shape: tuple[int, ...],
        fill_value: core_defs.Scalar,
        *,
        dtype: core_defs.DTypeLike | None = None,
        device: core_defs.Device | None = None,
    ) -> core_defs.NDArrayObject:
        dtype if dtype is not None else core_defs.dtype(type(fill_value))
        arr = self.empty(shape, dtype=dtype, device=device)
        arr[...] = fill_value
        return arr

    def asarray(
        self,
        data: core_defs.NDArrayObject,
        *,
        dtype: core_defs.DTypeLike | None = None,
        device: core_defs.Device | None = None,
        copy: bool | None = None,
    ) -> core_defs.NDArrayObject:
        assert copy is None
        if dtype is None:
            dtype = storage_utils.asarray(data).dtype

        arr = self.empty(data.shape, dtype=dtype, device=device)
        array_ns = _get_array_namespace(arr)  # TODO add compatibility function
        arr[...] = array_ns.asarray(data)
        return arr


# Type to be used by the end-user
FieldAllocator: TypeAlias = ArrayAPICreationNamespace | next_allocators.FieldBufferAllocationUtil


def _numpy_device_translator(device: core_defs.Device | None) -> Any:
    if device is None:
        return None
    if device.device_type == core_defs.DeviceType.CPU:
        return None  # or Literal['cpu']
    raise ValueError(f"NumPy does not support device type {device.device_type}.")


# TODO move to the correct place
_registry = {np: _numpy_device_translator}


def get_device_translator(array_ns: ArrayAPICreationNamespace) -> Callable[[core_defs.Device], Any]:
    return _registry.get(array_ns)


# class OurArrayApiCreationNamespace(ArrayAPICreationNamespace, Protocol): ...


# class ArrayAPIWrapper(OurArrayApiCreationNamespace):
#     # TODO the concrete one
#     ...


# class FieldBufferAllocatorWrapper(OurArrayApiCreationNamespace):
#     # TODO the concrete one that wraps a FieldBufferAllocationUtil and implements the functions by calling the allocator's allocation functions and then converting the resulting buffer to an array using the array API of choice
#     ...


def _aligned_index(
    domain: common.Domain, aligned_index: Sequence[common.NamedIndex] | None
) -> Sequence[int] | None:
    """Translates the absolute aligned index to a relative aligned index in the array buffer."""
    if aligned_index is not None:
        raise NotImplementedError("Aligned index is not yet implemented.")
    return None


def _as_array_api_construction_namespace(
    domain: common.Domain,
    allocator: FieldAllocator | None,
    device: core_defs.Device | None,
    aligned_index: Sequence[common.NamedIndex] | None = None,
) -> ArrayAPICreationNamespace:  # TODO actually our concrete one that takes our dtype and device
    return _as_array_api_construction_namespace_lower(
        domain.dims,
        allocator,
        device.device_type if device is not None else None,
        _aligned_index(domain, aligned_index),
    )


# @functools.cache # TODO
def _as_array_api_construction_namespace_lower(
    dims: Sequence[common.Dimension],
    allocator: FieldAllocator | None,
    device_type: core_defs.DeviceType | None,
    aligned_index: Sequence[int] | None = None,
) -> ArrayAPICreationNamespace:  # TODO actually our concrete one that takes our dtype and device
    if allocator is None:
        if device_type is None:
            device_type = core_defs.DeviceType.CPU  # fallback to CPU allocation # TODO discuss, we could also fallback to Numpy allocation
        allocator = next_allocators.device_allocators[device_type]

    assert allocator is not None
    if next_allocators.is_field_allocation_tool(allocator):
        allocator = next_allocators.get_allocator(
            allocator, strict=True
        )  # TODO this function could be inlined here?
        if device_type is not None and allocator.__gt_device_type__ != device_type:
            raise ValueError(
                f"Allocator {allocator} is for device type {allocator.__gt_device_type__}, but device type {device_type} was specified."
            )
        return FieldBufferAllocatorWrapper(
            allocator=allocator, dims=dims, aligned_index=aligned_index
        )
    elif is_array_api_creation_namespace(allocator):
        # TODO can/should we check if the namespace matches device_type if specified?
        return ArrayNamespaceWrapper(array_ns=allocator)
    else:
        # static analysis guarantees this already, but we want a user-facing error at runtime
        raise ValueError(f"Invalid field allocator: {allocator}.")


@eve.utils.with_fluid_partial
def empty(
    domain: common.DomainLike,
    dtype: core_defs.DTypeLike = core_defs.Float64DType(()),  # noqa: B008 [function-call-in-default-argument]
    *,
    aligned_index: Sequence[common.NamedIndex] | None = None,
    allocator: FieldAllocator = None,
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
        >>> IDim = gtx.Dimension("I")
        >>> a = gtx.empty({IDim: range(3, 10)}, allocator=gtx.itir_python)
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
    domain = common.domain(domain)
    array_api_constructor_namespace = _as_array_api_construction_namespace(
        domain=domain, allocator=allocator, device=device, aligned_index=aligned_index
    )

    # TODO double-check where we should resolve dtypelike->dtype
    arr = array_api_constructor_namespace.empty(domain.shape, dtype=dtype, device=device)
    return common._field(arr, domain=domain)


@eve.utils.with_fluid_partial
def zeros(
    domain: common.DomainLike,
    dtype: core_defs.DTypeLike = core_defs.Float64DType(()),  # noqa: B008 [function-call-in-default-argument]
    *,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    allocator: FieldAllocator | None = None,
    device: Optional[core_defs.Device] = None,
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
    domain = common.domain(domain)
    array_api_constructor_namespace = _as_array_api_construction_namespace(
        domain=domain, allocator=allocator, device=device, aligned_index=aligned_index
    )

    arr = array_api_constructor_namespace.zeros(domain.shape, dtype=dtype, device=device)
    return common._field(arr, domain=domain)


@eve.utils.with_fluid_partial
def ones(
    domain: common.DomainLike,
    dtype: core_defs.DTypeLike = core_defs.Float64DType(()),  # noqa: B008 [function-call-in-default-argument]
    *,
    aligned_index: Sequence[common.NamedIndex] | None = None,
    allocator: FieldAllocator | None = None,
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
    domain = common.domain(domain)
    array_api_constructor_namespace = _as_array_api_construction_namespace(
        domain=domain, allocator=allocator, device=device, aligned_index=aligned_index
    )

    arr = array_api_constructor_namespace.ones(domain.shape, dtype=dtype, device=device)
    return common._field(arr, domain=domain)


@eve.utils.with_fluid_partial
def full(
    domain: common.DomainLike,
    fill_value: core_defs.Scalar,
    dtype: core_defs.DTypeLike | None = None,
    *,
    aligned_index: Sequence[common.NamedIndex] | None = None,
    allocator: FieldAllocator | None = None,
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
    domain = common.domain(domain)
    array_api_constructor_namespace = _as_array_api_construction_namespace(
        domain=domain, allocator=allocator, device=device, aligned_index=aligned_index
    )

    arr = array_api_constructor_namespace.full(
        domain.shape, fill_value=fill_value, dtype=dtype, device=device
    )
    return common._field(arr, domain=domain)


@eve.utils.with_fluid_partial
def as_field(
    domain: common.DomainLike | Sequence[common.Dimension],
    data: core_defs.NDArrayObject,
    dtype: core_defs.DTypeLike | None = None,
    *,
    origin: Mapping[common.Dimension, int] | None = None,
    aligned_index: Sequence[common.NamedIndex] | None = None,
    allocator: FieldAllocator | None = None,
    device: core_defs.Device | None = None,
    copy: bool | None = None,
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
    if copy is not None:
        raise NotImplementedError("The 'copy' argument is not yet implemented.")

    dtype = core_defs.dtype(dtype) if dtype is not None else None
    assert dtype is None or dtype.tensor_shape == ()  # TODO in the other cases as well?

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
            raise ValueError(f"Cannot specify origin for a concrete domain {domain}")
        actual_domain = common.domain(cast(common.DomainLike, domain))

    # TODO(egparedes): allow zero-copy construction (no reallocation) if buffer has
    #   already the correct layout and device.
    shape = storage_utils.asarray(
        data
    ).shape  # TODO this is a quite expensive throw-away operation...
    if shape != actual_domain.shape:
        raise ValueError(f"Cannot construct 'Field' from array of shape '{shape}'.")

    if (allocator is None) and (device is None) and xtyping.supports_dlpack(data):
        device = core_defs.Device(*data.__dlpack_device__())

    arr = _as_array_api_construction_namespace(
        actual_domain, allocator, device, aligned_index=aligned_index
    ).asarray(data, dtype=dtype, device=device, copy=copy)
    return common._field(arr, domain=actual_domain)


@eve.utils.with_fluid_partial
def as_connectivity(
    domain: common.DomainLike | Sequence[common.Dimension],
    codomain: common.Dimension,
    data: core_defs.NDArrayObject,
    dtype: core_defs.DType | None = None,
    *,
    # TODO what about origin?
    allocator: FieldAllocator | None = None,
    device: core_defs.Device | None = None,
    skip_value: core_defs.IntegralScalar | eve.NothingType | None = eve.NOTHING,
    # TODO: copy=False
) -> common.Connectivity:
    """
    Construct a `Connectivity` from the given domain, codomain, and data.

    Arguments:
        domain: The domain of the connectivity. It can be either a `common.DomainLike` object or a
            sequence of `common.Dimension` objects.
        codomain: The codomain dimension of the connectivity.
        data: The data used to construct the connectivity field.
        dtype: The data type of the connectivity. If not provided, it will be inferred from the data.
        allocator: The allocator used to allocate the buffer for the connectivity. If not provided,
            a default allocator will be used.
        device: The device on which the connectivity will be allocated. If not provided, the default
            device will be used.
        skip_value: The value that signals missing entries in the neighbor table. Defaults to the default
            skip value if it is found in data, otherwise to `None` (= no skip value).

    Returns:
        The constructed connectivity field.

    Raises:
        ValueError: If the domain or codomain is invalid, or if the shape of the data does not match the domain shape.
    """
    if skip_value is eve.NOTHING:
        skip_value = (
            common._DEFAULT_SKIP_VALUE if (data == common._DEFAULT_SKIP_VALUE).any() else None
        )

    field = as_field(
        domain=domain,
        data=data,
        dtype=dtype,
        allocator=allocator,
        device=device,
    )

    connectivity_field = common._connectivity(
        field.ndarray, codomain=codomain, domain=field.domain, skip_value=skip_value
    )
    assert isinstance(connectivity_field, nd_array_field.NdArrayConnectivityField)

    return connectivity_field
