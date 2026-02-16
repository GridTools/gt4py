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
from typing import Any, Generic, Optional, TypeAlias, TypeVar, cast

import gt4py.eve as eve
import gt4py.next.allocators as next_allocators
import gt4py.next.common as common
import gt4py.next.embedded.nd_array_field as nd_array_field
import gt4py.storage.cartesian.utils as storage_utils
from gt4py._core import definitions as core_defs, ndarray_utils as core_ndarray_utils
from gt4py.eve import extended_typing as xtyping


# TODO translate into a description somewhere
# Notes (3 concepts): we need
# - translation of dtype/device to target namespace
# - absolute (domain) to relative (shape), includes aligned index
# - backing array api OR FieldBufferAllocator


# TODO Introduce a protocol and keep the concrete implementation for NDArray-based allocation (both array api and fieldbufferallocator) in a separate class?
# TODO maybe FieldAllocationNamespace can evolve to FieldNamespace (convering also all Field operations?)

# Type to be used by the end-user
FieldAllocator: TypeAlias = (
    core_ndarray_utils.ArrayCreationNamespace | next_allocators.FieldBufferAllocationUtil
)

# TODO make all other types private? check usage in ICON4Py


class FieldAllocationNamespace(abc.ABC):
    # TODO: expand to a full description
    """
    - emtpy, as_field, etc are public facing, they do the work that is shared between all subclasses:
      - conversion of dtype-like and domain-like
      - array to field
    - _empty, _as_field are to be implented by the subclasses:
      - note: we pass domain explicitly because it's needed together with aligned_index to compute the alignment relative to the buffer
    """

    def empty(
        self,
        domain: common.DomainLike,
        *,
        dtype: core_defs.DTypeLike | None = None,
    ) -> nd_array_field.NdArrayField:
        domain = common.domain(domain)
        dtype = core_defs.dtype(dtype)
        assert dtype is not None  # TODO check where to put the default
        res = common._field(self._empty(domain, dtype=dtype), domain=domain)
        assert isinstance(res, nd_array_field.NdArrayField)  # for typing
        return res

    @abc.abstractmethod
    def _empty(
        self, domain: common.Domain, *, dtype: core_defs.DType
    ) -> core_defs.NDArrayObject: ...

    def zeros(
        self,
        domain: common.DomainLike,
        dtype: core_defs.DTypeLike | None = None,
    ) -> nd_array_field.NdArrayField:
        domain = common.domain(domain)
        dtype = core_defs.dtype(dtype)
        assert dtype is not None
        res = common._field(self._zeros(domain, dtype=dtype), domain=domain)
        assert isinstance(res, nd_array_field.NdArrayField)  # for typing
        return res

    @abc.abstractmethod
    def _zeros(
        self, domain: common.Domain, *, dtype: core_defs.DType
    ) -> core_defs.NDArrayObject: ...

    def ones(
        self,
        domain: common.DomainLike,
        dtype: core_defs.DTypeLike | None = None,
    ) -> nd_array_field.NdArrayField:
        domain = common.domain(domain)
        dtype = core_defs.dtype(dtype)
        assert dtype is not None
        res = common._field(self._ones(domain, dtype=dtype), domain=domain)
        assert isinstance(res, nd_array_field.NdArrayField)  # for typing
        return res

    @abc.abstractmethod
    def _ones(
        self, domain: common.Domain, *, dtype: core_defs.DType
    ) -> core_defs.NDArrayObject: ...

    def full(
        self,
        domain: common.DomainLike,
        fill_value: core_defs.Scalar,
        dtype: core_defs.DTypeLike | None = None,
    ) -> nd_array_field.NdArrayField:
        domain = common.domain(domain)
        dtype = core_defs.dtype(dtype) if dtype is not None else core_defs.dtype(type(fill_value))
        assert dtype is not None
        res = common._field(self._full(domain, fill_value=fill_value, dtype=dtype), domain=domain)
        assert isinstance(res, nd_array_field.NdArrayField)  # for typing
        return res

    @abc.abstractmethod
    def _full(
        self,
        domain: common.Domain,
        fill_value: core_defs.Scalar,
        *,
        dtype: core_defs.DType,
    ) -> core_defs.NDArrayObject: ...

    def as_field(
        self,
        domain: common.DomainLike | Sequence[common.Dimension],
        data: core_defs.NDArrayObject,  # TODO rename to `obj`? # TODO what do we support as input?
        *,
        dtype: core_defs.DTypeLike | None = None,
        origin: Mapping[common.Dimension, int] | None = None,
        copy: bool | None = None,
    ) -> nd_array_field.NdArrayField:
        if copy is not None:
            raise NotImplementedError("The 'copy' argument is not yet implemented.")

        dtype = core_defs.dtype(dtype) if dtype is not None else None
        assert dtype is None or dtype.tensor_shape == ()  # TODO in the other cases as well?

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

        shape = storage_utils.asarray(
            data
        ).shape  # TODO this is a quite expensive throw-away operation...
        if shape != actual_domain.shape:
            raise ValueError(f"Cannot construct 'Field' from array of shape '{shape}'.")

        if dtype is None:  # TODO does this make sense?
            dtype = core_defs.dtype(storage_utils.asarray(data).dtype)

        arr = self._asarray(actual_domain, data, dtype=dtype, copy=copy)
        res = common._field(
            arr,
            domain=actual_domain,
        )
        assert isinstance(res, nd_array_field.NdArrayField)  # for typing
        return res

    @abc.abstractmethod
    def _asarray(
        self,
        domain: common.Domain,
        data: core_defs.NDArrayObject,
        *,
        dtype: core_defs.DType,
        copy: bool | None,
    ) -> core_defs.NDArrayObject: ...


_ANS = TypeVar("_ANS", bound=core_ndarray_utils.ArrayCreationNamespace)


@dataclasses.dataclass(frozen=True)
class ArrayNamespaceWrapper(FieldAllocationNamespace, Generic[_ANS]):
    array_ns: _ANS
    _device: dataclasses.InitVar[core_defs.Device | None] = None

    # device in the format expected by the array namespace
    device: Any = dataclasses.field(init=False)

    def __post_init__(self, _device: core_defs.Device | None) -> None:
        device = (
            core_ndarray_utils.get_device_translator(self.array_ns)(_device)
            if _device is not None
            else None
        )
        object.__setattr__(self, "device", device)

    def _to_array_ns_dtype(self, dtype: core_defs.DType | None) -> Any:
        if dtype is None:
            return None
        return getattr(self.array_ns, core_defs.dtype_to_name[dtype.scalar_type])

    def _empty(
        self,
        domain: common.Domain,
        *,
        dtype: core_defs.DType | None = None,
    ) -> core_defs.NDArrayObject:
        return self.array_ns.empty(
            domain.shape, dtype=self._to_array_ns_dtype(dtype), device=self.device
        )

    def _zeros(
        self,
        domain: common.Domain,
        *,
        dtype: core_defs.DType | None = None,
    ) -> core_defs.NDArrayObject:
        return self.array_ns.zeros(
            domain.shape, dtype=self._to_array_ns_dtype(dtype), device=self.device
        )

    def _ones(
        self,
        domain: common.Domain,
        *,
        dtype: core_defs.DType | None = None,
    ) -> core_defs.NDArrayObject:
        return self.array_ns.ones(
            domain.shape, dtype=self._to_array_ns_dtype(dtype), device=self.device
        )

    def _full(
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

    def _asarray(
        self,
        domain: common.Domain,
        data: core_defs.NDArrayObject,  # TODO rename to `obj`?
        *,
        dtype: core_defs.DType | None = None,
        copy: bool | None = None,
    ) -> core_defs.NDArrayObject:
        arr = self.array_ns.asarray(
            data, dtype=self._to_array_ns_dtype(dtype), device=self.device, copy=copy
        )
        assert domain.shape == arr.shape, (
            f"pre-condition of `_asarray` not met: {data.shape=} and {domain.shape} need to agree."
        )
        return arr


@dataclasses.dataclass(frozen=True)
class FieldBufferAllocatorWrapper(FieldAllocationNamespace):
    allocator: next_allocators.FieldBufferAllocatorProtocol
    device: core_defs.Device | None = None
    aligned_index: Sequence[common.NamedIndex] | None = None

    @functools.cached_property
    def device_id(self) -> int:
        return self.device.device_id if self.device is not None else 0

    def _empty(self, domain: common.Domain, *, dtype: core_defs.DType) -> core_defs.NDArrayObject:
        return self.allocator.__gt_allocate__(
            domain=domain,
            dtype=dtype,
            aligned_index=self.aligned_index,
            device_id=self.device_id,
        ).ndarray

    def _zeros(self, domain: common.Domain, *, dtype: core_defs.DType) -> core_defs.NDArrayObject:
        arr = self._empty(domain, dtype=dtype)
        arr[...] = 0  # type: ignore[index] # # `NDArrayObject` typing is not complete
        return arr

    def _ones(self, domain: common.Domain, *, dtype: core_defs.DType) -> core_defs.NDArrayObject:
        arr = self._empty(domain, dtype=dtype)
        arr[...] = 1  # type: ignore[index] # # `NDArrayObject` typing is not complete
        return arr

    def _full(
        self,
        domain: common.Domain,
        fill_value: core_defs.Scalar,
        *,
        dtype: core_defs.DType,
    ) -> core_defs.NDArrayObject:
        arr = self._empty(domain, dtype=dtype)
        arr[...] = fill_value  # type: ignore[index] # # `NDArrayObject` typing is not complete
        return arr

    def _asarray(
        self,
        domain: common.Domain,
        data: core_defs.NDArrayObject,
        *,
        dtype: core_defs.DType,
        copy: bool | None = None,
    ) -> core_defs.NDArrayObject:
        if copy is not None:
            # TODO(egparedes): allow zero-copy construction (no reallocation) if buffer has
            #   already the correct layout and device.
            raise NotImplementedError("The 'copy' argument is not yet supported.")
        assert dtype is not None
        arr = self._empty(domain, dtype=dtype)

        arr[...] = core_ndarray_utils.array_namespace(arr).asarray(data)  # type: ignore[index] # # `NDArrayObject` typing is not complete
        return arr


# TODO test all code paths
def _as_field_allocation_namespace(
    allocator: FieldAllocator | None,  # TODO should we allow None
    *,
    aligned_index: Sequence[common.NamedIndex] | None = None,
    device: core_defs.Device | None = None,
) -> FieldAllocationNamespace:
    if allocator is None:
        if device is None:
            allocator = next_allocators.device_allocators[core_defs.DeviceType.CPU]
        else:
            allocator = next_allocators.device_allocators[device.device_type]

    if core_ndarray_utils.is_array_api_creation_namespace(allocator):
        return ArrayNamespaceWrapper(array_ns=allocator, _device=device)
    elif next_allocators.is_field_allocation_tool(allocator):
        allocator = next_allocators.get_allocator(
            allocator, strict=True
        )  # TODO this function could be inlined here?
        assert allocator is not None  # for mypy
        if device is not None and allocator.__gt_device_type__ != device.device_type:
            raise ValueError(
                f"Allocator {allocator} is for device type {allocator.__gt_device_type__}, but device type {device.device_type} was specified."
            )
        return FieldBufferAllocatorWrapper(
            allocator=allocator,
            device=device,
            aligned_index=aligned_index,
        )
    else:
        raise ValueError(f"Invalid field allocator: {allocator}.")


@eve.utils.with_fluid_partial
def empty(
    domain: common.DomainLike,
    dtype: core_defs.DTypeLike = core_defs.Float64DType(()),  # noqa: B008 [function-call-in-default-argument]
    *,
    aligned_index: Sequence[common.NamedIndex] | None = None,
    allocator: FieldAllocator | None = None,
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

        Initialize a field in one dimension from an array namespace:

        >>> import numpy as
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
    field_array_allocation_api = _as_field_allocation_namespace(
        allocator, aligned_index=aligned_index, device=device
    )
    return field_array_allocation_api.empty(domain, dtype=dtype)


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
    field_array_allocation_api = _as_field_allocation_namespace(
        allocator, aligned_index=aligned_index, device=device
    )
    return field_array_allocation_api.zeros(domain, dtype=dtype)


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
    field_array_allocation_api = _as_field_allocation_namespace(
        allocator, aligned_index=aligned_index, device=device
    )
    return field_array_allocation_api.ones(domain, dtype=dtype)


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
    field_array_allocation_api = _as_field_allocation_namespace(
        allocator, aligned_index=aligned_index, device=device
    )
    return field_array_allocation_api.full(domain, fill_value=fill_value, dtype=dtype)


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
    if allocator is None and device is None and xtyping.supports_dlpack(data):
        # allocate for the device of the input data if no explicit allocator or device is given
        device = core_defs.Device(*data.__dlpack_device__())
    return _as_field_allocation_namespace(
        allocator, aligned_index=aligned_index, device=device
    ).as_field(domain=domain, data=data, dtype=dtype, origin=origin, copy=copy)


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

    assert skip_value is not eve.NOTHING  # for mypy
    connectivity_field = common._connectivity(
        field.ndarray,
        codomain=codomain,
        domain=field.domain,
        skip_value=cast(core_defs.IntegralScalar | None, skip_value),  # see assert above
    )
    assert isinstance(connectivity_field, nd_array_field.NdArrayConnectivityField)

    return connectivity_field
