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

import dataclasses
import functools
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import ClassVar

import numpy as np
from numpy import typing as npt

from gt4py._core import definitions as core_defs
from gt4py.eve.extended_typing import Any, Never, Optional, ParamSpec, TypeAlias, TypeVar
from gt4py.next import common
from gt4py.next.embedded import common as embedded_common, context as embedded_context
from gt4py.next.ffront import fbuiltins
from gt4py.next.iterator import embedded as itir_embedded


try:
    import cupy as cp
except ImportError:
    cp: Optional[ModuleType] = None  # type:ignore[no-redef]

try:
    from jax import numpy as jnp
except ImportError:
    jnp: Optional[ModuleType] = None  # type:ignore[no-redef]


def _make_builtin(
    builtin_name: str, array_builtin_name: str, reversed=False
) -> Callable[..., NdArrayField]:
    def _builtin_op(*fields: common.Field | core_defs.Scalar) -> NdArrayField:
        first = None
        for f in fields:
            if isinstance(f, NdArrayField):
                first = f
                break
        assert first is not None
        xp = first.__class__.array_ns
        op = getattr(xp, array_builtin_name)

        domain_intersection = embedded_common.intersect_domains(
            *[f.domain for f in fields if common.is_field(f)]
        )

        transformed: list[core_defs.NDArrayObject | core_defs.Scalar] = []
        for f in fields:
            if common.is_field(f):
                if f.domain == domain_intersection:
                    transformed.append(xp.asarray(f.ndarray))
                else:
                    f_broadcasted = (
                        _broadcast(f, domain_intersection.dims)
                        if f.domain.dims != domain_intersection.dims
                        else f
                    )
                    f_slices = _get_slices_from_domain_slice(
                        f_broadcasted.domain, domain_intersection
                    )
                    transformed.append(xp.asarray(f_broadcasted.ndarray[f_slices]))
            else:
                assert core_defs.is_scalar_type(f)
                transformed.append(f)
        if reversed:
            transformed = transformed[::-1]
        new_data = op(*transformed)
        return first.__class__.from_array(new_data, domain=domain_intersection)

    _builtin_op.__name__ = builtin_name
    return _builtin_op


_Value: TypeAlias = common.Field | core_defs.ScalarT
_P = ParamSpec("_P")
_R = TypeVar("_R", _Value, tuple[_Value, ...])


@dataclasses.dataclass(frozen=True)
class NdArrayField(
    common.MutableField[common.DimsT, core_defs.ScalarT], common.FieldBuiltinFuncRegistry
):
    """
    Shared field implementation for NumPy-like fields.

    Builtin function implementations are registered in a dictionary.
    Note: Currently, all concrete NdArray-implementations share
    the same implementation, dispatching is handled inside of the registered
    function via its namespace.
    """

    _domain: common.Domain
    _ndarray: core_defs.NDArrayObject

    array_ns: ClassVar[ModuleType]  # TODO(havogt) introduce a NDArrayNamespace protocol

    @property
    def domain(self) -> common.Domain:
        return self._domain

    @property
    def shape(self) -> tuple[int, ...]:
        return self._ndarray.shape

    @property
    def __gt_origin__(self) -> tuple[int, ...]:
        assert common.Domain.is_finite(self._domain)
        return tuple(-r.start for _, r in self._domain)

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        return self._ndarray

    def asnumpy(self) -> np.ndarray:
        if self.array_ns == cp:
            return cp.asnumpy(self._ndarray)
        else:
            return np.asarray(self._ndarray)

    def as_scalar(self) -> core_defs.ScalarT:
        if self.domain.ndim != 0:
            raise ValueError(
                "'as_scalar' is only valid on 0-dimensional 'Field's, got a {self.domain.ndim}-dimensional 'Field'."
            )
        return self.ndarray.item()

    @property
    def codomain(self) -> type[core_defs.ScalarT]:
        return self.dtype.scalar_type

    @property
    def dtype(self) -> core_defs.DType[core_defs.ScalarT]:
        return core_defs.dtype(self._ndarray.dtype.type)

    @classmethod
    def from_array(
        cls,
        data: (
            npt.ArrayLike | core_defs.NDArrayObject
        ),  # TODO: NDArrayObject should be part of ArrayLike
        /,
        *,
        domain: common.DomainLike,
        dtype: Optional[core_defs.DTypeLike] = None,
    ) -> NdArrayField:
        domain = common.domain(domain)
        xp = cls.array_ns

        xp_dtype = None if dtype is None else xp.dtype(core_defs.dtype(dtype).scalar_type)
        array = xp.asarray(data, dtype=xp_dtype)

        if dtype is not None:
            assert array.dtype.type == core_defs.dtype(dtype).scalar_type

        assert issubclass(array.dtype.type, core_defs.SCALAR_TYPES)

        assert all(isinstance(d, common.Dimension) for d in domain.dims), domain
        assert len(domain) == array.ndim
        assert all(s == 1 or len(r) == s for r, s in zip(domain.ranges, array.shape))

        return cls(domain, array)

    def remap(
        self: NdArrayField, connectivity: common.ConnectivityField | fbuiltins.FieldOffset
    ) -> NdArrayField:
        # For neighbor reductions, a FieldOffset is passed instead of an actual ConnectivityField
        if not common.is_connectivity_field(connectivity):
            assert isinstance(connectivity, fbuiltins.FieldOffset)
            connectivity = connectivity.as_connectivity_field()
        assert common.is_connectivity_field(connectivity)

        # Current implementation relies on skip_value == -1:
        # if we assume the indexed array has at least one element, we wrap around without out of bounds
        assert connectivity.skip_value is None or connectivity.skip_value == -1

        # Compute the new domain
        dim = connectivity.codomain
        dim_idx = self.domain.dim_index(dim)
        if dim_idx is None:
            raise ValueError(f"Incompatible index field, expected a field with dimension '{dim}'.")

        current_range: common.UnitRange = self.domain[dim_idx][1]
        new_ranges = connectivity.inverse_image(current_range)
        new_domain = self.domain.replace(dim_idx, *new_ranges)

        # perform contramap
        if not (connectivity.kind & common.ConnectivityKind.MODIFY_STRUCTURE):
            # shortcut for compact remap: don't change the array, only the domain
            new_buffer = self._ndarray
        else:
            # general case: first restrict the connectivity to the new domain
            restricted_connectivity_domain = common.Domain(*new_ranges)
            restricted_connectivity = (
                connectivity.restrict(restricted_connectivity_domain)
                if restricted_connectivity_domain != connectivity.domain
                else connectivity
            )
            assert common.is_connectivity_field(restricted_connectivity)

            # then compute the index array
            xp = self.array_ns
            new_idx_array = xp.asarray(restricted_connectivity.ndarray) - current_range.start
            # finally, take the new array
            new_buffer = xp.take(self._ndarray, new_idx_array, axis=dim_idx)

        return self.__class__.from_array(
            new_buffer,
            domain=new_domain,
            dtype=self.dtype,
        )

    __call__ = remap  # type: ignore[assignment]

    def restrict(self, index: common.AnyIndexSpec) -> common.Field:
        new_domain, buffer_slice = self._slice(index)
        new_buffer = self.ndarray[buffer_slice]
        new_buffer = self.__class__.array_ns.asarray(new_buffer)
        return self.__class__.from_array(new_buffer, domain=new_domain)

    __getitem__ = restrict

    def __setitem__(
        self: NdArrayField[common.DimsT, core_defs.ScalarT],
        index: common.AnyIndexSpec,
        value: common.Field | core_defs.NDArrayObject | core_defs.ScalarT,
    ) -> None:
        target_domain, target_slice = self._slice(index)

        if common.is_field(value):
            if not value.domain == target_domain:
                raise ValueError(
                    f"Incompatible 'Domain' in assignment. Source domain = '{value.domain}', target domain = '{target_domain}'."
                )
            value = value.ndarray

        assert hasattr(self.ndarray, "__setitem__")
        self._ndarray[target_slice] = value  # type: ignore[index] # np and cp allow index assignment, jax overrides

    __abs__ = _make_builtin("abs", "abs")

    __neg__ = _make_builtin("neg", "negative")

    __add__ = __radd__ = _make_builtin("add", "add")

    __pos__ = _make_builtin("pos", "positive")

    __sub__ = _make_builtin("sub", "subtract")
    __rsub__ = _make_builtin("sub", "subtract", reversed=True)

    __mul__ = __rmul__ = _make_builtin("mul", "multiply")

    __truediv__ = _make_builtin("div", "divide")
    __rtruediv__ = _make_builtin("div", "divide", reversed=True)

    __floordiv__ = _make_builtin("floordiv", "floor_divide")
    __rfloordiv__ = _make_builtin("floordiv", "floor_divide", reversed=True)

    __pow__ = _make_builtin("pow", "power")

    __mod__ = __rmod__ = _make_builtin("mod", "mod")

    __ne__ = _make_builtin("not_equal", "not_equal")  # type: ignore # mypy wants return `bool`

    __eq__ = _make_builtin("equal", "equal")  # type: ignore # mypy wants return `bool`

    __gt__ = _make_builtin("greater", "greater")

    __ge__ = _make_builtin("greater_equal", "greater_equal")

    __lt__ = _make_builtin("less", "less")

    __le__ = _make_builtin("less_equal", "less_equal")

    def __and__(self, other: common.Field | core_defs.ScalarT) -> NdArrayField:
        if self.dtype == core_defs.BoolDType():
            return _make_builtin("logical_and", "logical_and")(self, other)
        raise NotImplementedError("'__and__' not implemented for non-'bool' fields.")

    __rand__ = __and__

    def __or__(self, other: common.Field | core_defs.ScalarT) -> NdArrayField:
        if self.dtype == core_defs.BoolDType():
            return _make_builtin("logical_or", "logical_or")(self, other)
        raise NotImplementedError("'__or__' not implemented for non-'bool' fields.")

    __ror__ = __or__

    def __xor__(self, other: common.Field | core_defs.ScalarT) -> NdArrayField:
        if self.dtype == core_defs.BoolDType():
            return _make_builtin("logical_xor", "logical_xor")(self, other)
        raise NotImplementedError("'__xor__' not implemented for non-'bool' fields.")

    __rxor__ = __xor__

    def __invert__(self) -> NdArrayField:
        if self.dtype == core_defs.BoolDType():
            return _make_builtin("invert", "invert")(self)
        raise NotImplementedError("'__invert__' not implemented for non-'bool' fields.")

    def _slice(
        self, index: common.AnyIndexSpec
    ) -> tuple[common.Domain, common.RelativeIndexSequence]:
        new_domain = embedded_common.sub_domain(self.domain, index)

        index_sequence = common.as_any_index_sequence(index)
        slice_ = (
            _get_slices_from_domain_slice(self.domain, index_sequence)
            if common.is_absolute_index_sequence(index_sequence)
            else index_sequence
        )
        assert common.is_relative_index_sequence(slice_)
        return new_domain, slice_


@dataclasses.dataclass(frozen=True)
class NdArrayConnectivityField(  # type: ignore[misc] # for __ne__, __eq__
    common.ConnectivityField[common.DimsT, common.DimT],
    NdArrayField[common.DimsT, core_defs.IntegralScalar],
):
    _codomain: common.DimT
    _skip_value: Optional[core_defs.IntegralScalar]

    @functools.cached_property
    def _cache(self) -> dict:
        return {}

    @classmethod
    def __gt_builtin_func__(cls, _: fbuiltins.BuiltInFunction) -> Never:  # type: ignore[override]
        raise NotImplementedError()

    @property
    # type: ignore[override] # TODO(havogt): instead of inheriting from NdArrayField, steal implementation or common base
    def codomain(self) -> common.DimT:
        return self._codomain

    @property
    def skip_value(self) -> Optional[core_defs.IntegralScalar]:
        return self._skip_value

    @functools.cached_property
    def kind(self) -> common.ConnectivityKind:
        kind = common.ConnectivityKind.MODIFY_STRUCTURE
        if self.domain.ndim > 1:
            kind |= common.ConnectivityKind.MODIFY_RANK
            kind |= common.ConnectivityKind.MODIFY_DIMS
        if self.domain.dim_index(self.codomain) is None:
            kind |= common.ConnectivityKind.MODIFY_DIMS

        return kind

    @classmethod
    def from_array(  # type: ignore[override]
        cls,
        data: npt.ArrayLike | core_defs.NDArrayObject,
        /,
        codomain: common.DimT,
        *,
        domain: common.DomainLike,
        dtype: Optional[core_defs.DTypeLike] = None,
        skip_value: Optional[core_defs.IntegralScalar] = None,
    ) -> NdArrayConnectivityField:
        domain = common.domain(domain)
        xp = cls.array_ns

        xp_dtype = None if dtype is None else xp.dtype(core_defs.dtype(dtype).scalar_type)
        array = xp.asarray(data, dtype=xp_dtype)

        if dtype is not None:
            assert array.dtype.type == core_defs.dtype(dtype).scalar_type

        assert issubclass(array.dtype.type, core_defs.INTEGRAL_TYPES)

        assert all(isinstance(d, common.Dimension) for d in domain.dims), domain
        assert len(domain) == array.ndim
        assert all(len(r) == s or s == 1 for r, s in zip(domain.ranges, array.shape))

        assert isinstance(codomain, common.Dimension)

        return cls(
            domain,
            array,
            codomain,
            _skip_value=skip_value,
        )

    def inverse_image(
        self, image_range: common.UnitRange | common.NamedRange
    ) -> Sequence[common.NamedRange]:
        cache_key = hash((id(self.ndarray), self.domain, image_range))

        if (new_dims := self._cache.get(cache_key, None)) is None:
            xp = self.array_ns

            if not isinstance(
                image_range, common.UnitRange
            ):  # TODO(havogt): cleanup duplication with CartesianConnectivity
                if image_range[0] != self.codomain:
                    raise ValueError(
                        f"Dimension '{image_range[0]}' does not match the codomain dimension '{self.codomain}'."
                    )

                image_range = image_range[1]

            assert isinstance(image_range, common.UnitRange)

            assert common.UnitRange.is_finite(image_range)

            relative_ranges = _hypercube(self._ndarray, image_range, xp, self.skip_value)

            if relative_ranges is None:
                raise ValueError("Restriction generates non-contiguous dimensions.")

            new_dims = _relative_ranges_to_domain(relative_ranges, self.domain)

            self._cache[cache_key] = new_dims

        return new_dims

    def restrict(self, index: common.AnyIndexSpec) -> common.Field:
        cache_key = (id(self.ndarray), self.domain, index)

        if (restricted_connectivity := self._cache.get(cache_key, None)) is None:
            cls = self.__class__
            xp = cls.array_ns
            new_domain, buffer_slice = self._slice(index)
            new_buffer = xp.asarray(self.ndarray[buffer_slice])
            restricted_connectivity = cls(new_domain, new_buffer, self.codomain, self.skip_value)
            self._cache[cache_key] = restricted_connectivity

        return restricted_connectivity

    __getitem__ = restrict


def _relative_ranges_to_domain(
    relative_ranges: tuple[common.UnitRange, ...], domain: common.Domain
) -> common.Domain:
    return common.Domain(
        dims=domain.dims, ranges=[rr + ar.start for ar, rr in zip(domain.ranges, relative_ranges)]
    )


def _hypercube(
    index_array: core_defs.NDArrayObject,
    image_range: common.UnitRange,
    xp: ModuleType,
    skip_value: Optional[core_defs.IntegralScalar] = None,
) -> Optional[list[common.UnitRange]]:
    """
    Return the hypercube that contains all indices in `index_array` that are within `image_range`, or `None` if no such hypercube exists.

    If `skip_value` is given, the selected values are ignored. It returns the smallest hypercube.
    A bigger hypercube could be constructed by adding lines that contain only `skip_value`s.
    Example:
    index_array =  0  1 -1
                   3  4 -1
                  -1 -1 -1
    skip_value = -1
    would currently select the 2x2 range [0,2], [0,2], but could also select the 3x3 range [0,3], [0,3].
    """
    select_mask = (index_array >= image_range.start) & (index_array < image_range.stop)
    ignore_mask = None if skip_value is None else index_array == skip_value
    return _hypercube_from_mask(select_mask, xp, ignore_mask)


def _hypercube_from_mask(
    select_mask: core_defs.NDArrayObject,
    xp: ModuleType,
    ignore_mask: Optional[core_defs.NDArrayObject] = None,
) -> list[common.UnitRange]:
    nnz: tuple[core_defs.NDArrayObject, ...] = xp.nonzero(select_mask)
    slices = tuple(
        slice(xp.min(dim_nnz_indices), xp.max(dim_nnz_indices) + 1)
        if dim_nnz_indices.size > 0
        else slice(0, 0)  # TODO test this code path
        for dim_nnz_indices in nnz
    )
    hcube = select_mask[tuple(slices)]
    if ignore_mask is not None:
        hcube |= ignore_mask[tuple(slices)]
    if not xp.all(hcube):
        return None

    return [common.UnitRange(s.start, s.stop) for s in slices]


# -- Specialized implementations for builtin operations on array fields --

NdArrayField.register_builtin_func(
    fbuiltins.abs, NdArrayField.__abs__  # type: ignore[attr-defined]
)
NdArrayField.register_builtin_func(
    fbuiltins.power, NdArrayField.__pow__  # type: ignore[attr-defined]
)
# TODO gamma

for name in (
    fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES
    + fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES
    + fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES
):
    if name in ["abs", "power", "gamma"]:
        continue
    NdArrayField.register_builtin_func(getattr(fbuiltins, name), _make_builtin(name, name))

NdArrayField.register_builtin_func(
    fbuiltins.minimum, _make_builtin("minimum", "minimum")  # type: ignore[attr-defined]
)
NdArrayField.register_builtin_func(
    fbuiltins.maximum, _make_builtin("maximum", "maximum")  # type: ignore[attr-defined]
)
NdArrayField.register_builtin_func(
    fbuiltins.fmod, _make_builtin("fmod", "fmod")  # type: ignore[attr-defined]
)
NdArrayField.register_builtin_func(fbuiltins.where, _make_builtin("where", "where"))


def _mask_ranges(mask: core_defs.NDArrayObject) -> list[tuple[bool, common.UnitRange]]:
    # TODO: does it make sense to upgrade this naive algorithm to numpy?
    assert mask.ndim == 1
    cur = mask[0]
    ind = 0
    res = []
    for i in range(1, mask.shape[0]):
        if mask[i] != cur:
            res.append((cur, common.UnitRange(ind, i)))
            cur = mask[i]
            ind = i
    res.append((cur, common.UnitRange(ind, mask.shape[0])))
    return res


def _trim(lst: list[tuple[bool, common.Domain]]) -> list[tuple[bool, common.Domain]]:
    res = []
    found = False
    for v, d in lst:
        if d != common.Domain():
            res.append((v, d))
            found = True
        else:
            if found:
                raise ValueError("Out of bounds.")
    return res


def _concat_where(m, t, f):
    xp = t.__class__.array_ns
    if m.domain.ndim != 1:
        raise ValueError("Can only concatenate fields with a 1-dimensional mask.")
    mask_dim = m.domain.dims[0]

    relative_mask_ranges = _mask_ranges(m.ndarray)
    mask_domains = [
        (v, _relative_ranges_to_domain((mr,), m.domain)) for v, mr in relative_mask_ranges
    ]
    t_domain = t.domain if common.is_field(t) else common.Domain()
    f_domain = f.domain if common.is_field(f) else common.Domain()
    intersected_domains = [
        (v, embedded_common.intersect_domains(t_domain if v else f_domain, d))
        for v, d in mask_domains
    ]

    assert t.domain.dims == f.domain.dims  # TODO implement broadcasting

    transformed = []
    for v, d in _trim(intersected_domains):
        if v:
            if common.is_field(t):
                slices = _get_slices_from_domain_slice(t.domain, d)
                transformed.append(xp.asarray(t.ndarray[slices]))
            else:
                transformed.append(t)
        else:
            if common.is_field(f):
                slices = _get_slices_from_domain_slice(f.domain, d)
                transformed.append(xp.asarray(f.ndarray[slices]))
            else:
                transformed.append(f)
    mask_index = t_domain.dim_index(mask_dim)
    result = xp.concatenate(transformed, axis=mask_index)
    result_domain = t_domain.replace(
        mask_dim,
        (
            mask_dim,
            common.UnitRange(
                intersected_domains[0][1][mask_index][1].start,
                intersected_domains[-1][1][mask_index][1].stop,
            ),
        ),
    )
    return t.__class__.from_array(result, domain=result_domain)


# def _concat_where(m, t, f):
#     xp = t.__class__.array_ns
#     if m.domain.ndim != 1:
#         raise ValueError("Can only concatenate fields with a 1-dimensional mask.")
#     mask_dim = m.domain.dims[0]
#     mask_true_domain = _relative_ranges_to_domain(_hypercube_from_mask(m.ndarray, xp), m.domain)
#     t_domain = t.domain if common.is_field(t) else common.Domain()
#     true_domain = embedded_common.intersect_domains(t_domain, mask_true_domain)
#     mask_false_domain = _relative_ranges_to_domain(_hypercube_from_mask(~m.ndarray, xp), m.domain)
#     f_domain = f.domain if common.is_field(f) else common.Domain()
#     false_domain = embedded_common.intersect_domains(f_domain, mask_false_domain)

#     if common.is_field(t):
#         true_slices = _get_slices_from_domain_slice(t.domain, true_domain)
#         true_transformed = xp.asarray(t.ndarray[true_slices])
#     else:
#         true_transformed = t
#     if common.is_field(f):
#         false_slices = _get_slices_from_domain_slice(f.domain, false_domain)
#         false_transformed = xp.asarray(f.ndarray[false_slices])
#     else:
#         false_transformed = f

#     assert true_domain.dims == false_domain.dims  # TODO implement broadcasting
#     mask_index = true_domain.dim_index(mask_dim)
#     if true_domain[mask_dim][1].stop == false_domain[mask_dim][1].start:
#         result = xp.concatenate((true_transformed, false_transformed), axis=mask_index)
#         result_domain = true_domain.replace(
#             mask_dim,
#             (
#                 mask_dim,
#                 common.UnitRange(true_domain[mask_dim][1].start, false_domain[mask_dim][1].stop),
#             ),
#         )
#     elif false_domain[mask_dim][1].stop == true_domain[mask_dim][1].start:
#         result = xp.concatenate((false_transformed, true_transformed), axis=mask_index)
#         result_domain = true_domain.replace(
#             mask_dim,
#             (
#                 mask_dim,
#                 common.UnitRange(false_domain[mask_dim][1].start, true_domain[mask_dim][1].stop),
#             ),
#         )
#         # concatenate false then true
#     else:
#         raise ValueError("Mask does not split the domain into two non-overlapping parts.")

#     return t.__class__.from_array(result, domain=result_domain)


NdArrayField.register_builtin_func(fbuiltins.concat_where, _concat_where)


def _make_reduction(
    builtin_name: str, array_builtin_name: str, initial_value_op: Callable
) -> Callable[..., NdArrayField[common.DimsT, core_defs.ScalarT],]:
    def _builtin_op(
        field: NdArrayField[common.DimsT, core_defs.ScalarT], axis: common.Dimension
    ) -> NdArrayField[common.DimsT, core_defs.ScalarT]:
        xp = field.array_ns

        if not axis.kind == common.DimensionKind.LOCAL:
            raise ValueError("Can only reduce local dimensions.")
        if axis not in field.domain.dims:
            raise ValueError(f"Field can not be reduced as it doesn't have dimension '{axis}'.")
        if len([d for d in field.domain.dims if d.kind is common.DimensionKind.LOCAL]) > 1:
            raise NotImplementedError(
                "Reducing a field with more than one local dimension is not supported."
            )
        reduce_dim_index = field.domain.dims.index(axis)
        current_offset_provider = embedded_context.offset_provider.get(None)
        assert current_offset_provider is not None
        offset_definition = current_offset_provider[
            axis.value
        ]  # assumes offset and local dimension have same name
        assert isinstance(offset_definition, itir_embedded.NeighborTableOffsetProvider)
        new_domain = common.Domain(*[nr for nr in field.domain if nr[0] != axis])

        broadcast_slice = tuple(
            slice(None) if d in [axis, offset_definition.origin_axis] else xp.newaxis
            for d in field.domain.dims
        )
        masked_array = xp.where(
            xp.asarray(offset_definition.table[broadcast_slice]) != common.SKIP_VALUE,
            field.ndarray,
            initial_value_op(field),
        )

        return field.__class__.from_array(
            getattr(xp, array_builtin_name)(
                masked_array,
                axis=reduce_dim_index,
            ),
            domain=new_domain,
        )

    _builtin_op.__name__ = builtin_name
    return _builtin_op


NdArrayField.register_builtin_func(
    fbuiltins.neighbor_sum, _make_reduction("neighbor_sum", "sum", lambda x: x.dtype.scalar_type(0))
)
NdArrayField.register_builtin_func(
    fbuiltins.max_over, _make_reduction("max_over", "max", lambda x: x.array_ns.min(x._ndarray))
)
NdArrayField.register_builtin_func(
    fbuiltins.min_over, _make_reduction("min_over", "min", lambda x: x.array_ns.max(x._ndarray))
)


# -- Concrete array implementations --
# NumPy
_nd_array_implementations = [np]


@dataclasses.dataclass(frozen=True, eq=False)
class NumPyArrayField(NdArrayField):
    array_ns: ClassVar[ModuleType] = np


common._field.register(np.ndarray, NumPyArrayField.from_array)


@dataclasses.dataclass(frozen=True, eq=False)
class NumPyArrayConnectivityField(NdArrayConnectivityField):
    array_ns: ClassVar[ModuleType] = np


common._connectivity.register(np.ndarray, NumPyArrayConnectivityField.from_array)

# CuPy
if cp:
    _nd_array_implementations.append(cp)

    @dataclasses.dataclass(frozen=True, eq=False)
    class CuPyArrayField(NdArrayField):
        array_ns: ClassVar[ModuleType] = cp

    common._field.register(cp.ndarray, CuPyArrayField.from_array)

    @dataclasses.dataclass(frozen=True, eq=False)
    class CuPyArrayConnectivityField(NdArrayConnectivityField):
        array_ns: ClassVar[ModuleType] = cp

    common._connectivity.register(cp.ndarray, CuPyArrayConnectivityField.from_array)

# JAX
if jnp:
    _nd_array_implementations.append(jnp)

    @dataclasses.dataclass(frozen=True, eq=False)
    class JaxArrayField(NdArrayField):
        array_ns: ClassVar[ModuleType] = jnp

        def __setitem__(
            self,
            index: common.AnyIndexSpec,
            value: common.Field | core_defs.NDArrayObject | core_defs.ScalarT,
        ) -> None:
            # TODO(havogt): use something like `self.ndarray = self.ndarray.at(index).set(value)`
            raise NotImplementedError("'__setitem__' for JaxArrayField not yet implemented.")

    common._field.register(jnp.ndarray, JaxArrayField.from_array)


def _broadcast(field: common.Field, new_dimensions: tuple[common.Dimension, ...]) -> common.Field:
    domain_slice: list[slice | None] = []
    named_ranges = []
    for dim in new_dimensions:
        if (pos := embedded_common._find_index_of_dim(dim, field.domain)) is not None:
            domain_slice.append(slice(None))
            named_ranges.append((dim, field.domain[pos][1]))
        else:
            domain_slice.append(field.__class__.array_ns.newaxis)
            named_ranges.append((dim, common.UnitRange.infinite()))
    return common._field(field.ndarray[tuple(domain_slice)], domain=common.Domain(*named_ranges))


def _builtins_broadcast(
    field: common.Field | core_defs.Scalar, new_dimensions: tuple[common.Dimension, ...]
) -> common.Field:  # separated for typing reasons
    if common.is_field(field):
        return _broadcast(field, new_dimensions)
    raise AssertionError("Scalar case not reachable from 'fbuiltins.broadcast'.")


NdArrayField.register_builtin_func(fbuiltins.broadcast, _builtins_broadcast)


def _astype(field: common.Field | core_defs.ScalarT | tuple, type_: type) -> NdArrayField:
    if isinstance(field, NdArrayField):
        return field.__class__.from_array(field.ndarray.astype(type_), domain=field.domain)
    raise AssertionError("This is the NdArrayField implementation of 'fbuiltins.astype'.")


NdArrayField.register_builtin_func(fbuiltins.astype, _astype)


def _get_slices_from_domain_slice(
    domain: common.Domain,
    domain_slice: common.Domain | Sequence[common.NamedRange | common.NamedIndex | Any],
) -> common.RelativeIndexSequence:
    """Generate slices for sub-array extraction based on named ranges or named indices within a Domain.

    This function generates a tuple of slices that can be used to extract sub-arrays from a field. The provided
    named ranges or indices specify the dimensions and ranges of the sub-arrays to be extracted.

    Args:
        domain (common.Domain): The Domain object representing the original field.
        domain_slice (DomainSlice): A sequence of dimension names and associated ranges.

    Returns:
        tuple[slice | int | None, ...]: A tuple of slices representing the sub-array extraction along each dimension
                                       specified in the Domain. If a dimension is not included in the named indices
                                       or ranges, a None is used to indicate expansion along that axis.
    """
    slice_indices: list[slice | common.IntIndex] = []

    for pos_old, (dim, _) in enumerate(domain):
        if (pos := embedded_common._find_index_of_dim(dim, domain_slice)) is not None:
            index_or_range = domain_slice[pos][1]
            slice_indices.append(_compute_slice(index_or_range, domain, pos_old))
        else:
            slice_indices.append(slice(None))
    return tuple(slice_indices)


def _compute_slice(
    rng: common.UnitRange | common.IntIndex, domain: common.Domain, pos: int
) -> slice | common.IntIndex:
    """Compute a slice or integer based on the provided range, domain, and position.

    Args:
        rng (DomainRange): The range to be computed as a slice or integer.
        domain (common.Domain): The domain containing dimension information.
        pos (int): The position of the dimension in the domain.

    Returns:
        slice | int: Slice if `new_rng` is a UnitRange, otherwise an integer.

    Raises:
        ValueError: If `new_rng` is not an integer or a UnitRange.
    """
    if isinstance(rng, common.UnitRange):
        start = (
            rng.start - domain.ranges[pos].start
            if common.UnitRange.is_left_finite(domain.ranges[pos])
            else None
        )
        stop = (
            rng.stop - domain.ranges[pos].start
            if common.UnitRange.is_right_finite(domain.ranges[pos])
            else None
        )
        return slice(start, stop)
    elif common.is_int_index(rng):
        assert common.Domain.is_finite(domain)
        return rng - domain.ranges[pos].start
    else:
        raise ValueError(f"Can only use integer or UnitRange ranges, provided type: '{type(rng)}'.")
