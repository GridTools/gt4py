# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, Never, Optional, ParamSpec, TypeAlias, TypeVar, cast

import array_api_compat
import numpy as np
from numpy import typing as npt

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import (
    array_ops,
    common as embedded_common,
    context as embedded_context,
    exceptions as embedded_exceptions,
)
from gt4py.next.ffront import experimental, fbuiltins


def _ops_of(*values: Any) -> array_ops.ArrayOps:
    # TODO(havogt): the first field's array library decides, so mixing libraries depends on the
    #   order of the operands; reject mixed libraries or define a priority instead.
    for v in values:
        if isinstance(v, NdArrayField):
            return array_ops.of(v._ndarray)
    raise AssertionError("No 'NdArrayField' found in the arguments.")


def _make_builtin(
    builtin_name: str, array_builtin_name: str, reverse: bool = False
) -> Callable[..., NdArrayField]:
    def _builtin_op(*fields: common.Field | core_defs.Scalar) -> NdArrayField:
        ops = _ops_of(*fields)
        op = ops.elementwise_function(array_builtin_name)

        domain_intersection = embedded_common.domain_intersection(
            *[f.domain for f in fields if isinstance(f, common.Field)]
        )

        arrays: list[core_defs.NDArrayObject] = []
        transformed: list[core_defs.NDArrayObject | core_defs.Scalar] = []
        for f in fields:
            if isinstance(f, common.Field):
                if f.domain == domain_intersection:
                    array = ops.asarray(f.ndarray)
                else:
                    f_broadcasted = _broadcast(f, domain_intersection.dims)
                    f_slices = _get_slices_from_domain_slice(
                        f_broadcasted.domain, domain_intersection
                    )
                    array = ops.asarray(f_broadcasted.ndarray[f_slices])
                arrays.append(array)
                transformed.append(array)
            else:
                assert core_defs.is_scalar_type(f)
                transformed.append(f)
        if len(arrays) < len(transformed):
            transformed = [ops.operand(v, arrays[0]) for v in transformed]
        if reverse:
            transformed.reverse()
        new_data = op(*transformed)
        return NdArrayField.from_array(new_data, domain=domain_intersection)

    _builtin_op.__name__ = builtin_name
    return _builtin_op


def _asarray(
    data: npt.ArrayLike | core_defs.NDArrayObject, dtype: Optional[core_defs.DTypeLike]
) -> core_defs.NDArrayObject:
    try:
        ops = array_ops.of(data)
    except TypeError:
        ops = array_ops.default()
    scalar_type = None if dtype is None else core_defs.dtype(dtype).scalar_type
    array = ops.asarray(data, dtype=scalar_type)
    assert ops.scalar_type(array.dtype) in (
        core_defs.SCALAR_TYPES if scalar_type is None else (scalar_type,)
    )
    return array


_Value: TypeAlias = common.Field | core_defs.ScalarT
_P = ParamSpec("_P")
_R = TypeVar("_R", _Value, tuple[_Value, ...])


@dataclasses.dataclass(frozen=True)
class NdArrayField(common.MutableField[common.DimsT, core_defs.ScalarT]):
    """
    Field backed by an array of any Array API compatible library.

    The array operations are taken from the `array_ops.ArrayOps` of the buffer's
    namespace, so one class serves all array libraries.
    """

    _domain: common.Domain
    _ndarray: core_defs.NDArrayObject

    @property
    def array_ns(self) -> ModuleType:
        """The module of the array library holding the buffer, e.g. `numpy`."""
        # TODO(havogt): return the Array API namespace (`array_ops.of(self._ndarray).xp`) instead of the library's
        #   own module; users comparing it to `numpy` should query the device instead.
        return array_ops.of(self._ndarray).array_ns

    @classmethod
    def __gt_builtin_func__(cls, /, func: fbuiltins.BuiltInFunction[_R, _P]) -> Callable[_P, _R]:
        return _builtin_implementations.get(func, NotImplemented)

    def __getstate__(self) -> dict[str, Any]:
        # Serialize only the dataclass fields, excluding cached properties
        # stored in `__dict__` (which may not be picklable).
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

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
        array = _asarray(data, dtype)

        assert all(isinstance(d, common.Dimension) for d in domain.dims), domain
        assert len(domain) == array.ndim
        assert all(s == 1 or len(r) == s for r, s in zip(domain.ranges, array.shape))

        return cls(domain, array)

    @functools.cached_property
    def __gt_origin__(self) -> tuple[int, ...]:
        assert common.Domain.is_finite(self.domain)
        return tuple(-r.start for r in self.domain.ranges)

    @functools.cached_property
    def __gt_buffer_info__(self) -> common.BufferInfo:
        """
        Interface to retrieve the low-level description of a Field buffer.

        Since by default NdArrayFields are implemented as frozen dataclasses,
        and therefore the backing ndarray cannot be replaced after creation,
        this is implemented as a cached property for performance reasons.

        NDArrayField subclasses where the backing ndarray can be replaced
        should override this and make it a regular property.
        """
        return array_ops.of(self._ndarray).buffer_info(self.ndarray)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._ndarray.shape

    @property
    def domain(self) -> common.Domain:
        return self._domain

    @property
    def codomain(self) -> type[core_defs.ScalarT]:
        return self.dtype.scalar_type

    @functools.cached_property
    def dtype(self) -> core_defs.DType[core_defs.ScalarT]:
        return core_defs.dtype(array_ops.of(self._ndarray).scalar_type(self._ndarray.dtype))

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        return self._ndarray

    def asnumpy(self) -> np.ndarray:
        return array_ops.of(self._ndarray).to_numpy(self._ndarray)

    def as_scalar(self) -> core_defs.ScalarT:
        if self.domain.ndim != 0:
            raise ValueError(
                f"'as_scalar' is only valid on 0-dimensional 'Field's, got a {self.domain.ndim}-dimensional 'Field'."
            )
        # note: `.item()` will return a Python type, therefore we use indexing with an empty tuple
        return self.asnumpy()[()]  # type: ignore[return-value] # should be ensured by the 0-d check

    def premap(
        self: NdArrayField,
        *connectivities: common.Connectivity | fbuiltins.FieldOffset,
    ) -> NdArrayField:
        """
        Rearrange the field content using the provided connectivities (index mappings).

        This operation is conceptually equivalent to a regular composition of mappings
        `f∘c`, being `c` the `connectivity` argument and `f` the `self` data field.
        Note that the connectivity field appears at the right of the composition
        operator and the data field at the left.

        The composition operation is only well-defined when the codomain of `c: A → B`
        matches the domain of `f: B → ℝ` and it would then result in a new mapping
        `f∘c: A → ℝ` defined as `(f∘c)(x) = f(c(x))`. When remaping a field whose
        domain has multiple dimensions `f: A × B → ℝ`, the domain of the connectivity
        argument used in the right hand side of the operator should therefore have the
        same product of dimensions `c: S × T → A × B`. Such a mapping can also be
        expressed as a pair of mappings `c1: S × T → A` and `c2: S × T → B`, and this
        is actually the only supported form in GT4Py because `Connectivity` instances
        can only deal with a single dimension in its codomain. This approach makes
        connectivities reusable for any combination of dimensions in a field domain
        and matches the NumPy advanced indexing API, which basically is a
        composition of mappings of natural numbers representing tensor indices.

        In general, the `premap()` function is able to deal with data fields with multiple
        dimensions even if only one connectivity is passed. Connectivity arguments are then
        expanded to fully defined connectivities for each dimension in the domain of the
        field according to some rules covering the most common use cases.

        Assuming a field `f: Field[Dims[A, B], DT]` the following cases are supported:

        - If the connectivity domain only contains dimensions which are NOT part of the
          field domain (new dimensions), this function will use the same rules of
          advanced-indexing and replace the connectivity codomain dimension by its domain
          dimensions. A way to think about this is that the data field is transformed into
          a curried mapping whose domain only contains the connectivity codomain dimension,
          then composed as usual with the connectivity, and finally uncurried again:

            `f: A × B → ℝ` => `f': A → (B → ℝ)`
            `c: X × Y → A`
            `(f'∘c): X × Y → (B → ℝ)` => `(f∘c): X × Y × B → ℝ`

        - If the connectivity domain only contains dimensions which are ALREADY part of the
          data field domain, the connectivity field would be interpreted as an homomorphic
          function which preserves the domain dimensions. A way to think about this is that
          the connectivity defines how the current field data gets translated and rearranged
          into new domain ranges, and the mappings for the missing domain dimensions
          are assumed to be identities:

            `f: A × B × C → ℝ`
            `c: A × B → A` => `c0: A × B × C → A`, `c1: A × B × C → B`, `c2: A × B × C → C`
            `(f∘c): A × B × C → ℝ` => `(f∘(c0 × c1 × c2)): A × B × C → ℝ)`

        Note that cartesian shifts (e.g. `I → I_half`, `(I+1): I → I`) are just simpler
        versions of these cases where the internal structure of the data (codomain) is
        preserved and therefore the `premap` operation can be implemented as a compact
        domain translation (i.e. only transform the domain without altering the data). Such affine
        connectivities only relabel the domain; data-rearranging cases are handled as
        advanced-indexing gathers (:class:`common.GatherConnectivity`).

        Args:
            *connectivities: connectivities to be used for the `premap` operation. If only one
                connectivity is passed, it will be expanded to fully defined connectivities for
                each dimension in the domain of the field according to the rules described above.
                If more than one connectivity is passed, they all must satisfy:
                - they are either all affine (domain-only) or all gather connectivities
                - their codomains are pairwise distinct
                - no connectivity reads a dimension that another one replaces (i.e. removes
                  from the field domain by introducing new dimensions in its place)

        """  # noqa: RUF002  # TODO(egparedes): move docstring to the `premap` builtin function when it exists

        conn_fields: list[common.Connectivity] = []
        codomains_counter: collections.Counter[common.Dimension] = collections.Counter()

        for connectivity in connectivities:
            # For neighbor reductions, a FieldOffset is passed instead of an actual Connectivity
            if not isinstance(connectivity, common.Connectivity):
                assert isinstance(connectivity, fbuiltins.FieldOffset)
                connectivity = connectivity.as_connectivity_field()
            assert isinstance(connectivity, common.Connectivity)

            # the neighbor reductions mask skipped neighbors by `common._DEFAULT_SKIP_VALUE`
            assert connectivity.skip_value is None or connectivity.skip_value == -1

            conn_fields.append(connectivity)
            codomains_counter[connectivity.codomain] += 1

        if unknown_dims := [dim for dim in codomains_counter.keys() if dim not in self.domain.dims]:
            raise ValueError(
                f"Incompatible dimensions in the connectivity codomain(s) {unknown_dims}"
                f"while pre-mapping a field with domain {self.domain}."
            )

        if repeated_codomain_dims := [dim for dim, count in codomains_counter.items() if count > 1]:
            raise ValueError(
                "All connectivities must have different codomains but some are repeated:"
                f" {repeated_codomain_dims}."
            )

        is_gather = [isinstance(c, common.GatherConnectivity) for c in conn_fields]
        if any(is_gather) and not all(is_gather):
            raise ValueError(
                "Mixing connectivities that rearrange the data (gather) with connectivities that "
                "only relabel the domain (affine) is not allowed."
            )

        if not any(is_gather):
            return _domain_premap(self, *conn_fields)

        # Reject only order-dependent chains: a connectivity reading a dimension that another
        # one removes (one whose codomain is not in its own domain, so it gets replaced in the
        # output). Connectivities that keep their codomain do not remove any dimension and can
        # always be combined.
        removed = {c.codomain for c in conn_fields if c.codomain not in c.domain.dims}
        for c in conn_fields:
            if reads := removed & ({*c.domain.dims} - {c.codomain}):
                raise ValueError(
                    f"Cannot 'premap' with connectivities where one reads dimension(s) {reads} that "
                    "another replaces; apply such chained remaps in separate 'premap' calls."
                )

        return _gather_premap(self, *cast(list[common.GatherConnectivity], conn_fields))

    def __call__(
        self,
        index_field: common.Connectivity | fbuiltins.FieldOffset,
        *args: common.Connectivity | fbuiltins.FieldOffset,
    ) -> common.Field:
        return functools.reduce(
            lambda field, current_index_field: field.premap(current_index_field),
            [index_field, *args],
            self,
        )

    def restrict(self, index: common.AnyIndexSpec) -> NdArrayField:
        new_domain, buffer_slice = self._slice(index)
        return self.__class__.from_array(self.ndarray[buffer_slice], domain=new_domain)

    __getitem__ = restrict

    def __setitem__(
        self: NdArrayField[common.DimsT, core_defs.ScalarT],
        index: common.AnyIndexSpec,
        value: common.Field | core_defs.NDArrayObject | core_defs.ScalarT,
    ) -> None:
        target_domain, target_slice = self._slice(index, for_assignment=True)

        if isinstance(value, common.Field):
            if not value.domain == target_domain:
                raise ValueError(
                    f"Incompatible 'Domain' in assignment. Source domain = '{value.domain}', target domain = '{target_domain}'."
                )
            value = value.ndarray

        new_buffer = array_ops.of(self._ndarray).setitem(self._ndarray, target_slice, value)
        if new_buffer is not self._ndarray:
            object.__setattr__(self, "_ndarray", new_buffer)

    __abs__ = _make_builtin("abs", "abs")

    __neg__ = _make_builtin("neg", "negative")

    __add__ = __radd__ = _make_builtin("add", "add")

    __pos__ = _make_builtin("pos", "positive")

    __sub__ = _make_builtin("sub", "subtract")
    __rsub__ = _make_builtin("sub", "subtract", reverse=True)

    __mul__ = __rmul__ = _make_builtin("mul", "multiply")

    __truediv__ = _make_builtin("div", "divide")
    __rtruediv__ = _make_builtin("div", "divide", reverse=True)

    __floordiv__ = _make_builtin("floordiv", "floor_divide")
    __rfloordiv__ = _make_builtin("floordiv", "floor_divide", reverse=True)

    __pow__ = _make_builtin("pow", "power")

    __mod__ = _make_builtin("mod", "mod")
    __rmod__ = _make_builtin("mod", "mod", reverse=True)

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
        self, index: common.AnyIndexSpec, *, for_assignment: bool = False
    ) -> tuple[common.Domain, common.RelativeIndexSequence]:
        index = embedded_common.canonicalize_any_index_sequence(index)
        new_domain = embedded_common.sub_domain(self.domain, index)

        index_sequence = common.as_any_index_sequence(index)
        if common.is_absolute_index_sequence(index_sequence):
            buffer_slice = _get_slices_from_domain_slice(self.domain, index_sequence)
            if for_assignment:
                # writing the only element of a broadcast axis would change its whole range
                return new_domain, buffer_slice
            return new_domain, _broadcast_axes_slice(buffer_slice, self.domain, self._ndarray.shape)
        assert common.is_relative_index_sequence(index_sequence)
        if len(index_sequence) < self._ndarray.ndim and Ellipsis not in index_sequence:
            return new_domain, (*index_sequence, Ellipsis)
        return new_domain, index_sequence


def _broadcast_axes_slice(
    slice_: common.RelativeIndexSequence, domain: common.Domain, shape: tuple[int, ...]
) -> common.RelativeIndexSequence:
    """`slice_` with every index into a size-1 axis spanning a longer range (a broadcast axis) reading its only element."""
    return tuple(
        (slice(None) if isinstance(s, slice) else 0)
        if size == 1 and not (common.UnitRange.is_finite(rng) and len(rng) == 1)
        else s
        for s, size, rng in zip(slice_, shape, domain.ranges, strict=True)
    )


@dataclasses.dataclass(frozen=True)
class NdArrayConnectivityField(
    common.GatherConnectivity[common.DimsT, common.DimT],
    NdArrayField[common.DimsT, core_defs.IntegralScalar],
):
    _codomain: common.DimT
    _skip_value: Optional[core_defs.IntegralScalar]

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
        array = _asarray(data, dtype)

        assert issubclass(array_ops.of(array).scalar_type(array.dtype), core_defs.INTEGRAL_TYPES)

        assert all(isinstance(d, common.Dimension) for d in domain.dims), domain
        assert len(domain) == array.ndim
        assert all(len(r) == s or s == 1 for r, s in zip(domain.ranges, array.shape))

        assert isinstance(codomain, common.Dimension)

        return cls(domain, array, codomain, _skip_value=skip_value)

    @classmethod
    def __gt_builtin_func__(cls, _: fbuiltins.BuiltInFunction) -> Never:  # type: ignore[override]
        raise NotImplementedError()

    @property
    def codomain(self) -> common.DimT:  # type: ignore[override] # TODO(havogt): instead of inheriting from NdArrayField, steal implementation or common base
        return self._codomain

    @property
    def skip_value(self) -> Optional[core_defs.IntegralScalar]:
        return self._skip_value

    # This embedded run-time cache is only used to speed up repeated calls to
    # `inverse_image` and `restrict`, and it should not be considered part of
    # the connectivity field definition, and therefore it should not be serialized.
    @functools.cached_property
    def _cache(self) -> dict:
        return {}

    def inverse_image(self, image_range: common.UnitRange | common.NamedRange) -> common.Domain:
        cache_key = hash((id(self.ndarray), self.domain, image_range))

        if (new_domain := self._cache.get(cache_key, None)) is None:
            if not isinstance(
                image_range, common.UnitRange
            ):  # TODO(havogt): cleanup duplication with CartesianConnectivity
                if image_range.dim != self.codomain:
                    raise ValueError(
                        f"Dimension '{image_range.dim}' does not match the codomain dimension '{self.codomain}'."
                    )

                image_range = image_range.unit_range

            assert isinstance(image_range, common.UnitRange)
            assert common.UnitRange.is_finite(image_range)

            slices = _hyperslice(
                self._ndarray, image_range, array_ops.of(self._ndarray).xp, self.skip_value
            )
            if slices is None:
                raise ValueError("Restriction generates non-contiguous or empty dimensions.")

            new_domain = self.domain.slice_at[slices]
            self._cache[cache_key] = new_domain

        return new_domain

    def restrict(self, index: common.AnyIndexSpec) -> NdArrayConnectivityField:
        cache_key = (id(self.ndarray), self.domain, index)

        if (restricted_connectivity := self._cache.get(cache_key, None)) is None:
            new_domain, buffer_slice = self._slice(index)
            restricted_connectivity = self.__class__(
                new_domain,
                array_ops.of(self._ndarray).asarray(self.ndarray[buffer_slice]),
                self.codomain,
                self.skip_value,
            )
            self._cache[cache_key] = restricted_connectivity

        return restricted_connectivity

    __getitem__ = restrict


def _domain_premap(data: NdArrayField, *connectivities: common.Connectivity) -> NdArrayField:
    """`premap` implementation transforming only the field domain not the data (i.e. translation and relocation)."""
    new_domain = data.domain
    for connectivity in connectivities:
        dim = connectivity.codomain
        dim_idx = data.domain.dim_index(dim)
        if dim_idx is None:
            raise ValueError(
                f"Incompatible index field expects a data field with dimension '{dim}'"
                f"but got '{data.domain}'."
            )

        current_range: common.UnitRange = data.domain[dim_idx].unit_range
        new_ranges = connectivity.inverse_image(current_range)
        new_domain = new_domain.replace(dim_idx, *new_ranges)

    return data.__class__.from_array(data._ndarray, domain=new_domain, dtype=data.dtype)


def _gather_output_domain(
    field_domain: common.Domain, connectivities: Sequence[common.GatherConnectivity]
) -> common.Domain:
    """Output domain of a simultaneous gather: each codomain is replaced by the dimensions of its
    connectivity's domain; dimensions shared with the field domain are intersected in place."""
    domain = field_domain
    for conn in connectivities:
        cod = conn.codomain
        # the connectivity's domain, restricted to where it maps into the codomain's range
        narrowed = {
            nr.dim: nr.unit_range for nr in conn.inverse_image(field_domain[cod].unit_range)
        }
        # dimensions the connectivity adds that are not in the field yet
        introduced = [
            common.NamedRange(dim, rng) for dim, rng in narrowed.items() if dim not in domain.dims
        ]
        result: list[common.NamedRange] = []
        for nr in domain:
            if nr.dim == cod:  # the codomain expands into the connectivity's domain dimensions
                if cod in narrowed:  # keep the codomain itself when it maps to itself
                    result.append(common.NamedRange(cod, nr.unit_range & narrowed[cod]))
                result.extend(introduced)
            elif nr.dim in narrowed:  # a dimension shared with the connectivity: narrow it
                result.append(common.NamedRange(nr.dim, nr.unit_range & narrowed[nr.dim]))
            else:
                result.append(nr)
        domain = common.Domain(*result)
    return domain


def _gather_premap(data: NdArrayField, *connectivities: common.GatherConnectivity) -> NdArrayField:
    """`premap` via a single gather (dimension-preserving and -introducing cases)."""
    ops = array_ops.of(data._ndarray)
    new_domain = _gather_output_domain(data.domain, connectivities)
    if len(connectivities) == 1:
        result = _single_axis_gather_premap(data, connectivities[0], new_domain, ops)
        if result is not None:
            return result
    return _generic_gather_premap(data, connectivities, new_domain, ops)


def _generic_gather_premap(
    data: NdArrayField,
    connectivities: Sequence[common.GatherConnectivity],
    new_domain: common.Domain,
    ops: array_ops.ArrayOps,
) -> NdArrayField:
    """`premap` as one advanced-index gather; handles every case `_gather_premap` accepts."""
    conn_by_codomain = {conn.codomain: conn for conn in connectivities}

    # one index array per original field dimension (the connectivity's, or identity), broadcast over
    # the output domain and shifted to 0-based buffer indices, then a single advanced-index gather
    take_indices = tuple(
        (
            _connectivity_index_array(
                conn_by_codomain[dim],
                new_domain,
                ops,
                codomain_start=data.domain[dim].unit_range.start,
            )
            if dim in conn_by_codomain
            else _identity_index_array(new_domain, dim, ops, like=data._ndarray)
        )
        - data.domain[dim].unit_range.start
        for dim in data.domain.dims
    )
    new_buffer = data._ndarray[take_indices]
    return data.__class__.from_array(new_buffer, domain=new_domain, dtype=data.dtype)


def _single_axis_gather_premap(
    data: NdArrayField,
    connectivity: common.GatherConnectivity,
    new_domain: common.Domain,
    ops: array_ops.ArrayOps,
) -> Optional[NdArrayField]:
    """
    `premap` as one `ArrayOps.take` along the codomain axis, or `None` if the gather is not of that form.

    An optimization of `_generic_gather_premap` with the same result, except at skipped neighbors,
    whose values are unspecified. It applies when the output dimensions are the field dimensions with
    the codomain replaced, in place, by the dimensions of the connectivity's domain, i.e. when the
    connectivity reads no other dimension of the field.
    """
    dims = data.domain.dims
    axis = data.domain.dim_index(connectivity.codomain, allow_missing=False)
    if new_domain.dims != (*dims[:axis], *connectivity.domain.dims, *dims[axis + 1 :]):
        return None
    sub_domain = common.Domain(*(new_domain[d] for d in connectivity.domain.dims))
    conn = connectivity if sub_domain == connectivity.domain else connectivity.restrict(sub_domain)
    indices = ops.asarray(conn.ndarray)
    if (start := data.domain[axis].unit_range.start) != 0:
        indices = indices - start
    new_buffer = ops.take(data._ndarray, indices, axis=axis)
    return data.__class__.from_array(new_buffer, domain=new_domain, dtype=data.dtype)


def _connectivity_index_array(
    connectivity: common.GatherConnectivity,
    domain: common.Domain,
    ops: array_ops.ArrayOps,
    *,
    codomain_start: int,
) -> core_defs.NDArrayObject:
    """
    `connectivity`'s index table laid out over `domain` (not yet shifted to 0-based).

    Skip values are replaced by `codomain_start`, so that skipped neighbors index the field in bounds.
    """
    # restrict the table to the output ranges of the connectivity's own dimensions
    sub_domain = common.Domain(*(domain[d] for d in connectivity.domain.dims))
    conn = connectivity if sub_domain == connectivity.domain else connectivity.restrict(sub_domain)
    xp = ops.xp
    arr = ops.asarray(conn.ndarray)
    if conn.skip_value is not None:
        arr = xp.where(arr == conn.skip_value, codomain_start, arr)
    # the axis of `arr` for each output dimension: the connectivity's own axis, or a fresh appended one
    ndim = conn.domain.ndim
    fresh_axis = {
        dim: ndim + i for i, dim in enumerate(d for d in domain.dims if d not in conn.domain.dims)
    }
    transposed_axes = tuple(
        fresh_axis[dim] if dim in fresh_axis else conn.domain.dim_index(dim, allow_missing=False)
        for dim in domain.dims
    )
    if fresh_axis:  # add size-1 axes for output dimensions the connectivity does not span
        arr = xp.reshape(arr, (*arr.shape, *(1,) * len(fresh_axis)))
    arr = xp.permute_dims(arr, transposed_axes)  # reorder to the output dimension order
    if arr.shape != domain.shape:
        arr = xp.broadcast_to(arr, domain.shape)  # broadcast the size-1 axes to the full shape
    return arr


def _identity_index_array(
    domain: common.Domain,
    dim: common.Dimension,
    ops: array_ops.ArrayOps,
    *,
    like: core_defs.NDArrayObject,
    dtype: Optional[type] = None,
) -> core_defs.NDArrayObject:
    """Index array selecting `dim` unchanged over `domain`, in the field's index space, on the device of `like`."""
    xp = ops.xp
    d_idx = domain.dim_index(dim, allow_missing=False)
    unit_range = domain[d_idx].unit_range
    indices = xp.arange(
        unit_range.start,
        unit_range.stop,
        dtype=None if dtype is None else ops.dtype(dtype),
        device=ops.device(like),
    )
    shape = tuple(len(unit_range) if i == d_idx else 1 for i in range(len(domain)))
    return xp.broadcast_to(xp.reshape(indices, shape), domain.shape)


def _hyperslice(
    index_array: core_defs.NDArrayObject,
    image_range: common.UnitRange,
    xp: ModuleType,
    skip_value: Optional[core_defs.IntegralScalar] = None,
) -> Optional[tuple[slice, ...]]:
    """
    Return the hypercube slice that contains all indices in `index_array` that are within `image_range`, or `None` if no such hypercube exists.

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

    if not xp.any(select_mask):
        return None

    nnz: tuple[core_defs.NDArrayObject, ...] = xp.nonzero(select_mask)

    slices = tuple(
        slice(int(xp.min(dim_nnz_indices)), int(xp.max(dim_nnz_indices)) + 1)
        for dim_nnz_indices in nnz
    )
    hcube = select_mask[tuple(slices)]
    if skip_value is not None:
        ignore_mask = index_array == skip_value
        hcube = hcube | ignore_mask[tuple(slices)]
    if not xp.all(hcube):
        return None

    return slices


# -- Specialized implementations for builtin operations on array fields --


_builtin_implementations: dict[fbuiltins.BuiltInFunction, Callable[..., Any]] = {}


def _register_builtin(builtin: fbuiltins.BuiltInFunction, impl: Callable[..., Any]) -> None:
    """Register `impl` for `builtin`, unless the `ArrayOps` of the first field argument replaces it."""

    def dispatch(*args: Any, **kwargs: Any) -> Any:
        return _ops_of(*args).builtin_overrides.get(builtin, impl)(*args, **kwargs)

    dispatch.__name__ = getattr(impl, "__name__", builtin.name)
    assert builtin not in _builtin_implementations
    _builtin_implementations[builtin] = dispatch


_register_builtin(
    fbuiltins.abs,
    NdArrayField.__abs__,
)
_register_builtin(
    fbuiltins.power,
    NdArrayField.__pow__,
)

for name in (
    fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES
    + fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES
    + fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES
):
    if name in ["abs", "power"]:
        continue
    _register_builtin(getattr(fbuiltins, name), _make_builtin(name, name))

_register_builtin(
    fbuiltins.minimum,
    _make_builtin("minimum", "minimum"),
)
_register_builtin(
    fbuiltins.maximum,
    _make_builtin("maximum", "maximum"),
)
_register_builtin(
    fbuiltins.fmod,
    _make_builtin("fmod", "fmod"),
)
_register_builtin(fbuiltins.where, _make_builtin("where", "where"))


def _to_field(value: common.Field | core_defs.Scalar, like: NdArrayField) -> common.Field:
    # TODO(havogt): this function is only to workaround broadcasting of scalars, once we have a ConstantField, we can broadcast to that directly
    return (
        value
        if isinstance(value, common.Field)
        else NdArrayField.from_array(
            array_ops.of(like._ndarray).asarray(value, like=like.ndarray), domain=common.Domain()
        )
    )


def _intersect_fields(
    *fields: common.Field | core_defs.Scalar,
    ignore_dims: Optional[common.Dimension | tuple[common.Dimension, ...]] = None,
) -> tuple[common.Field, ...]:
    # TODO(havogt): this function could be moved to common, but then requires a broadcast implementation for all field implementations;
    # currently blocked, because requiring the `_to_field` function, see comment there.
    like = next(f for f in fields if isinstance(f, NdArrayField))
    promoted_dims = common.promote_dims(
        *(f.domain.dims for f in fields if isinstance(f, common.Field))
    )
    broadcasted_fields = [_broadcast(_to_field(f, like), promoted_dims) for f in fields]

    intersected_domains = embedded_common.restrict_to_intersection(
        *[f.domain for f in broadcasted_fields], ignore_dims=ignore_dims
    )

    ops = array_ops.of(like._ndarray)
    return tuple(
        NdArrayField.from_array(
            ops.asarray(f.ndarray[_get_slices_from_domain_slice(f.domain, intersected_domain)]),
            domain=intersected_domain,
        )
        for f, intersected_domain in zip(broadcasted_fields, intersected_domains, strict=True)
    )


def _stack_domains(*domains: common.Domain, dim: common.Dimension) -> common.Domain:
    if not domains:
        return common.Domain()
    dim_start = domains[0][dim].unit_range.start
    dim_stop = domains[-1][dim].unit_range.stop
    return domains[0].replace(dim, common.NamedRange(dim, common.UnitRange(dim_start, dim_stop)))


def _concat(*fields: common.Field, dim: common.Dimension) -> common.Field:
    # TODO(havogt): this function could be extended to a general concat
    # currently only concatenate along the given dimension
    sorted_fields = sorted(fields, key=lambda f: f.domain[dim].unit_range.start)

    for prev, curr in itertools.pairwise(sorted_fields):
        left = prev.domain[dim].unit_range.stop
        right = curr.domain[dim].unit_range.start
        if left > right:
            raise ValueError("Fields to concatenate must not overlap.")
        if left < right:
            raise embedded_exceptions.NonContiguousDomain(f"Cannot concatenate fields along {dim}.")
    new_domain = _stack_domains(*[f.domain for f in sorted_fields], dim=dim)
    xp = _ops_of(*sorted_fields).xp
    return NdArrayField.from_array(
        xp.concat(
            [xp.broadcast_to(f.ndarray, f.domain.shape) for f in sorted_fields],
            axis=new_domain.dim_index(dim, allow_missing=False),
        ),
        domain=new_domain,
    )


def _invert_domain(domain: common.Domain) -> tuple[common.Domain, ...]:
    assert domain.ndim == 1
    dim = domain.dims[0]
    rng = domain.ranges[0]

    if rng.is_empty():
        return (common.Domain(dims=(dim,), ranges=(common.UnitRange.infinite(),)),)

    result = []
    if rng.start is not common.Infinity.NEGATIVE:
        result.append(
            common.Domain(
                dims=(dim,), ranges=(common.UnitRange(common.Infinity.NEGATIVE, rng.start),)
            )
        )
    if rng.stop is not common.Infinity.POSITIVE:
        result.append(
            common.Domain(
                dims=(dim,), ranges=(common.UnitRange(rng.stop, common.Infinity.POSITIVE),)
            )
        )
    return tuple(result)


def _size0_field(
    like: NdArrayField, dims: tuple[common.Dimension, ...], dtype: core_defs.DType
) -> NdArrayField:
    ops = array_ops.of(like._ndarray)
    return NdArrayField.from_array(
        ops.xp.empty(
            (0,) * len(dims), dtype=ops.dtype(dtype.scalar_type), device=ops.device(like.ndarray)
        ),
        domain=common.Domain(dims=dims, ranges=(common.UnitRange(0, 0),) * len(dims)),
    )


def _concat_where(
    domain: common.Domain,
    true_field: common.Field,
    false_field: common.Field,
) -> common.Field:
    if domain.ndim != 1:
        raise NotImplementedError(
            "'concat_where': Can only concatenate fields with a 1-dimensional domain."
        )
    domain_dim = domain.dims[0]

    # intersect the field in dimensions orthogonal to the domain, then all slices in the domain field have same domain
    t_broadcasted, f_broadcasted = _intersect_fields(
        true_field, false_field, ignore_dims=domain_dim
    )

    true_domain = embedded_common.domain_intersection(t_broadcasted.domain, domain)
    t_slices = () if true_domain.is_empty() else (t_broadcasted[true_domain],)

    inverted_domains = _invert_domain(domain)
    false_domains = tuple(
        intersection
        for d in inverted_domains
        if not (
            intersection := embedded_common.domain_intersection(f_broadcasted.domain, d)
        ).is_empty()
    )
    f_slices = tuple(f_broadcasted[d] for d in false_domains)

    if len(t_slices) + len(f_slices) == 0:
        # no data to concatenate, return an empty field
        like = next(f for f in (true_field, false_field) if isinstance(f, NdArrayField))
        return _size0_field(like, dims=t_broadcasted.domain.dims, dtype=true_field.dtype)
    return _concat(*f_slices, *t_slices, dim=domain_dim)


_register_builtin(experimental.concat_where, _concat_where)


def _as_offset(offset: fbuiltins.FieldOffset, offset_field: NdArrayField) -> common.Connectivity:
    if not fbuiltins.is_cartesian_offset(offset):
        target_dims = ", ".join(d.value for d in offset.target)
        raise ValueError(
            f"'as_offset' is only supported for Cartesian offsets "
            f"(single target dimension equal to source dimension); "
            f"got source '{offset.source.value}' and target ({target_dims})."
        )
    source_dim = offset.source
    coords = _identity_index_array(
        offset_field.domain,
        source_dim,
        array_ops.of(offset_field._ndarray),
        like=offset_field.ndarray,
        dtype=fbuiltins.IndexType,
    )
    return common._connectivity(
        offset_field.ndarray + coords, codomain=source_dim, domain=offset_field.domain
    )


_register_builtin(experimental.as_offset, _as_offset)


def _make_reduction(
    builtin_name: str, array_builtin_name: str, initial_value_op: Callable
) -> Callable[..., NdArrayField[common.DimsT, core_defs.ScalarT]]:
    def _builtin_op(
        field: NdArrayField[common.DimsT, core_defs.ScalarT], axis: common.Dimension
    ) -> NdArrayField[common.DimsT, core_defs.ScalarT]:
        ops = array_ops.of(field._ndarray)

        if not axis.kind == common.DimensionKind.LOCAL:
            raise ValueError("Can only reduce local dimensions.")
        if axis not in field.domain.dims:
            raise ValueError(f"Field can not be reduced as it doesn't have dimension '{axis}'.")
        if len([d for d in field.domain.dims if d.kind is common.DimensionKind.LOCAL]) > 1:
            raise NotImplementedError(
                "Reducing a field with more than one local dimension is not supported."
            )
        reduce_dim_index = field.domain.dims.index(axis)
        current_offset_provider = embedded_context.get_offset_provider(None)
        assert current_offset_provider is not None
        offset_definition = common.get_offset(
            current_offset_provider, axis.value
        )  # assumes offset and local dimension have same name
        assert common.is_neighbor_table(offset_definition)
        new_domain = common.Domain(*[nr for nr in field.domain if nr.dim != axis])

        broadcast_slice = tuple(
            slice(None) if d in [axis, offset_definition.domain.dims[0]] else None
            for d in field.domain.dims
        )
        masked_array = ops.where(
            ops.asarray(offset_definition.ndarray[broadcast_slice]) != common._DEFAULT_SKIP_VALUE,
            field.ndarray,
            initial_value_op(field),
        )

        return NdArrayField.from_array(
            getattr(ops.xp, array_builtin_name)(masked_array, axis=reduce_dim_index),
            domain=new_domain,
        )

    _builtin_op.__name__ = builtin_name
    return _builtin_op


_register_builtin(
    fbuiltins.neighbor_sum, _make_reduction("neighbor_sum", "sum", lambda x: x.dtype.scalar_type(0))
)
_register_builtin(
    fbuiltins.max_over,
    _make_reduction("max_over", "max", lambda x: array_ops.of(x._ndarray).xp.min(x._ndarray)),
)
_register_builtin(
    fbuiltins.min_over,
    _make_reduction("min_over", "min", lambda x: array_ops.of(x._ndarray).xp.max(x._ndarray)),
)


def _field_from_array(
    data: Any,
    /,
    *,
    domain: Optional[common.DomainLike] = None,
    dtype: Optional[core_defs.DType] = None,
) -> NdArrayField:
    if not array_api_compat.is_array_api_obj(data):
        raise NotImplementedError(f"Cannot construct a 'Field' from '{type(data)}'.")
    assert domain is not None
    return NdArrayField.from_array(data, domain=domain, dtype=dtype)


def _connectivity_from_array(
    data: Any,
    /,
    codomain: common.Dimension,
    *,
    domain: Optional[common.DomainLike] = None,
    dtype: Optional[core_defs.DType] = None,
    skip_value: Optional[core_defs.IntegralScalar] = None,
) -> NdArrayConnectivityField:
    if not array_api_compat.is_array_api_obj(data):
        raise NotImplementedError(f"Cannot construct a 'Connectivity' from '{type(data)}'.")
    assert domain is not None
    return NdArrayConnectivityField.from_array(
        data, codomain, domain=domain, dtype=dtype, skip_value=skip_value
    )


common._field.register(object, _field_from_array)
common._connectivity.register(object, _connectivity_from_array)


def _broadcast(field: common.Field, new_dimensions: Sequence[common.Dimension]) -> common.Field:
    if field.domain.dims == new_dimensions:
        return field
    domain_slice: list[slice | None] = []
    named_ranges = []
    for dim in new_dimensions:
        if (pos := embedded_common._find_index_of_dim(dim, field.domain)) is not None:
            domain_slice.append(slice(None))
            named_ranges.append(common.NamedRange(dim, field.domain[pos].unit_range))
        else:
            domain_slice.append(None)
            named_ranges.append(common.NamedRange(dim, common.UnitRange.infinite()))
    return common._field(field.ndarray[tuple(domain_slice)], domain=common.Domain(*named_ranges))


def _builtins_broadcast(
    field: common.Field | core_defs.Scalar, new_dimensions: tuple[common.Dimension, ...]
) -> common.Field:  # separated for typing reasons
    if isinstance(field, common.Field):
        return _broadcast(field, new_dimensions)
    raise AssertionError("Scalar case not reachable from 'fbuiltins.broadcast'.")


_register_builtin(fbuiltins.broadcast, _builtins_broadcast)


def _astype(field: common.Field | core_defs.ScalarT | tuple, type_: type) -> NdArrayField:
    if isinstance(field, NdArrayField):
        ops = array_ops.of(field._ndarray)
        return NdArrayField.from_array(
            ops.xp.astype(field.ndarray, ops.dtype(type_)), domain=field.domain
        )
    raise AssertionError("This is the NdArrayField implementation of 'fbuiltins.astype'.")


_register_builtin(fbuiltins.astype, _astype)  # type: ignore[arg-type]  # because fbuiltins.astype is overloaded


def _get_slices_from_domain_slice(
    domain: common.Domain,
    domain_slice: common.Domain | Sequence[common.NamedRange | common.NamedIndex],
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
            _, index_or_range = domain_slice[pos]
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
