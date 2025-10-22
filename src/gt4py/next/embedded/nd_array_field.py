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
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

import numpy as np
from numpy import typing as npt

from gt4py._core import definitions as core_defs
from gt4py.eve.extended_typing import (
    ClassVar,
    Iterable,
    Never,
    Optional,
    ParamSpec,
    TypeAlias,
    TypeVar,
    cast,
)
from gt4py.next import common
from gt4py.next.embedded import (
    common as embedded_common,
    context as embedded_context,
    exceptions as embedded_exceptions,
)
from gt4py.next.ffront import experimental, fbuiltins


try:
    import cupy as cp
except ImportError:
    cp: Optional[ModuleType] = None  # type: ignore[no-redef]

try:
    from jax import numpy as jnp
except ImportError:
    jnp: Optional[ModuleType] = None  # type: ignore[no-redef]

try:
    import dace
except ImportError:
    dace: Optional[ModuleType] = None  # type: ignore[no-redef]


def _get_nd_array_class(*fields: common.Field | core_defs.Scalar) -> type[NdArrayField]:
    for f in fields:
        if isinstance(f, NdArrayField):
            return f.__class__
    raise AssertionError("No 'NdArrayField' found in the arguments.")


def _make_builtin(
    builtin_name: str, array_builtin_name: str, reverse: bool = False
) -> Callable[..., NdArrayField]:
    def _builtin_op(*fields: common.Field | core_defs.Scalar) -> NdArrayField:
        cls_ = _get_nd_array_class(*fields)
        xp = cls_.array_ns
        op = getattr(xp, array_builtin_name)

        domain_intersection = embedded_common.domain_intersection(
            *[f.domain for f in fields if isinstance(f, common.Field)]
        )

        transformed: list[core_defs.NDArrayObject | core_defs.Scalar] = []
        for f in fields:
            if isinstance(f, common.Field):
                if f.domain == domain_intersection:
                    transformed.append(xp.asarray(f.ndarray))
                else:
                    f_broadcasted = _broadcast(f, domain_intersection.dims)
                    f_slices = _get_slices_from_domain_slice(
                        f_broadcasted.domain, domain_intersection
                    )
                    transformed.append(xp.asarray(f_broadcasted.ndarray[f_slices]))
            else:
                assert core_defs.is_scalar_type(f)
                transformed.append(f)
        if reverse:
            transformed.reverse()
        new_data = op(*transformed)
        return cls_.from_array(new_data, domain=domain_intersection)

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

    array_ns: ClassVar[ModuleType]  # TODO(havogt): introduce a NDArrayNamespace protocol
    array_byte_bounds: ClassVar[  # TODO(egparedes): make this part of the previous protocol
        Callable[[npt.NDArray], tuple[int, int]]
        | Callable[[core_defs.NDArrayObject], tuple[int, int]]
    ]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "array_ns") and not hasattr(cls, "array_byte_bounds"):
            # This is needed to initialize `array_byte_bounds` only once per subclass
            try:
                cls.array_byte_bounds = staticmethod(
                    getattr(cls.array_ns, "byte_bounds", None)
                    or cls.array_ns.lib.array_utils.byte_bounds
                )
            except AttributeError:
                pass

    @property
    def domain(self) -> common.Domain:
        return self._domain

    @property
    def shape(self) -> tuple[int, ...]:
        return self._ndarray.shape

    @functools.cached_property
    def __gt_origin__(self) -> tuple[int, ...]:
        assert common.Domain.is_finite(self._domain)
        return tuple(-r.start for r in self._domain.ranges)

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
                f"'as_scalar' is only valid on 0-dimensional 'Field's, got a {self.domain.ndim}-dimensional 'Field'."
            )
        # note: `.item()` will return a Python type, therefore we use indexing with an empty tuple
        return self.asnumpy()[()]  # type: ignore[return-value] # should be ensured by the 0-d check

    @property
    def codomain(self) -> type[core_defs.ScalarT]:
        return self.dtype.scalar_type

    @functools.cached_property
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
        domain translation (i.e. only transform the domain without altering the data).

        A table showing the relation between the connectivity kind and the supported cases
        is shown in :class:`common.ConnectivityKind`.

        Args:
            *connectivities: connectivities to be used for the `premap` operation. If only one
                connectivity is passed, it will be expanded to fully defined connectivities for
                each dimension in the domain of the field according to the rules described above.
                If more than one connectivity is passed, they all must satisfy:
                - be of the same kind or encode only compact domain transformations
                - the codomain of each connectivity must be different
                - for reshuffling operations, all connectivities must have the same domain
                (Note that remapping operations only support a single connectivity argument.)

        """  # noqa: RUF002  # TODO(egparedes): move docstring to the `premap` builtin function when it exists

        conn_fields: list[common.Connectivity] = []
        codomains_counter: collections.Counter[common.Dimension] = collections.Counter()

        for connectivity in connectivities:
            # For neighbor reductions, a FieldOffset is passed instead of an actual Connectivity
            if not isinstance(connectivity, common.Connectivity):
                assert isinstance(connectivity, fbuiltins.FieldOffset)
                connectivity = connectivity.as_connectivity_field()
            assert isinstance(connectivity, common.Connectivity)

            # Current implementation relies on skip_value == -1:
            # if we assume the indexed array has at least one element,
            # we wrap around without out of bounds access
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

        if any(c.kind & common.ConnectivityKind.ALTER_STRUCT for c in conn_fields) and any(
            (~c.kind & common.ConnectivityKind.ALTER_STRUCT) for c in conn_fields
        ):
            raise ValueError(
                "Mixing connectivities that change the data structure with connectivities that do not is not allowed."
            )

        # Select actual implementation of the transformation
        if not (conn_fields[0].kind & common.ConnectivityKind.ALTER_STRUCT):
            return _domain_premap(self, *conn_fields)

        if any(c.kind & common.ConnectivityKind.ALTER_DIMS for c in conn_fields) and any(
            (~c.kind & common.ConnectivityKind.ALTER_DIMS) for c in conn_fields
        ):
            raise ValueError(
                "Mixing connectivities that change the dimensions in the domain with connectivities that do not is not allowed."
            )

        if not (conn_fields[0].kind & common.ConnectivityKind.ALTER_DIMS):
            assert all(isinstance(c, NdArrayConnectivityField) for c in conn_fields)
            return _reshuffling_premap(self, *cast(list[NdArrayConnectivityField], conn_fields))

        assert len(conn_fields) == 1
        return _remapping_premap(self, conn_fields[0])

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

        if isinstance(value, common.Field):
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
        self, index: common.AnyIndexSpec
    ) -> tuple[common.Domain, common.RelativeIndexSequence]:
        index = embedded_common.canonicalize_any_index_sequence(index)
        new_domain = embedded_common.sub_domain(self.domain, index)

        index_sequence = common.as_any_index_sequence(index)
        slice_ = (
            _get_slices_from_domain_slice(self.domain, index_sequence)
            if common.is_absolute_index_sequence(index_sequence)
            else index_sequence
        )
        assert common.is_relative_index_sequence(slice_)
        return new_domain, slice_

    @property
    def _data_buffer_ptr_(self) -> int:
        """
        Returns the pointer of the underlying data buffer.

        Subclasses should redefine this property with the fastest possible
        implementation.

        This is mainly needed to interface with high-performance code generated by
        gt4py backends. Therefore, this is considered for now an internal attribute
        and it should not be accessed directly by users.
        """
        return self.array_byte_bounds(self._ndarray)[0]  # type: ignore[call-arg,misc] # array_byte_bounds is a staticmethod

    if dace:

        def _dace_data_ptr(self) -> int:
            return self._data_buffer_ptr_

        def _dace_descriptor(self) -> dace.data.Data:
            return dace.data.create_datadescriptor(self.ndarray)

    else:

        def _dace_data_ptr(self) -> int:
            raise NotImplementedError(
                "data_ptr is only supported when the 'dace' module is available."
            )

        def _dace_descriptor(self) -> dace.data.Data:
            raise NotImplementedError(
                "__descriptor__ is only supported when the 'dace' module is available."
            )

    data_ptr = _dace_data_ptr
    """
    Returns the pointer of the underlying data buffer.

    Fully equivalent to `self._data_buffer_ptr_`. It is only defined to emulate the
    PyTorch API for DaCe interoperability.

    Note:
        This method is experimental and will be likely removed in future versions.
    """

    __descriptor__ = _dace_descriptor
    """Extension of NdArrayField adding SDFGConvertible support in GT4Py Programs."""


@dataclasses.dataclass(frozen=True)
class NdArrayConnectivityField(
    common.Connectivity[common.DimsT, common.DimT],
    NdArrayField[common.DimsT, core_defs.IntegralScalar],
):
    _codomain: common.DimT
    _skip_value: Optional[core_defs.IntegralScalar]
    _kind: Optional[common.ConnectivityKind] = None

    def __post_init__(self) -> None:
        assert self._kind is None or bool(self._kind & common.ConnectivityKind.ALTER_DIMS) == (
            self.domain.dim_index(self.codomain) is not None
        )

    @functools.cached_property
    def _cache(self) -> dict:
        return {}

    @classmethod
    def __gt_builtin_func__(cls, _: fbuiltins.BuiltInFunction) -> Never:  # type: ignore[override]
        raise NotImplementedError()

    @property
    def codomain(self) -> common.DimT:  # type: ignore[override] # TODO(havogt): instead of inheriting from NdArrayField, steal implementation or common base
        return self._codomain

    @property
    def skip_value(self) -> Optional[core_defs.IntegralScalar]:
        return self._skip_value

    @property
    def kind(self) -> common.ConnectivityKind:
        if self._kind is None:
            object.__setattr__(
                self,
                "_kind",
                common.ConnectivityKind.ALTER_STRUCT
                | (
                    common.ConnectivityKind.ALTER_DIMS
                    if self.domain.dim_index(self.codomain) is None
                    else common.ConnectivityKind(0)
                ),
            )
            assert self._kind is not None

        return self._kind

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

        return cls(domain, array, codomain, _skip_value=skip_value)

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

            xp = self.array_ns
            slices = _hyperslice(self._ndarray, image_range, xp, self.skip_value)
            if slices is None:
                raise ValueError("Restriction generates non-contiguous dimensions.")

            new_domain = self.domain.slice_at[slices]
            self._cache[cache_key] = new_domain

        return new_domain

    def restrict(self, index: common.AnyIndexSpec) -> NdArrayConnectivityField:
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


def _reshuffling_premap(
    data: NdArrayField, *connectivities: NdArrayConnectivityField
) -> NdArrayField:
    # Check that all connectivities have the same domain
    assert len(connectivities) == 1 or all(
        c.domain == connectivities[0].domain for c in connectivities[1:]
    )

    connectivity = connectivities[0]
    xp = data.array_ns

    # Reorder and complete connectivity dimensions to match the field domain
    # It should be enough to check this only the first connectivity
    # since all connectivities must have the same domain
    transposed_axes = []
    expanded_axes: list[int] = []
    transpose_needed = False
    for new_dim_idx, dim in enumerate(data.domain.dims):
        if (dim_idx := connectivity.domain.dim_index(dim)) is None:
            expanded_axes.append(connectivity.domain.ndim + len(expanded_axes))
            dim_idx = expanded_axes[-1]
        transposed_axes.append(dim_idx)
        transpose_needed = transpose_needed | (dim_idx != new_dim_idx)

    # Broadcast connectivity arrays to match the full domain
    conn_map = {}
    new_ranges = data.domain.ranges
    for conn in connectivities:
        conn_ndarray = conn.ndarray
        if expanded_axes:
            conn_ndarray = xp.expand_dims(conn_ndarray, axis=expanded_axes)
        if transpose_needed:
            conn_ndarray = xp.transpose(conn_ndarray, transposed_axes)
        if conn_ndarray.shape != data.domain.shape:
            conn_ndarray = xp.broadcast_to(conn_ndarray, data.domain.shape)
        if conn_ndarray is not conn.ndarray:
            conn = conn.__class__.from_array(
                conn_ndarray, domain=data.domain, codomain=conn.codomain
            )
        conn_map[conn.codomain] = conn
        dim_idx = data.domain.dim_index(conn.codomain, allow_missing=False)
        current_range: common.UnitRange = data.domain.ranges[dim_idx]
        new_conn_ranges = connectivity.inverse_image(current_range).ranges
        new_ranges = tuple(r & s for r, s in zip(new_ranges, new_conn_ranges))

    conns_dims = [c.domain.dims for c in conn_map.values()]
    for i in range(len(conns_dims) - 1):
        if conns_dims[i] != conns_dims[i + 1]:
            raise ValueError(
                f"All premapping connectivities must have the same dimensions, got: {conns_dims}."
            )

    new_domain = common.Domain(dims=data.domain.dims, ranges=new_ranges)

    # Create identity connectivities for the missing domain dimensions
    for dim in data.domain.dims:
        if dim not in conn_map:
            conn_map[dim] = _identity_connectivity(new_domain, dim, cls=type(connectivity))

    # Take data
    take_indices = tuple(
        conn_map[dim].ndarray - data.domain[dim].unit_range.start  # shift to 0-based indexing
        for dim in data.domain.dims
    )
    new_buffer = data._ndarray.__getitem__(take_indices)

    return data.__class__.from_array(
        new_buffer,
        domain=new_domain,
        dtype=data.dtype,
    )


def _remapping_premap(data: NdArrayField, connectivity: common.Connectivity) -> NdArrayField:
    new_dims = {*connectivity.domain.dims} - {connectivity.codomain}
    if repeated_dims := (new_dims & {*data.domain.dims}):
        raise ValueError(f"Remapped field will contain repeated dimensions '{repeated_dims}'.")

    # Compute the new domain
    dim = connectivity.codomain
    dim_idx = data.domain.dim_index(dim)
    if dim_idx is None:
        raise ValueError(f"Incompatible index field, expected a field with dimension '{dim}'.")

    current_range: common.UnitRange = data.domain[dim_idx][1]
    new_ranges = connectivity.inverse_image(current_range)
    new_domain = data.domain.replace(dim_idx, *new_ranges)

    # Perform premap:
    xp = data.array_ns

    # 1- first restrict the connectivity to the new domain
    restricted_connectivity_domain = common.Domain(*new_ranges)
    restricted_connectivity = (
        connectivity.restrict(restricted_connectivity_domain)
        if restricted_connectivity_domain != connectivity.domain
        else connectivity
    )
    assert isinstance(restricted_connectivity, common.Connectivity)

    # 2- then compute the index array
    new_idx_array = xp.asarray(restricted_connectivity.ndarray) - current_range.start

    # 3- finally, take the new array
    new_buffer = xp.take(data._ndarray, new_idx_array, axis=dim_idx)

    return data.__class__.from_array(
        new_buffer,
        domain=new_domain,
        dtype=data.dtype,
    )


_NdConnT = TypeVar("_NdConnT", bound=NdArrayConnectivityField)


def _identity_connectivity(
    domain: common.Domain, codomain: common.DimT, *, cls: type[_NdConnT]
) -> _NdConnT:
    assert codomain in domain.dims
    xp = cls.array_ns
    shape = domain.shape
    d_idx = domain.dim_index(codomain, allow_missing=False)
    indices = xp.arange(domain[d_idx].unit_range.start, domain[d_idx].unit_range.stop)
    result = cls.from_array(
        xp.broadcast_to(
            indices[
                tuple(slice(None) if i == d_idx else None for i, dim in enumerate(domain.dims))
            ],
            shape,
        ),
        codomain=codomain,
        domain=domain,
        dtype=int,
    )

    return cast(_NdConnT, result)


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

    nnz: tuple[core_defs.NDArrayObject, ...] = xp.nonzero(select_mask)

    slices = tuple(
        slice(xp.min(dim_nnz_indices).item(), xp.max(dim_nnz_indices).item() + 1)
        for dim_nnz_indices in nnz
    )
    hcube = select_mask[tuple(slices)]
    if skip_value is not None:
        ignore_mask = index_array == skip_value
        hcube |= ignore_mask[tuple(slices)]
    if not xp.all(hcube):
        return None

    return slices


# -- Specialized implementations for builtin operations on array fields --

NdArrayField.register_builtin_func(
    fbuiltins.abs,  # type: ignore[attr-defined]
    NdArrayField.__abs__,
)
NdArrayField.register_builtin_func(
    fbuiltins.power,  # type: ignore[attr-defined]
    NdArrayField.__pow__,
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
    fbuiltins.minimum,  # type: ignore[attr-defined]
    _make_builtin("minimum", "minimum"),
)
NdArrayField.register_builtin_func(
    fbuiltins.maximum,  # type: ignore[attr-defined]
    _make_builtin("maximum", "maximum"),
)
NdArrayField.register_builtin_func(
    fbuiltins.fmod,  # type: ignore[attr-defined]
    _make_builtin("fmod", "fmod"),
)
NdArrayField.register_builtin_func(fbuiltins.where, _make_builtin("where", "where"))


def _compute_mask_slices(
    mask: core_defs.NDArrayObject,
) -> list[tuple[bool, slice]]:
    """Take a 1-dimensional mask and return a sequence of mappings from boolean values to slices."""
    # TODO: does it make sense to upgrade this naive algorithm to numpy?
    assert mask.ndim == 1
    cur = bool(mask[0].item())
    ind = 0
    res = []
    for i in range(1, mask.shape[0]):
        # Use `.item()` to extract the scalar from a 0-d array in case of e.g. cupy
        if (mask_i := bool(mask[i].item())) != cur:
            res.append((cur, slice(ind, i)))
            cur = mask_i
            ind = i
    res.append((cur, slice(ind, mask.shape[0])))
    return res


def _trim_empty_domains(
    lst: Iterable[tuple[bool, common.Domain]],
) -> list[tuple[bool, common.Domain]]:
    """Remove empty domains from beginning and end of the list."""
    lst = list(lst)
    if not lst:
        return lst
    if lst[0][1].is_empty():
        return _trim_empty_domains(lst[1:])
    if lst[-1][1].is_empty():
        return _trim_empty_domains(lst[:-1])
    return lst


def _to_field(
    value: common.Field | core_defs.Scalar, nd_array_field_type: type[NdArrayField]
) -> common.Field:
    # TODO(havogt): this function is only to workaround broadcasting of scalars, once we have a ConstantField, we can broadcast to that directly
    return (
        value
        if isinstance(value, common.Field)
        else nd_array_field_type.from_array(
            nd_array_field_type.array_ns.asarray(value), domain=common.Domain()
        )
    )


def _intersect_fields(
    *fields: common.Field | core_defs.Scalar,
    ignore_dims: Optional[common.Dimension | tuple[common.Dimension, ...]] = None,
) -> tuple[common.Field, ...]:
    # TODO(havogt): this function could be moved to common, but then requires a broadcast implementation for all field implementations;
    # currently blocked, because requiring the `_to_field` function, see comment there.
    nd_array_class = _get_nd_array_class(*fields)
    promoted_dims = common.promote_dims(
        *(f.domain.dims for f in fields if isinstance(f, common.Field))
    )
    broadcasted_fields = [_broadcast(_to_field(f, nd_array_class), promoted_dims) for f in fields]

    intersected_domains = embedded_common.restrict_to_intersection(
        *[f.domain for f in broadcasted_fields], ignore_dims=ignore_dims
    )

    return tuple(
        nd_array_class.from_array(
            f.ndarray[_get_slices_from_domain_slice(f.domain, intersected_domain)],
            domain=intersected_domain,
        )
        for f, intersected_domain in zip(broadcasted_fields, intersected_domains, strict=True)
    )


def _stack_domains(*domains: common.Domain, dim: common.Dimension) -> Optional[common.Domain]:
    if not domains:
        return common.Domain()
    dim_start = domains[0][dim].unit_range.start
    dim_stop = dim_start
    for domain in domains:
        if not domain[dim].unit_range.start == dim_stop:
            return None
        else:
            dim_stop = domain[dim].unit_range.stop
    return domains[0].replace(dim, common.NamedRange(dim, common.UnitRange(dim_start, dim_stop)))


def _concat(*fields: common.Field, dim: common.Dimension) -> common.Field:
    # TODO(havogt): this function could be extended to a general concat
    # currently only concatenate along the given dimension and requires the fields to be ordered

    if (
        len(fields) > 1
        and not embedded_common.domain_intersection(*[f.domain for f in fields]).is_empty()
    ):
        raise ValueError("Fields to concatenate must not overlap.")
    new_domain = _stack_domains(*[f.domain for f in fields], dim=dim)
    if new_domain is None:
        raise embedded_exceptions.NonContiguousDomain(f"Cannot concatenate fields along {dim}.")
    nd_array_class = _get_nd_array_class(*fields)
    return nd_array_class.from_array(
        nd_array_class.array_ns.concatenate(
            [nd_array_class.array_ns.broadcast_to(f.ndarray, f.domain.shape) for f in fields],
            axis=new_domain.dim_index(dim, allow_missing=False),
        ),
        domain=new_domain,
    )


def _concat_where(
    mask_field: common.Field, true_field: common.Field, false_field: common.Field
) -> common.Field:
    cls_ = _get_nd_array_class(mask_field, true_field, false_field)
    xp = cls_.array_ns
    if mask_field.domain.ndim != 1:
        raise NotImplementedError(
            "'concat_where': Can only concatenate fields with a 1-dimensional mask."
        )
    mask_dim = mask_field.domain.dims[0]

    # intersect the field in dimensions orthogonal to the mask, then all slices in the mask field have same domain
    t_broadcasted, f_broadcasted = _intersect_fields(true_field, false_field, ignore_dims=mask_dim)

    # TODO(havogt): for clarity, most of it could be implemented on named_range in the masked dimension, but we currently lack the utils
    # compute the consecutive ranges (first relative, then domain) of true and false values
    mask_values_to_slices_mapping: Iterable[tuple[bool, slice]] = _compute_mask_slices(
        mask_field.ndarray
    )
    mask_values_to_domain_mapping: Iterable[tuple[bool, common.Domain]] = (
        (mask, mask_field.domain.slice_at[domain_slice])
        for mask, domain_slice in mask_values_to_slices_mapping
    )
    # mask domains intersected with the respective fields
    mask_values_to_intersected_domains_mapping: Iterable[tuple[bool, common.Domain]] = (
        (
            mask_value,
            embedded_common.domain_intersection(
                t_broadcasted.domain if mask_value else f_broadcasted.domain, mask_domain
            ),
        )
        for mask_value, mask_domain in mask_values_to_domain_mapping
    )

    # remove the empty domains from the beginning and end
    mask_values_to_intersected_domains_mapping = _trim_empty_domains(
        mask_values_to_intersected_domains_mapping
    )
    if any(d.is_empty() for _, d in mask_values_to_intersected_domains_mapping):
        raise embedded_exceptions.NonContiguousDomain(
            f"In 'concat_where', cannot concatenate the following 'Domain's: {[d for _, d in mask_values_to_intersected_domains_mapping]}."
        )

    # slice the fields with the domain ranges
    transformed = [
        t_broadcasted[d] if v else f_broadcasted[d]
        for v, d in mask_values_to_intersected_domains_mapping
    ]

    # stack the fields together
    if transformed:
        return _concat(*transformed, dim=mask_dim)
    else:
        result_domain = common.Domain(common.NamedRange(mask_dim, common.UnitRange(0, 0)))
        result_array = xp.empty(result_domain.shape)
    return cls_.from_array(result_array, domain=result_domain)


NdArrayField.register_builtin_func(experimental.concat_where, _concat_where)  # type: ignore[arg-type] # TODO(havogt): this is still the "old" concat_where, needs to be replaced in a next PR


def _make_reduction(
    builtin_name: str, array_builtin_name: str, initial_value_op: Callable
) -> Callable[..., NdArrayField[common.DimsT, core_defs.ScalarT]]:
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
        current_offset_provider = embedded_context.get_offset_provider(None)
        assert current_offset_provider is not None
        offset_definition = common.get_offset(
            current_offset_provider, axis.value
        )  # assumes offset and local dimension have same name
        assert common.is_neighbor_table(offset_definition)
        new_domain = common.Domain(*[nr for nr in field.domain if nr.dim != axis])

        broadcast_slice = tuple(
            slice(None) if d in [axis, offset_definition.domain.dims[0]] else xp.newaxis
            for d in field.domain.dims
        )
        masked_array = xp.where(
            xp.asarray(offset_definition.ndarray[broadcast_slice]) != common._DEFAULT_SKIP_VALUE,
            field.ndarray,
            initial_value_op(field),
        )

        return field.__class__.from_array(
            getattr(xp, array_builtin_name)(masked_array, axis=reduce_dim_index), domain=new_domain
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

    # It is only possible to cache the data buffer pointer in this way
    # because the backing np.ndarray is never replaced after creation,
    # since this is a frozen dataclass.
    @functools.cached_property
    def _data_buffer_ptr_(self) -> int:
        return self._ndarray.ctypes.data  # type: ignore[attr-defined]  # np.ndarray has `ctypes` attribute


common._field.register(np.ndarray, NumPyArrayField.from_array)


@dataclasses.dataclass(frozen=True, eq=False)
class NumPyArrayConnectivityField(NdArrayConnectivityField):
    array_ns: ClassVar[ModuleType] = np

    _data_buffer_ptr_ = functools.cached_property(NumPyArrayField._data_buffer_ptr_.func)  # type: ignore[attr-defined]  # cached_property has `func` attribute


common._connectivity.register(np.ndarray, NumPyArrayConnectivityField.from_array)

# CuPy
if cp:
    _nd_array_implementations.append(cp)

    # Same as in the NumPy case:
    # It is only possible to cache the data buffer pointer in this way
    # because the backing np.ndarray is never replaced after creation
    # since this is a frozen dataclass.
    @dataclasses.dataclass(frozen=True, eq=False)
    class CuPyArrayField(NdArrayField):
        array_ns: ClassVar[ModuleType] = cp

        @functools.cached_property
        def _data_buffer_ptr_(self) -> int:
            return self._ndarray.data.ptr  # type: ignore[attr-defined]  # cp.ndarray has `data` attribute

    common._field.register(cp.ndarray, CuPyArrayField.from_array)

    @dataclasses.dataclass(frozen=True, eq=False)
    class CuPyArrayConnectivityField(NdArrayConnectivityField):
        array_ns: ClassVar[ModuleType] = cp

        _data_buffer_ptr_ = functools.cached_property(CuPyArrayField._data_buffer_ptr_.func)  # type: ignore[attr-defined]  # cached_property has `func` attribute

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
            domain_slice.append(None)  # np.newaxis
            named_ranges.append(common.NamedRange(dim, common.UnitRange.infinite()))
    return common._field(field.ndarray[tuple(domain_slice)], domain=common.Domain(*named_ranges))


def _builtins_broadcast(
    field: common.Field | core_defs.Scalar, new_dimensions: tuple[common.Dimension, ...]
) -> common.Field:  # separated for typing reasons
    if isinstance(field, common.Field):
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
