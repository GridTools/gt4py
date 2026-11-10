# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`ArrayOps` for JAX arrays, and `NdArrayField` as a JAX pytree."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Optional, TypeAlias, cast

import array_api_compat
import jax
import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import array_ops, nd_array_field
from gt4py.next.ffront import experimental, fbuiltins


class JaxOps(array_ops.ArrayOps):
    def operand(self, value: Any, like: core_defs.NDArrayObject) -> Any:
        return value

    def _elementwise(self, name: str) -> Callable[..., Any]:
        if name == "gamma":
            import jax.scipy.special

            return jax.scipy.special.gamma
        return super()._elementwise(name)

    def setitem(
        self, array: core_defs.NDArrayObject, key: common.RelativeIndexSequence, value: Any
    ) -> core_defs.NDArrayObject:
        return array.at[key].set(value)  # type: ignore[attr-defined] # `NDArrayObject` typing is not complete

    def to_numpy(self, array: core_defs.NDArrayObject) -> np.ndarray:
        return np.asarray(array)

    def buffer_info(self, array: core_defs.NDArrayObject) -> common.BufferInfo:
        raise NotImplementedError("'__gt_buffer_info__' for JAX arrays not yet implemented.")


array_ops.register(array_api_compat.is_jax_namespace, JaxOps)

# TODO(havogt): we currently enable 64-bit support by default, but we might want to make this configurable via the GT4Py config
jax.config.update("jax_enable_x64", True)


def _flatten_field(
    field: nd_array_field.NdArrayField,
) -> tuple[tuple[core_defs.NDArrayObject], common.Domain]:
    return (field.ndarray,), field.domain


def _unflatten_field(
    domain: common.Domain, children: tuple[core_defs.NDArrayObject]
) -> nd_array_field.NdArrayField:
    return nd_array_field.NdArrayField(domain, children[0])  # type: ignore[abstract] # mypy does not see '__gt_builtin_func__' as implemented by 'FieldBuiltinFuncRegistry'


# JAX registers pytrees by type, so fields of every array library are pytrees, not only those
# holding JAX arrays: like a bare NumPy array, a NumPy-backed field passed to `jax.jit` is
# copied to a JAX array on every call.
jax.tree_util.register_pytree_node(
    nd_array_field.NdArrayField,  # type: ignore[type-abstract, unused-ignore] # only reported when 'jax' is installed, see '_unflatten_field'
    _flatten_field,
    _unflatten_field,
)


def _as_offset(offset: fbuiltins.FieldOffset, offset_field: nd_array_field.NdArrayField) -> Any:
    # adding the coordinates would stage the table under a trace even for concrete offsets
    with jax.ensure_compile_time_eval():
        return nd_array_field._as_offset(offset, offset_field)


JaxOps.builtin_overrides = {experimental.as_offset: _as_offset}


# Under `jax.jit` a connectivity is in one of three cases (ADR 0029), each its own class:
# its table is concrete, traced with a concrete twin, or traced without values.


@dataclasses.dataclass(frozen=True, eq=False)
class _ConcreteConnectivity(nd_array_field.NdArrayConnectivityField):
    """A concrete jax table, which stays concrete when used inside a trace."""

    def _inverse_image(self, image_range: common.UnitRange) -> common.Domain:
        with jax.ensure_compile_time_eval():
            return super()._inverse_image(image_range)

    def restrict(self, index: common.AnyIndexSpec) -> _ConcreteConnectivity:
        with jax.ensure_compile_time_eval():
            return cast(_ConcreteConnectivity, super().restrict(index))

    __getitem__ = restrict


@dataclasses.dataclass(frozen=True, eq=False)
class _TwinnedConnectivity(nd_array_field.NdArrayConnectivityField):
    """A traced table whose values are known from a concrete connectivity with the same table."""

    _twin: nd_array_field.NdArrayConnectivityField

    def _inverse_image(self, image_range: common.UnitRange) -> common.Domain:
        return self._twin._inverse_image(image_range)

    def restrict(self, index: common.AnyIndexSpec) -> _TwinnedConnectivity:
        cache_key = (id(self.ndarray), self.domain, index)
        restricted: Optional[_TwinnedConnectivity]
        if (restricted := self._cache.get(cache_key, None)) is None:
            new_domain, buffer_slice = self._slice(index)
            restricted = self._cache[cache_key] = _TwinnedConnectivity(
                new_domain,
                self._ndarray[buffer_slice],
                self.codomain,
                self.skip_value,
                self._twin.restrict(index),
            )
        return restricted

    __getitem__ = restrict


@dataclasses.dataclass(frozen=True, eq=False)
class _ValueFreeConnectivity(nd_array_field.NdArrayConnectivityField):
    """A traced table without concrete values, e.g. from `as_offset` of traced offsets."""

    def _inverse_image(self, image_range: common.UnitRange) -> common.Domain:
        # every target is assumed to lie inside the field; `_gather_index` clamps the others
        return self.domain

    def _gather_index(
        self, index: core_defs.NDArrayObject, image_range: common.UnitRange
    ) -> core_defs.NDArrayObject:
        return jax.numpy.clip(index, 0, len(image_range) - 1)  # type: ignore[arg-type, return-value, unused-ignore] # `NDArrayObject` typing is not complete


def _connectivity_class(
    table: core_defs.NDArrayObject, twin: Optional[nd_array_field.NdArrayConnectivityField]
) -> type[nd_array_field.NdArrayConnectivityField]:
    if not isinstance(table, jax.core.Tracer):
        return _ConcreteConnectivity
    return _ValueFreeConnectivity if twin is None else _TwinnedConnectivity


def _connectivity_from_array(
    data: Any,
    /,
    codomain: common.Dimension,
    *,
    domain: Optional[common.DomainLike] = None,
    dtype: Optional[core_defs.DType] = None,
    skip_value: Optional[core_defs.IntegralScalar] = None,
) -> nd_array_field.NdArrayConnectivityField:
    assert domain is not None
    return _connectivity_class(data, None).from_array(
        data, codomain, domain=domain, dtype=dtype, skip_value=skip_value
    )


common._connectivity.register(jax.Array, _connectivity_from_array)
# jax >= 0.11: 'Tracer' is no longer a subclass of 'jax.Array' for 'singledispatch'
common._connectivity.register(jax.core.Tracer, _connectivity_from_array)


class _TwinHandle:
    """Pytree aux data carrying a concrete connectivity, compared by the identity of its table."""

    __slots__ = ("connectivity",)

    def __init__(self, connectivity: nd_array_field.NdArrayConnectivityField) -> None:
        self.connectivity = connectivity

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _TwinHandle)
            and self.connectivity._ndarray is other.connectivity._ndarray
        )

    def __hash__(self) -> int:
        return id(self.connectivity._ndarray)


_ConnectivityAuxData: TypeAlias = tuple[
    common.Domain, common.Dimension, Optional[core_defs.IntegralScalar], Optional[_TwinHandle]
]


def _flatten_connectivity(
    connectivity: nd_array_field.NdArrayConnectivityField,
) -> tuple[tuple[core_defs.NDArrayObject], _ConnectivityAuxData]:
    if isinstance(connectivity, _TwinnedConnectivity):
        handle: Optional[_TwinHandle] = _TwinHandle(connectivity._twin)
    elif isinstance(connectivity, _ValueFreeConnectivity):
        handle = None
    else:
        handle = _TwinHandle(connectivity)
    return (connectivity._ndarray,), (
        connectivity.domain,
        connectivity.codomain,
        connectivity.skip_value,
        handle,
    )


def _unflatten_connectivity(
    aux_data: _ConnectivityAuxData, children: tuple[core_defs.NDArrayObject]
) -> nd_array_field.NdArrayConnectivityField:
    # one unflatten for all classes: `lax.scan` and `lax.cond` rebuild results through the
    # unflatten of the input's class, so the class must follow from the children
    domain, codomain, skip_value, handle = aux_data
    (table,) = children
    twin = None if handle is None else handle.connectivity
    if twin is not None and isinstance(table, jax.core.Tracer):
        return _TwinnedConnectivity(domain, table, codomain, skip_value, twin)
    return _connectivity_class(table, twin)(domain, table, codomain, skip_value)  # type: ignore[abstract, unused-ignore] # see '_unflatten_field'


# jax registers pytree nodes by exact type
for _cls in (
    nd_array_field.NdArrayConnectivityField,
    _ConcreteConnectivity,
    _TwinnedConnectivity,
    _ValueFreeConnectivity,
):
    jax.tree_util.register_pytree_node(
        _cls,  # type: ignore[type-abstract, unused-ignore] # see 'NdArrayField' above
        _flatten_connectivity,
        _unflatten_connectivity,
    )
