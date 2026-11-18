# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
`ArrayOps` and `ConnectivityOps` for JAX arrays, and the embedded fields as JAX pytrees.

Everything gt4py knows about JAX tracing is in this module (ADR 0029). A
connectivity table inside a trace is in one of three cases, told apart by
`JaxConnectivityOps.values`:

- concrete: operations reading its values run under `jax.ensure_compile_time_eval`;
- a tracer standing for a concrete table, which it got as a pytree leaf, or
  restricted from such a tracer: the values are those of the concrete table;
- any other tracer: there are no values, all its targets are assumed to be in
  range and the gather clamps them.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, NamedTuple, Optional

import array_api_compat
import jax
import jax.numpy as jnp
import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import array_ops, connectivity_ops, nd_array_field
from gt4py.next.ffront import experimental, fbuiltins


class _WeakIdDict:
    """Values for objects which are found by identity and referenced weakly."""

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: dict[int, tuple[weakref.ref, Any]] = {}

    def __setitem__(self, obj: Any, value: Any) -> None:
        key = id(obj)
        self._items[key] = (weakref.ref(obj, lambda _: self._items.pop(key, None)), value)

    def get(self, obj: Any) -> Any:
        ref, value = self._items.get(id(obj), (None, None))
        return value if ref is not None and ref() is obj else None


class _TableIdentity:
    """Pytree aux data of a connectivity: equal if `values` is the same array."""

    __slots__ = ("values",)

    def __init__(self, values: core_defs.NDArrayObject) -> None:
        self.values = values

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _TableIdentity) and self.values is other.values

    def __hash__(self) -> int:
        return id(self.values)

    def __repr__(self) -> str:
        return f"<values of the array at {id(self.values):#x}>"


def _has_values(ndarray: Any) -> bool:
    return isinstance(ndarray, jax.Array) and not isinstance(ndarray, jax.core.Tracer)


#: per concrete table, its restrictions by buffer slice: restricting the same table again returns
#: the same array, which the pytree aux data identifies
_restricted_values = _WeakIdDict()


def _restrict_values(
    values: core_defs.NDArrayObject, buffer_slice: common.RelativeIndexSequence
) -> core_defs.NDArrayObject:
    if (by_slice := _restricted_values.get(values)) is None:
        by_slice = _restricted_values[values] = {}
    if (restricted := by_slice.get(buffer_slice)) is None:
        with jax.ensure_compile_time_eval():
            restricted = by_slice[buffer_slice] = values[buffer_slice]
    return restricted


@dataclasses.dataclass(frozen=True, eq=False)
class JaxConnectivityOps(connectivity_ops.ConnectivityOps):
    #: The concrete array with the values of the connectivity's table, `None` if they exist
    #: only at run time. The table itself if it is concrete.
    values: Optional[core_defs.NDArrayObject] = None

    def inverse_image_box(
        self,
        ndarray: core_defs.NDArrayObject,
        image_range: common.UnitRange,
        skip_value: Optional[core_defs.IntegralScalar] = None,
    ) -> Optional[tuple[slice, ...]]:
        if self.values is None:
            return (slice(None),) * ndarray.ndim
        with jax.ensure_compile_time_eval():
            return super().inverse_image_box(self.values, image_range, skip_value)

    @classmethod
    def from_ndarray(cls, ndarray: core_defs.NDArrayObject) -> JaxConnectivityOps:
        return cls(array_ops.of(ndarray), ndarray if _has_values(ndarray) else None)

    def restrict(
        self, ndarray: core_defs.NDArrayObject, buffer_slice: common.RelativeIndexSequence
    ) -> tuple[core_defs.NDArrayObject, connectivity_ops.ConnectivityOps]:
        if self.values is None:
            return super().restrict(ndarray, buffer_slice)
        values = _restrict_values(self.values, buffer_slice)
        restricted = (
            values if self.values is ndarray else super().restrict(ndarray, buffer_slice)[0]
        )
        return restricted, dataclasses.replace(self, values=values)


def _as_offset(offset: fbuiltins.FieldOffset, offset_field: nd_array_field.NdArrayField) -> Any:
    # adding the coordinates would stage the table under a trace even for concrete offsets
    with jax.ensure_compile_time_eval():
        return nd_array_field._as_offset(offset, offset_field)


class JaxOps(array_ops.ArrayOps):
    builtin_overrides: ClassVar[Mapping[Any, Callable[..., Any]]] = {
        experimental.as_offset: _as_offset
    }

    def operand(self, value: Any, like: core_defs.NDArrayObject) -> Any:
        return value

    def _elementwise_function(self, name: str) -> Callable[..., Any]:
        if name == "gamma":
            import jax.scipy.special

            return jax.scipy.special.gamma
        return super()._elementwise_function(name)

    def setitem(
        self, array: core_defs.NDArrayObject, key: common.RelativeIndexSequence, value: Any
    ) -> core_defs.NDArrayObject:
        self.check_assignable(value)
        return array.at[key].set(value)  # type: ignore[attr-defined] # `NDArrayObject` typing is not complete

    def gather(
        self, array: core_defs.NDArrayObject, indices: tuple[core_defs.NDArrayObject, ...]
    ) -> core_defs.NDArrayObject:
        # an index from a table without values may be out of range: it is clamped, at both ends
        return array.at[indices].get(mode="clip", wrap_negative_indices=False)  # type: ignore[attr-defined] # `NDArrayObject` typing is not complete

    def to_numpy(self, array: core_defs.NDArrayObject) -> np.ndarray:
        return np.asarray(array)

    def buffer_info(self, array: core_defs.NDArrayObject) -> common.BufferInfo:
        raise NotImplementedError("'__gt_buffer_info__' for JAX arrays not yet implemented.")


array_ops.register(JaxOps(jnp))
connectivity_ops.register(jnp, JaxConnectivityOps)

# TODO(havogt): we currently enable 64-bit support by default, but we might want to make this configurable via the GT4Py config
jax.config.update("jax_enable_x64", True)


def _flatten_field(
    field: nd_array_field.NdArrayField,
) -> tuple[tuple[core_defs.NDArrayObject], common.Domain]:
    # JAX also flattens pytrees whose leaves are not arrays (e.g. `jax.tree.map` results, `vmap`
    # axis specifications, or the placeholders of `lax.cond` branches, which are array API objects
    # too), so only arrays of other libraries are rejected
    assert not (
        array_api_compat.is_numpy_array(field.ndarray)
        or array_api_compat.is_cupy_array(field.ndarray)
    ), f"Only fields of JAX arrays are JAX pytrees, got a field of '{type(field.ndarray)}'."
    return (field.ndarray,), field.domain


def _unflatten_field(
    domain: common.Domain, children: tuple[core_defs.NDArrayObject]
) -> nd_array_field.NdArrayField:
    return nd_array_field.NdArrayField(domain, children[0])


jax.tree_util.register_pytree_node(
    nd_array_field.NdArrayField,
    _flatten_field,
    _unflatten_field,
)


class _ConnectivityAuxData(NamedTuple):
    domain: common.Domain
    codomain: common.Dimension
    skip_value: Optional[core_defs.IntegralScalar]
    values: Optional[_TableIdentity]


def _flatten_connectivity(
    connectivity: nd_array_field.NdArrayConnectivityField,
) -> tuple[tuple[core_defs.NDArrayObject], _ConnectivityAuxData]:
    ndarray = connectivity._ndarray
    # the aux data identifies the values of a table by its buffer, which a mutable array does not
    # allow. The placeholders of JAX are array API objects too, so "not a JAX array" is no test.
    assert not (
        array_api_compat.is_numpy_array(ndarray) or array_api_compat.is_cupy_array(ndarray)
    ), f"Only connectivities of JAX arrays are JAX pytrees, got a table of '{type(ndarray)}'."
    try:
        ops = connectivity._connectivity_ops
    except TypeError:
        # a placeholder that is no array has `ConnectivityOps` only if unflattening gave it some
        ops = None
    values = ops.values if isinstance(ops, JaxConnectivityOps) else None
    return (ndarray,), _ConnectivityAuxData(
        connectivity.domain,
        connectivity.codomain,
        connectivity.skip_value,
        None if values is None else _TableIdentity(values),
    )


def _unflatten_connectivity(
    aux_data: _ConnectivityAuxData, children: tuple[core_defs.NDArrayObject]
) -> nd_array_field.NdArrayConnectivityField:
    (ndarray,) = children
    identity = aux_data.values
    # a child without values stands for the flattened table: a tracer, or one of the placeholders
    # jax rebuilds the branches of `lax.cond` and the carry of `lax.scan` around. Under `vmap` the
    # child is one element of the flattened table, whose values the aux data does not have.
    ops = (
        None
        if identity is None
        or _has_values(ndarray)
        or getattr(ndarray, "shape", identity.values.shape) != identity.values.shape
        else connectivity_ops.of(identity.values)
    )
    return nd_array_field.NdArrayConnectivityField(
        aux_data.domain,
        ndarray,
        aux_data.codomain,
        aux_data.skip_value,
        connectivity_ops=ops,
    )  # type: ignore[abstract, unused-ignore] # see '_unflatten_field'


# jax registers pytree nodes by exact type, so the connectivity needs its own registration
jax.tree_util.register_pytree_node(
    nd_array_field.NdArrayConnectivityField,  # type: ignore[type-abstract, unused-ignore] # see 'NdArrayField' above
    _flatten_connectivity,
    _unflatten_connectivity,
)
