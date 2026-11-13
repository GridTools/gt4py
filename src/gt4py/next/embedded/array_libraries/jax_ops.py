# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
`ArrayOps` for JAX arrays, and the embedded fields as JAX pytrees.

Everything gt4py knows about JAX tracing is in this module (ADR 0029). A
connectivity table inside a trace is in one of three cases, told apart by
`_values`:

- concrete: operations reading its values run under `jax.ensure_compile_time_eval`;
- a tracer standing for a concrete table, which it got as a pytree leaf, or
  restricted from such a tracer: the values are read from `_StaticValues`;
- any other tracer: there are no values, all its targets are assumed to be in
  range and the gather clamps them.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Optional, TypeAlias

import array_api_compat
import jax
import jax.numpy as jnp
import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import array_ops, nd_array_field
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


class _StaticValues:
    """
    The concrete values of `table`, a placeholder of a trace, and of the tables restricted from it.

    Tables are found by identity, so any other table has no values here.
    """

    __slots__ = ("_restricted", "table", "values")

    def __init__(self, values: core_defs.NDArrayObject, table: Any) -> None:
        self.values = values
        self.table = table
        # weak: a restricted table may belong to a trace which ends before the one of `table`
        self._restricted = _WeakIdDict()

    def add_restricted(self, table: Any, values: core_defs.NDArrayObject) -> None:
        self._restricted[table] = values

    def get(self, table: Any) -> Optional[core_defs.NDArrayObject]:
        return self.values if table is self.table else self._restricted.get(table)


def _has_values(table: Any) -> bool:
    return isinstance(table, jax.Array) and not isinstance(table, jax.core.Tracer)


def _values(table: Any, static: Optional[_StaticValues]) -> Optional[core_defs.NDArrayObject]:
    """Return the concrete array with the values of `table`, `None` if they exist only at run time."""
    if _has_values(table):
        return table
    return None if static is None else static.get(table)


#: per concrete table, its restrictions by key: restricting the same table again returns the same
#: array, which the pytree aux data identifies
_restricted_values = _WeakIdDict()


def _restrict_values(
    values: core_defs.NDArrayObject, key: common.RelativeIndexSequence
) -> core_defs.NDArrayObject:
    if (by_key := _restricted_values.get(values)) is None:
        by_key = _restricted_values[values] = {}
    if (restricted := by_key.get(key)) is None:
        with jax.ensure_compile_time_eval():
            restricted = by_key[key] = values[key]
    return restricted


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
        return array.at[key].set(value)  # type: ignore[attr-defined] # `NDArrayObject` typing is not complete

    def hyperslice(
        self,
        table: core_defs.NDArrayObject,
        image_range: common.UnitRange,
        skip_value: Optional[core_defs.IntegralScalar] = None,
        *,
        static: Optional[_StaticValues] = None,
    ) -> Optional[tuple[slice, ...]]:
        values = _values(table, static)
        if values is None:
            return (slice(None),) * table.ndim
        with jax.ensure_compile_time_eval():
            return super().hyperslice(values, image_range, skip_value)

    def restrict_table(
        self,
        table: core_defs.NDArrayObject,
        key: common.RelativeIndexSequence,
        *,
        static: Optional[_StaticValues] = None,
    ) -> core_defs.NDArrayObject:
        values = _values(table, static)
        if values is None:
            return super().restrict_table(table, key)
        restricted_values = _restrict_values(values, key)
        if values is table:
            return restricted_values
        assert static is not None
        restricted = super().restrict_table(table, key)
        static.add_restricted(restricted, restricted_values)
        return restricted

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

# TODO(havogt): we currently enable 64-bit support by default, but we might want to make this configurable via the GT4Py config
jax.config.update("jax_enable_x64", True)


def _flatten_field(
    field: nd_array_field.NdArrayField,
) -> tuple[tuple[core_defs.NDArrayObject], common.Domain]:
    # JAX also flattens pytrees whose leaves are not arrays (e.g. `jax.tree.map` results or
    # `vmap` axis specifications), so only arrays of other libraries are rejected
    assert not array_api_compat.is_array_api_obj(field.ndarray) or array_api_compat.is_jax_array(
        field.ndarray
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


_ConnectivityAuxData: TypeAlias = tuple[
    common.Domain, common.Dimension, Optional[core_defs.IntegralScalar], Optional[_TableIdentity]
]


def _flatten_connectivity(
    connectivity: nd_array_field.NdArrayConnectivityField,
) -> tuple[tuple[core_defs.NDArrayObject], _ConnectivityAuxData]:
    table = connectivity._ndarray
    # the aux data identifies the values of a table by its buffer, which a mutable array does not
    # allow. The placeholders of JAX are array API objects too, so "not a JAX array" is no test.
    assert not (array_api_compat.is_numpy_array(table) or array_api_compat.is_cupy_array(table)), (
        f"Only connectivities of JAX arrays are JAX pytrees, got a table of '{type(table)}'."
    )
    values = _values(table, connectivity._static)
    return (table,), (
        connectivity.domain,
        connectivity.codomain,
        connectivity.skip_value,
        None if values is None else _TableIdentity(values),
    )


def _unflatten_connectivity(
    aux_data: _ConnectivityAuxData, children: tuple[core_defs.NDArrayObject]
) -> nd_array_field.NdArrayConnectivityField:
    domain, codomain, skip_value, identity = aux_data
    (table,) = children
    # a child without values stands for the flattened table: a tracer, or one of the placeholders
    # jax rebuilds the branches of `lax.cond` and the carry of `lax.scan` around
    static = (
        None if identity is None or _has_values(table) else _StaticValues(identity.values, table)
    )
    return nd_array_field.NdArrayConnectivityField(
        domain, table, codomain, skip_value, _static=static
    )  # type: ignore[abstract, unused-ignore] # see '_unflatten_field'


# jax registers pytree nodes by exact type, so the connectivity needs its own registration
jax.tree_util.register_pytree_node(
    nd_array_field.NdArrayConnectivityField,  # type: ignore[type-abstract, unused-ignore] # see 'NdArrayField' above
    _flatten_connectivity,
    _unflatten_connectivity,
)
