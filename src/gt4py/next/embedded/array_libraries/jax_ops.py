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
- a tracer standing for a concrete table, which it got as a pytree leaf: the
  values are read from that table, kept in a `_StaticTable`;
- a tracer computed inside the trace: there are no values, all its targets are
  assumed to be in range and the gather clamps them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Optional, TypeAlias

import array_api_compat
import jax
import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import array_ops, nd_array_field
from gt4py.next.ffront import experimental, fbuiltins


class _StaticTable:
    """
    The concrete array `values` holding the values of `table`, a placeholder of a trace.

    It is also the pytree aux data of a connectivity, without `table`: two are equal
    if their values are the same array.
    """

    __slots__ = ("table", "values")

    def __init__(self, values: core_defs.NDArrayObject, table: Any = None) -> None:
        self.values = values
        self.table = table

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StaticTable) and self.values is other.values

    def __hash__(self) -> int:
        return id(self.values)

    def __repr__(self) -> str:
        return f"<values of the array at {id(self.values):#x}>"


def _has_values(table: Any) -> bool:
    return isinstance(table, np.ndarray | jax.Array) and not isinstance(table, jax.core.Tracer)


def _values(table: Any, static: Optional[_StaticTable]) -> Optional[core_defs.NDArrayObject]:
    """Return the concrete array with the values of `table`, `None` if they exist only at run time."""
    if _has_values(table):
        return table
    # a connectivity copied with another table still carries the `static` of the old one
    return static.values if static is not None and static.table is table else None


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

    def _elementwise(self, name: str) -> Callable[..., Any]:
        if name == "gamma":
            import jax.scipy.special

            return jax.scipy.special.gamma
        return super()._elementwise(name)

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
        static: Optional[_StaticTable] = None,
    ) -> Optional[tuple[slice, ...]]:
        values = _values(table, static)
        if values is None:
            return (slice(None),) * table.ndim
        if values is not table:
            return array_ops.of(values).hyperslice(values, image_range, skip_value)
        with jax.ensure_compile_time_eval():
            return super().hyperslice(table, image_range, skip_value)

    def restrict_table(
        self,
        table: core_defs.NDArrayObject,
        key: common.RelativeIndexSequence,
        *,
        static: Optional[_StaticTable] = None,
    ) -> tuple[core_defs.NDArrayObject, Optional[_StaticTable]]:
        values = _values(table, static)
        if values is table:
            with jax.ensure_compile_time_eval():
                return super().restrict_table(table, key)
        restricted, _ = super().restrict_table(table, key)
        if values is None:
            return restricted, None
        restricted_values, _ = array_ops.of(values).restrict_table(values, key)
        return restricted, _StaticTable(restricted_values, restricted)

    def gather_index(
        self,
        index: core_defs.NDArrayObject,
        size: int,
        table: core_defs.NDArrayObject,
        *,
        static: Optional[_StaticTable] = None,
    ) -> core_defs.NDArrayObject:
        if _values(table, static) is None:
            return jax.numpy.clip(index, 0, size - 1)  # type: ignore[arg-type, return-value, unused-ignore] # `NDArrayObject` typing is not complete
        return index

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


_ConnectivityAuxData: TypeAlias = tuple[
    common.Domain, common.Dimension, Optional[core_defs.IntegralScalar], Optional[_StaticTable]
]


def _flatten_connectivity(
    connectivity: nd_array_field.NdArrayConnectivityField,
) -> tuple[tuple[core_defs.NDArrayObject], _ConnectivityAuxData]:
    table = connectivity._ndarray
    values = _values(table, connectivity._static)
    return (table,), (
        connectivity.domain,
        connectivity.codomain,
        connectivity.skip_value,
        None if values is None else _StaticTable(values),
    )


def _unflatten_connectivity(
    aux_data: _ConnectivityAuxData, children: tuple[core_defs.NDArrayObject]
) -> nd_array_field.NdArrayConnectivityField:
    domain, codomain, skip_value, static = aux_data
    (table,) = children
    if static is not None:
        # jax compares the aux data of the branches of `lax.cond` and of the carry of `lax.scan`
        # on trees it rebuilds around placeholders that are not tracers, so they keep it as well
        static = None if _has_values(table) else _StaticTable(static.values, table)
    return nd_array_field.NdArrayConnectivityField(domain, table, codomain, skip_value, static)  # type: ignore[abstract, unused-ignore] # see '_unflatten_field'


# jax registers pytree nodes by exact type, so the connectivity needs its own registration
jax.tree_util.register_pytree_node(
    nd_array_field.NdArrayConnectivityField,  # type: ignore[type-abstract, unused-ignore] # see 'NdArrayField' above
    _flatten_connectivity,
    _unflatten_connectivity,
)
