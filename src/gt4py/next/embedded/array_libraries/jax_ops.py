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
connectivity table inside a trace is a tracer, and one of two cases:

- it stands for a concrete table, which it got as a pytree leaf, or it was
  restricted from such a tracer: its static values are a NumPy view of the
  concrete table, from the pytree aux data;
- it was computed in the trace by `as_offset`: there are no static values, all
  its targets are assumed to be in range and the gather clamps them.

A concrete table closed over by a trace is not supported.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, NamedTuple, Optional

import array_api_compat
import jax
import jax.numpy as jnp
import numpy as np

from gt4py._core import definitions as core_defs, ndarray_utils as core_ndarray_utils
from gt4py.next import common
from gt4py.next.embedded import array_ops, nd_array_field
from gt4py.next.ffront import experimental, fbuiltins


class _TableIdentity:
    """
    Pytree aux data of a connectivity: equal if `values` is the same read-only memory.

    NumPy returns a new view of a JAX array on every `np.asarray`, so the views of one
    buffer, and their restrictions with the same slice, are told apart by their memory.
    """

    __slots__ = ("_key", "values")

    values: np.ndarray
    _key: tuple[int, tuple[int, ...], tuple[int, ...], str]

    def __init__(self, values: core_defs.NDArrayObject) -> None:
        assert isinstance(values, np.ndarray) and not values.flags.writeable
        # holding the array keeps its memory alive, so no other array can get its address
        self.values = values
        self._key = (
            values.__array_interface__["data"][0],
            values.shape,
            values.strides,
            values.dtype.str,
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _TableIdentity) and self._key == other._key

    def __hash__(self) -> int:
        return hash(self._key)

    def __repr__(self) -> str:
        return f"<values of the array at {self._key[0]:#x}>"


def _has_values(ndarray: Any) -> bool:
    return isinstance(ndarray, jax.Array) and not isinstance(ndarray, jax.core.Tracer)


def _read_only_view(array: jax.Array) -> np.ndarray:
    values = np.asarray(array)
    values.flags.writeable = False
    return values


def _as_offset(offset: fbuiltins.FieldOffset, offset_field: nd_array_field.NdArrayField) -> Any:
    connectivity = nd_array_field._as_offset(offset, offset_field)
    assert isinstance(connectivity, nd_array_field.NdArrayConnectivityField)
    if isinstance(connectivity.ndarray, jax.core.Tracer):
        # computed in the trace: its values exist only at run time
        return dataclasses.replace(connectivity, static_values=None)
    return connectivity


def _layout_strides(array: Any) -> tuple[int, ...]:
    """Element strides of the physical layout of `array`, which need not be row-major."""
    # Older JAX versions name `format` `layout`, and its device-local part `device_local_layout`.
    array_format = getattr(array, "format", None) or array.layout
    layout = getattr(array_format, "device_local_layout", None) or array_format.layout
    shape = array.shape
    strides = [0] * len(shape)
    stride = 1
    for dim in reversed(layout.major_to_minor):
        strides[dim] = stride
        stride *= shape[dim]
    return tuple(strides)


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

    def take(
        self, array: core_defs.NDArrayObject, indices: core_defs.NDArrayObject, *, axis: int
    ) -> core_defs.NDArrayObject:
        return jnp.take(array, indices, axis=axis, mode="clip")

    def to_numpy(self, array: core_defs.NDArrayObject) -> np.ndarray:
        return np.asarray(array)

    def buffer_info(self, array: core_defs.NDArrayObject) -> core_ndarray_utils.BufferInfo:
        # JAX's `__dlpack__` rejects `stream=-1` and blocks the host on `stream=None`.
        return core_ndarray_utils.BufferInfo(
            data_ptr=array.unsafe_buffer_pointer(),  # type: ignore[attr-defined] # `NDArrayObject` typing is not complete
            shape=array.shape,
            elem_strides=_layout_strides(array),
            dtype=core_defs.dtype(self.scalar_type(array.dtype)),
            device=core_ndarray_utils.device_from_dlpack(*array.__dlpack_device__()),  # type: ignore[attr-defined] # `NDArrayObject` typing is not complete
        )


array_ops.register(JaxOps(jnp))

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
    values: Any = connectivity._static_values
    if values is ndarray:
        # a placeholder that is no array has no values; NumPy, so that a restriction inside the
        # trace is not staged
        values = _read_only_view(ndarray) if _has_values(ndarray) else None
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
    # a child that is not a concrete array stands for the flattened table: a tracer, or one of the
    # placeholders jax rebuilds the branches of `lax.cond` and the carry of `lax.scan` around.
    # Under `vmap` the child is one element of the flattened table, whose values the aux data
    # does not have.
    static_values = (
        nd_array_field._FromBuffer.TOKEN
        if _has_values(ndarray)
        or array_api_compat.is_numpy_array(ndarray)
        or array_api_compat.is_cupy_array(ndarray)
        else None
        if identity is None
        or getattr(ndarray, "shape", identity.values.shape) != identity.values.shape
        else identity.values
    )
    return nd_array_field.NdArrayConnectivityField(
        aux_data.domain,
        ndarray,
        aux_data.codomain,
        aux_data.skip_value,
        static_values=static_values,  # type: ignore[arg-type] # `NDArrayObject` typing is not complete
    )  # type: ignore[abstract, unused-ignore] # see '_unflatten_field'


# jax registers pytree nodes by exact type, so the connectivity needs its own registration
jax.tree_util.register_pytree_node(
    nd_array_field.NdArrayConnectivityField,  # type: ignore[type-abstract, unused-ignore] # see 'NdArrayField' above
    _flatten_connectivity,
    _unflatten_connectivity,
)
