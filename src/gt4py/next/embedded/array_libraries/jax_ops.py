# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`ArrayOps` for JAX arrays, and `NdArrayField` as a JAX pytree."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any, Optional, TypeAlias

import array_api_compat
import jax
import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import array_ops, nd_array_field


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

    def is_traced(self, array: core_defs.NDArrayObject) -> bool:
        return isinstance(array, jax.core.Tracer)

    def concrete_eval(self) -> contextlib.AbstractContextManager:
        return jax.ensure_compile_time_eval()

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
    concrete = connectivity._concrete
    handle = None if concrete is None else _TwinHandle(concrete)
    return (connectivity._ndarray,), (
        connectivity.domain,
        connectivity.codomain,
        connectivity.skip_value,
        handle,
    )


def _unflatten_connectivity(
    aux_data: _ConnectivityAuxData, children: tuple[core_defs.NDArrayObject]
) -> nd_array_field.NdArrayConnectivityField:
    domain, codomain, skip_value, handle = aux_data
    twin = None if handle is None else handle.connectivity
    return nd_array_field.NdArrayConnectivityField(domain, children[0], codomain, skip_value, twin)  # type: ignore[abstract, unused-ignore] # see '_unflatten_field'


# jax registers pytree nodes by exact type, so the connectivity needs its own registration
jax.tree_util.register_pytree_node(
    nd_array_field.NdArrayConnectivityField,  # type: ignore[type-abstract, unused-ignore] # see 'NdArrayField' above
    _flatten_connectivity,
    _unflatten_connectivity,
)
