# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`ArrayOps` for JAX arrays, and `NdArrayField` as a JAX pytree."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import array_api_compat
import jax
import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import array_ops, nd_array_field


class JaxOps(array_ops.ArrayOps):
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

    def to_numpy(self, array: core_defs.NDArrayObject) -> np.ndarray:
        return np.asarray(array)

    def buffer_info(self, array: core_defs.NDArrayObject) -> common.BufferInfo:
        raise NotImplementedError("'__gt_buffer_info__' for JAX arrays not yet implemented.")


array_ops.register("jax.numpy", JaxOps)

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
