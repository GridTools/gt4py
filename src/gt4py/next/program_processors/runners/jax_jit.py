# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Embedded execution of programs and field operators under `jax.jit`.

Fields, connectivities and scalars are arguments of the jitted function; the
`Dimension`s of the offset provider, a `domain=` argument and the parameters in
`static_params` are static.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from gt4py.next import backend as next_backend, common, named_collections
from gt4py.next.embedded import nd_array_field
from gt4py.next.embedded.array_libraries import jax_ops
from gt4py.next.type_system import type_specifications as ts


def _is_field(x: Any) -> bool:
    return isinstance(x, common.Field)


def _assignable_fields(tree: Any) -> list[nd_array_field.NdArrayField]:
    return [
        leaf
        for leaf in jax.tree.leaves(tree, is_leaf=_is_field)
        if isinstance(leaf, nd_array_field.NdArrayField)
        and not isinstance(leaf, common.Connectivity)
    ]


@dataclasses.dataclass(frozen=True, kw_only=True)
class JaxJitBackend(next_backend.EmbeddedJitBackend):
    """
    Runs programs and field operators embedded, under `jax.jit`.

    JAX arrays are immutable, so an assignment to an `out` field gives it a new
    buffer: the jitted function returns the new buffer of each field it assigned
    to, and the caller's fields get them.
    """

    def jit(
        self,
        entry: Callable[..., None],
        signature: inspect.Signature,
        static_params: Sequence[str],
        param_types: Iterable[ts.TypeSpec],
    ) -> Callable[..., None]:
        for type_ in param_types:
            for cls in named_collections.dataclass_types(type_):
                jax_ops.register_named_collection(cls)

        def traced(*args: Any, **kwargs: Any) -> list[Any]:
            fields = _assignable_fields((args, kwargs))
            buffers = [field.ndarray for field in fields]
            entry(*args, **kwargs)
            return [
                field.ndarray if field.ndarray is not buffer else None
                for field, buffer in zip(fields, buffers, strict=True)
            ]

        traced.__signature__ = signature  # type: ignore[attr-defined] # for `static_argnames`
        jitted = jax.jit(traced, static_argnames=tuple(static_params))

        def run(*args: Any, **kwargs: Any) -> None:
            new_buffers = jitted(*args, **kwargs)
            for field, buffer in zip(_assignable_fields((args, kwargs)), new_buffers, strict=True):
                if buffer is not None:
                    field._set_buffer(buffer)

        run.jitted = jitted  # type: ignore[attr-defined] # to inspect the traces
        return run


jax_jit = JaxJitBackend(name="jax_jit", allocator=jnp)
