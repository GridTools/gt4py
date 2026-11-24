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
`Dimension`s of the offset provider and the parameters in `static_params` are static.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp

from gt4py._core import ndarray_utils as core_ndarray_utils
from gt4py.next import backend as next_backend, common
from gt4py.next.embedded import nd_array_field


def _is_field(x: Any) -> bool:
    return isinstance(x, common.Field)


@dataclasses.dataclass(frozen=True)
class _Alias:
    """A field that was passed before, at leaf `index` of the arguments."""

    index: int


jax.tree_util.register_static(_Alias)


def _is_field_or_alias(x: Any) -> bool:
    return isinstance(x, common.Field | _Alias)


def _mark_aliases(tree: Any) -> Any:
    leaves, treedef = jax.tree.flatten(tree, is_leaf=_is_field)
    first: dict[int, int] = {}
    aliased = False
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, common.Field) and (index := first.setdefault(id(leaf), i)) != i:
            leaves[i] = _Alias(index)
            aliased = True
    return jax.tree.unflatten(treedef, leaves) if aliased else tree


def _resolve_aliases(tree: Any) -> Any:
    leaves, treedef = jax.tree.flatten(tree, is_leaf=_is_field_or_alias)
    return jax.tree.unflatten(
        treedef, [leaves[leaf.index] if isinstance(leaf, _Alias) else leaf for leaf in leaves]
    )


def _assignable_fields(tree: Any) -> list[nd_array_field.NdArrayField]:
    return [
        leaf
        for leaf in jax.tree.leaves(tree, is_leaf=_is_field)
        if isinstance(leaf, nd_array_field.NdArrayField)
        and not isinstance(leaf, common.Connectivity)
    ]


@dataclasses.dataclass(frozen=True, kw_only=True)
class JaxJitBackend(next_backend.EmbeddedBackend):
    """
    Runs programs and field operators embedded, under `jax.jit`.

    JAX arrays are immutable, so an assignment to an `out` field gives it a new
    buffer: the jitted function returns the new buffer of each field it assigned
    to, and the caller's fields get them. A field passed more than once is one
    field in the trace, so its aliases see the assignment.
    """

    def wrap(
        self,
        entry: Callable[..., None],
        signature: inspect.Signature,
        static_params: Sequence[str],
    ) -> Callable[..., None]:
        def traced(*args: Any, **kwargs: Any) -> list[Any]:
            fields = _assignable_fields((args, kwargs))
            buffers = [field.ndarray for field in fields]
            args, kwargs = _resolve_aliases((args, kwargs))
            entry(*args, **kwargs)
            return [
                field.ndarray if field.ndarray is not buffer else None
                for field, buffer in zip(fields, buffers, strict=True)
            ]

        traced.__signature__ = signature  # type: ignore[attr-defined] # for `static_argnames`
        jitted = jax.jit(traced, static_argnames=tuple(static_params))

        def run(*args: Any, **kwargs: Any) -> None:
            args, kwargs = _mark_aliases((args, kwargs))
            new_buffers = jitted(*args, **kwargs)
            for field, buffer in zip(_assignable_fields((args, kwargs)), new_buffers, strict=True):
                if buffer is not None:
                    field._set_buffer(buffer)

        run.__wrapped__ = jitted  # type: ignore[attr-defined]
        return run


jax_jit = JaxJitBackend(name="jax_jit", allocator=cast(core_ndarray_utils.ArrayNamespace, jnp))
