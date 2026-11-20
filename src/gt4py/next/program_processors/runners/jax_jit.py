# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Embedded execution of programs and field operators under `jax.jit`.

`jax_jit` (created on first access, it requires `jax`) runs the Python definition
on fields of JAX arrays inside `jax.jit`. Fields, connectivities and scalars are
arguments of the jitted function (see `embedded_jit` for the static ones).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from gt4py.next import backend as next_backend, common
from gt4py.next.embedded import context as embedded_context, nd_array_field
from gt4py.next.program_processors.runners import embedded_jit
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass(frozen=True)
class JaxJitBackend(embedded_jit.EmbeddedJitBackend):
    """Runs programs and field operators embedded, under `jax.jit`."""

    def jit(
        self,
        function: Callable[..., None],
        static_argnames: Iterable[str] = (),
        param_names: Sequence[str] = (),
        param_types: Iterable[ts.TypeSpec] = (),
    ) -> JitRunner:
        import jax

        for type_ in param_types:
            for cls in embedded_jit.named_collection_dataclasses(type_):
                if cls not in _registered_dataclasses:
                    jax.tree_util.register_dataclass(
                        cls, data_fields=[f.name for f in dataclasses.fields(cls)], meta_fields=[]
                    )
                    _registered_dataclasses.add(cls)
        return JitRunner(function, frozenset(static_argnames), tuple(param_names))


_registered_dataclasses: set[type] = set()


def _is_field(x: Any) -> bool:
    return isinstance(x, common.Field)


def _written_fields(tree: Any) -> list[nd_array_field.NdArrayField]:
    import jax

    return [
        leaf
        for leaf in jax.tree.leaves(tree, is_leaf=_is_field)
        if isinstance(leaf, nd_array_field.NdArrayField)
        and not isinstance(leaf, common.Connectivity)
    ]


class JitRunner(embedded_jit.EmbeddedJitRunner):
    """
    One `jax.jit` of an embedded entry point.

    The jitted function returns, for each field of the arguments, its new buffer
    if the entry point assigned to it and `None` otherwise; the new buffers are
    then swapped into the caller's fields.
    """

    def __init__(
        self,
        function: Callable[..., None],
        static_argnames: frozenset[str],
        param_names: tuple[str, ...],
    ) -> None:
        import jax

        super().__init__(function, static_argnames, param_names)
        self._jitted = jax.jit(self._traced, static_argnames=("static_kwargs", "static_provider"))

    def _traced(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        connectivities: dict[str, Any],
        *,
        static_kwargs: embedded_jit.StaticItems,
        static_provider: embedded_jit.StaticItems,
    ) -> list[Any]:
        fields = _written_fields((args, kwargs))
        buffers = [field.ndarray for field in fields]
        with embedded_context.update(offset_provider={**connectivities, **dict(static_provider)}):
            self._function(*args, **kwargs, **dict(static_kwargs))
        return [
            field.ndarray if field.ndarray is not buffer else None
            for field, buffer in zip(fields, buffers, strict=True)
        ]

    def _run(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        connectivities: dict[str, Any],
        static_kwargs: embedded_jit.StaticItems,
        static_provider: embedded_jit.StaticItems,
    ) -> None:
        new_buffers = self._jitted(
            args,
            kwargs,
            connectivities,
            static_kwargs=static_kwargs,
            static_provider=static_provider,
        )
        for field, buffer in zip(_written_fields((args, kwargs)), new_buffers, strict=True):
            if buffer is not None:
                field._set_buffer(buffer)


def __getattr__(name: str) -> JaxJitBackend:
    if name != "jax_jit":
        raise AttributeError(f"Module '{__name__}' has no attribute '{name}'.")
    import jax.numpy as jnp

    from gt4py.next.embedded.array_libraries import (
        jax_ops,  # noqa: F401 [unused-import] # registers the pytrees
    )

    backend = JaxJitBackend(
        name="jax_jit",
        executor=embedded_jit.not_compiled,
        allocator=jnp,
        transforms=next_backend.DEFAULT_TRANSFORMS,
    )
    globals()[name] = backend
    return backend


jax_jit: JaxJitBackend
