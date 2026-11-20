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
arguments of the jitted function; other offset provider entries (`Dimension`s of
Cartesian offsets) and the parameters named in `static_params` are static.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Iterable
from typing import Any

from gt4py.next import backend as next_backend, common
from gt4py.next.embedded import context as embedded_context, nd_array_field


@dataclasses.dataclass(frozen=True)
class JaxJitBackend(next_backend.Backend):
    """Runs programs and field operators embedded, under `jax.jit`."""

    def jit(
        self,
        function: Callable[..., None],
        static_argnames: Iterable[str] = (),
        signature: inspect.Signature | None = None,
    ) -> JitRunner:
        return JitRunner(function, frozenset(static_argnames), signature)


def _not_compiled(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("The 'jax_jit' backend runs embedded under 'jax.jit'; it does not compile.")


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


class JitRunner:
    """
    One `jax.jit` of an embedded entry point, called like a compiled programs pool.

    The jitted function returns, for each field of the arguments, its new buffer
    if the entry point assigned to it and `None` otherwise; the new buffers are
    then swapped into the caller's fields.
    """

    def __init__(
        self,
        function: Callable[..., None],
        static_argnames: frozenset[str],
        signature: inspect.Signature | None,
    ) -> None:
        import jax

        self._function = function
        self._static_argnames = static_argnames
        self._signature = signature
        self._jitted = jax.jit(self._traced, static_argnames=("static_kwargs", "static_provider"))

    def _traced(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        connectivities: dict[str, Any],
        *,
        static_kwargs: tuple[tuple[str, Any], ...],
        static_provider: tuple[tuple[str, Any], ...],
    ) -> list[Any]:
        fields = _written_fields((args, kwargs))
        buffers = [field.ndarray for field in fields]
        with embedded_context.update(offset_provider={**connectivities, **dict(static_provider)}):
            self._function(*args, **kwargs, **dict(static_kwargs))
        return [
            field.ndarray if field.ndarray is not buffer else None
            for field, buffer in zip(fields, buffers, strict=True)
        ]

    def __call__(
        self,
        *args: Any,
        offset_provider: common.OffsetProvider,
        enable_jit: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if self._static_argnames:
            if self._signature is not None:
                bound = self._signature.bind(*args, **kwargs)
                args, kwargs = (), dict(bound.arguments)
            static_kwargs = tuple(
                sorted((name, kwargs.pop(name)) for name in self._static_argnames & kwargs.keys())
            )
        else:
            static_kwargs = ()

        connectivities: dict[str, Any] = {}
        static_provider: list[tuple[str, Any]] = []
        for name, entry in offset_provider.items():
            if isinstance(entry, common.Connectivity):
                connectivities[name] = entry
            else:
                static_provider.append((name, entry))

        new_buffers = self._jitted(
            args,
            kwargs,
            connectivities,
            static_kwargs=static_kwargs,
            static_provider=tuple(sorted(static_provider)),
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
        executor=_not_compiled,
        allocator=jnp,
        transforms=next_backend.DEFAULT_TRANSFORMS,
    )
    globals()[name] = backend
    return backend


jax_jit: JaxJitBackend
