# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Embedded execution of programs and field operators under `torch.compile`.

`torch_compile` (created on first access, it requires `torch`) runs the Python
definition on fields of torch tensors in a `torch.compile` region, through a
`torch.compiler.nonstrict_trace` function, as `torch_ops` describes. Fields and
connectivities are arguments of the region; Dynamo specializes on the other
arguments (see `embedded_jit` for the static ones).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from gt4py.next import backend as next_backend
from gt4py.next.embedded import context as embedded_context
from gt4py.next.program_processors.runners import embedded_jit
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass(frozen=True)
class TorchCompileBackend(embedded_jit.EmbeddedJitBackend):
    """Runs programs and field operators embedded, under `torch.compile`."""

    def jit(
        self,
        function: Callable[..., None],
        static_argnames: Iterable[str] = (),
        param_names: Sequence[str] = (),
        param_types: Iterable[ts.TypeSpec] = (),
    ) -> CompileRunner:
        import torch.utils._pytree as torch_pytree

        for type_ in param_types:
            for cls in embedded_jit.named_collection_dataclasses(type_):
                if cls not in _registered_dataclasses:
                    torch_pytree.register_dataclass(cls)
                    _registered_dataclasses.add(cls)
        return CompileRunner(function, frozenset(static_argnames), tuple(param_names))


_registered_dataclasses: set[type] = set()


class CompileRunner(embedded_jit.EmbeddedJitRunner):
    """
    One `torch.compile` region per combination of static arguments.

    The static arguments are closed over by the non-strictly traced function,
    since `nonstrict_trace` only takes tensors, pytrees and constants. The entry
    point assigns to its `out` fields in place, which the region applies to the
    caller's tensors.
    """

    def __init__(
        self,
        function: Callable[..., None],
        static_argnames: frozenset[str],
        param_names: tuple[str, ...],
    ) -> None:
        super().__init__(function, static_argnames, param_names)
        self._embedded_error: BaseException | None = None
        self._regions: dict[
            tuple[embedded_jit.StaticItems, embedded_jit.StaticItems], Callable[..., Any]
        ] = {}

    def _region(
        self, static_kwargs: embedded_jit.StaticItems, static_provider: embedded_jit.StaticItems
    ) -> Callable[..., Any]:
        import torch

        function = self._function

        @torch.compiler.nonstrict_trace
        def run_embedded(
            args: tuple[Any, ...], kwargs: dict[str, Any], connectivities: dict[str, Any]
        ) -> tuple[()]:
            with embedded_context.update(
                offset_provider={**connectivities, **dict(static_provider)}
            ):
                try:
                    function(*args, **kwargs, **dict(static_kwargs))
                except Exception as error:
                    # Dynamo reports it as an unsupported output type
                    self._embedded_error = error
                    raise
            # `nonstrict_trace` does not support a `None` output
            return ()

        def region(
            args: tuple[Any, ...], kwargs: dict[str, Any], connectivities: dict[str, Any]
        ) -> tuple[()]:
            return run_embedded(args, kwargs, connectivities)

        # Dynamo caches per code object, with a recompile limit: every region needs its own
        region.__code__ = region.__code__.replace()
        # with dynamic shapes, Dynamo makes domain bounds symbolic, which cannot be in a pytree context
        return torch.compile(region, fullgraph=True, dynamic=False)

    def _run(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        connectivities: dict[str, Any],
        static_kwargs: embedded_jit.StaticItems,
        static_provider: embedded_jit.StaticItems,
    ) -> None:
        import torch
        import torch.utils._pytree as torch_pytree

        key = (static_kwargs, static_provider)
        if (region := self._regions.get(key)) is None:
            region = self._regions[key] = self._region(static_kwargs, static_provider)
        args, kwargs = torch_pytree.tree_map(_as_region_argument, (args, kwargs))
        try:
            region(args, kwargs, connectivities)
        except torch._dynamo.exc.Unsupported as unsupported:
            if (error := self._embedded_error) is None:
                raise
            self._embedded_error = None
            raise error from unsupported


def _as_region_argument(value: Any) -> Any:
    import numpy as np
    import torch

    # `nonstrict_trace` takes no NumPy scalars. Dynamo specializes on Python integers and
    # booleans, as for domain bounds and conditions; a floating-point scalar becomes a 0-d
    # tensor, which keeps its dtype.
    if isinstance(value, np.bool_ | np.integer):
        return value.item()
    if isinstance(value, np.generic):
        return torch.as_tensor(value)
    return value


def __getattr__(name: str) -> TorchCompileBackend:
    if name != "torch_compile":
        raise AttributeError(f"Module '{__name__}' has no attribute '{name}'.")
    import torch

    from gt4py.next.embedded.array_libraries import (
        torch_ops,  # noqa: F401 [unused-import] # registers the pytrees
    )

    backend = TorchCompileBackend(
        name="torch_compile",
        executor=embedded_jit.not_compiled,
        allocator=torch,
        transforms=next_backend.DEFAULT_TRANSFORMS,
    )
    globals()[name] = backend
    return backend


torch_compile: TorchCompileBackend
