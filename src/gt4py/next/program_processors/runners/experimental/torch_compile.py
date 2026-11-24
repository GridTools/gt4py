# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Embedded execution of programs and field operators under `torch.compile`.

The embedded code runs in a region through a `torch.compiler.nonstrict_trace`
function (see `torch_ops`). Fields, connectivities and scalars are arguments of the
region; the `Dimension`s of the offset provider and the parameters in
`static_params` are static.
"""

from __future__ import annotations

import contextvars
import dataclasses
import inspect
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import torch

from gt4py._core import ndarray_utils as core_ndarray_utils
from gt4py.next import backend as next_backend, common


# Dynamo reports an exception of the embedded code as an unsupported output type
_embedded_error: contextvars.ContextVar[Exception | None] = contextvars.ContextVar(
    "_embedded_error", default=None
)

_Static = tuple[tuple[str, Any], ...]


def _as_region_argument(value: Any) -> Any:
    # Dynamo specializes on Python and NumPy scalars, which `nonstrict_trace` does not
    # take either: as 0-d tensors they are traced, as by `jax.jit`, and keep their dtype
    if isinstance(value, float):
        return torch.tensor(value, dtype=torch.float64)
    if isinstance(value, bool | int | np.generic):
        return torch.as_tensor(value)
    return value


def _region(entry: Callable[..., None], static: _Static) -> Callable[..., Any]:
    @torch.compiler.nonstrict_trace
    def run_embedded(kwargs: dict[str, Any], connectivities: dict[str, Any]) -> tuple[()]:
        static_kwargs = dict(static)
        offset_provider = {**connectivities, **dict(static_kwargs.pop("offset_provider"))}
        try:
            entry(**kwargs, **static_kwargs, offset_provider=offset_provider)
        except Exception as error:
            _embedded_error.set(error)
            raise
        # `nonstrict_trace` does not support a `None` output
        return ()

    def region(kwargs: dict[str, Any], connectivities: dict[str, Any]) -> tuple[()]:
        return run_embedded(kwargs, connectivities)

    # Dynamo's recompile limit is per code object: each region needs its own
    region.__code__ = region.__code__.replace()
    # `dynamic=False`: see `torch_ops`
    return torch.compile(region, fullgraph=True, dynamic=False)


@dataclasses.dataclass(frozen=True, kw_only=True)
class TorchCompileBackend(next_backend.EmbeddedBackend):
    """
    Runs programs and field operators embedded, under `torch.compile`.

    Dynamo cannot rebuild frozen dataclasses such as `Dimension` and `Domain` from a
    region input, so the static arguments are closed over by one region per
    combination of their values. The `out` fields are assigned to in place.
    """

    def wrap(
        self,
        entry: Callable[..., None],
        signature: inspect.Signature,
        static_params: Sequence[str],
    ) -> Callable[..., None]:
        static_names = frozenset(static_params)
        regions: dict[_Static, Callable[..., Any]] = {}

        def run(*args: Any, **kwargs: Any) -> None:
            arguments = signature.bind(*args, **kwargs).arguments
            connectivities = {}
            static_provider = {}
            for name, value in arguments.pop("offset_provider").items():
                if isinstance(value, common.Connectivity):
                    connectivities[name] = value
                else:
                    static_provider[name] = value
            static = (
                ("offset_provider", tuple(sorted(static_provider.items()))),
                *sorted((name, arguments.pop(name)) for name in static_names & arguments.keys()),
            )
            if (region := regions.get(static)) is None:
                region = regions[static] = _region(entry, static)
            dynamic = torch.utils._pytree.tree_map(_as_region_argument, arguments)
            token = _embedded_error.set(None)
            try:
                region(dynamic, connectivities)
            except torch._dynamo.exc.Unsupported as unsupported:
                if (error := _embedded_error.get()) is None:
                    raise
                raise error from unsupported
            finally:
                _embedded_error.reset(token)

        return run


torch_compile = TorchCompileBackend(
    name="torch_compile", allocator=cast(core_ndarray_utils.ArrayNamespace, torch)
)
