# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Embedded execution of programs and field operators inside the tracing compiler of an array library.

The backends (`jax_jit.jax_jit`, `torch_compile.torch_compile`) run the Python
definition on fields of their library, inside `jax.jit` or `torch.compile`.
Fields, connectivities and the other arguments go to the compiled function; the
offset provider entries that are no connectivities (`Dimension`s of Cartesian
offsets) and the parameters named in `static_params` are static.
"""

from __future__ import annotations

import abc
import dataclasses
import pkgutil
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from gt4py.next import backend as next_backend, common
from gt4py.next.type_system import type_specifications as ts


StaticItems = tuple[tuple[str, Any], ...]


@dataclasses.dataclass(frozen=True)
class EmbeddedJitBackend(next_backend.Backend, abc.ABC):
    """Runs programs and field operators embedded, inside the tracing compiler of an array library."""

    @abc.abstractmethod
    def jit(
        self,
        function: Callable[..., None],
        static_argnames: Iterable[str] = (),
        param_names: Sequence[str] = (),
        param_types: Iterable[ts.TypeSpec] = (),
    ) -> EmbeddedJitRunner: ...


def not_compiled(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("An embedded jit backend runs the program embedded; it does not compile.")


def named_collection_dataclasses(type_: ts.TypeSpec) -> Iterable[type]:
    """The dataclasses of the named collections in `type_`, which are no pytrees by themselves."""
    if isinstance(type_, ts.NamedCollectionType):
        cls = pkgutil.resolve_name(type_.original_python_type)
        if isinstance(cls, type) and dataclasses.is_dataclass(cls):
            yield cls
    if isinstance(type_, ts.TupleType | ts.NamedCollectionType):
        for element_type in type_.types:
            yield from named_collection_dataclasses(element_type)


class EmbeddedJitRunner(abc.ABC):
    """
    One compiled embedded entry point, called like a compiled programs pool.

    Splits the arguments into the ones the compiled function takes and the static
    ones: the parameters in `static_argnames` and the offset provider entries
    that are no connectivities, both as sorted `(name, value)` items.
    """

    def __init__(
        self,
        function: Callable[..., None],
        static_argnames: frozenset[str],
        param_names: tuple[str, ...],
    ) -> None:
        self._function = function
        self._static_argnames = static_argnames
        # static parameters passed positionally are only found by name
        self._param_names = param_names if static_argnames & set(param_names) else ()

    @abc.abstractmethod
    def _run(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        connectivities: dict[str, Any],
        static_kwargs: StaticItems,
        static_provider: StaticItems,
    ) -> None: ...

    def __call__(
        self,
        *args: Any,
        offset_provider: common.OffsetProvider,
        enable_jit: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if self._static_argnames:
            if self._param_names:
                kwargs = {**dict(zip(self._param_names, args)), **kwargs}
                args = ()
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

        self._run(args, kwargs, connectivities, static_kwargs, tuple(sorted(static_provider)))
