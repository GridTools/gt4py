# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Interface for fencil processors.

Fencil processors are functions which take an ``iterator.ir.FencilDefinition``
along with input values for the fencil. Those which execute the fencil with
the given arguments (possibly by generating code along the way) are fencil
executors. Those which generate any kind of string based on the fencil
and (optionally) input values are fencil formatters.

For more information refer to
``gt4py/docs/functional/architecture/007-Fencil-Processors.md``
"""
from __future__ import annotations

from typing import Callable, Protocol, TypeGuard, TypeVar, cast

from functional.iterator.ir import FencilDefinition

from .source_modules import SourceModule


OutputT = TypeVar("OutputT", covariant=True)
ProcessorKindT = TypeVar("ProcessorKindT", bound="FencilProcessorProtocol", covariant=True)


class FencilProcessorFunction(Protocol[OutputT]):
    def __call__(self, fencil: FencilDefinition, *args, **kwargs) -> OutputT:
        ...


class FencilProcessorProtocol(FencilProcessorFunction[OutputT], Protocol[OutputT, ProcessorKindT]):
    @property
    def kind(self) -> type[ProcessorKindT]:
        ...


class FencilFormatter(FencilProcessorProtocol[str, "FencilFormatter"], Protocol):
    @property
    def kind(self) -> type[FencilFormatter]:
        return FencilFormatter


def fencil_formatter(func: FencilProcessorFunction[str]) -> FencilFormatter:
    """
    Turn a formatter function into a FencilFormatter.

    Examples:
    ---------
    >>> @fencil_formatter
    ... def format_foo(fencil: FencilDefinition, *args, **kwargs) -> str:
    ...     '''A very useless fencil formatter.'''
    ...     return "foo"

    >>> ensure_processor_kind(format_foo, FencilFormatter)
    """
    # this operation effectively changes the type of func and that is the intention here
    func.kind = FencilFormatter  # type: ignore[attr-defined]
    return cast(FencilProcessorProtocol[str, FencilFormatter], func)


class FencilSourceModuleGenerator(
    FencilProcessorProtocol[SourceModule, "FencilSourceModuleGenerator"], Protocol
):
    @property
    def kind(self) -> type[FencilSourceModuleGenerator]:
        return FencilSourceModuleGenerator


def fencil_source_module_generator(
    func: FencilProcessorFunction[SourceModule],
) -> FencilSourceModuleGenerator:
    """
    Wrap a source module generator function in a ``FencilSourceModuleGenerator`` instance.

    Examples:
    ---------
    >>> from .source_modules.source_modules import Function
    >>> @fencil_source_module_generator
    ... def generate_foo(fencil: FencilDefinition, *args, **kwargs) -> SourceModule:
    ...     '''A very useless fencil formatter.'''
    ...     return SourceModule(entry_point=Function(fencil.id, []), library_deps=[], source_code="foo", language="foo")

    >>> ensure_processor_kind(generate_foo, FencilSourceModuleGenerator)
    """
    # this operation effectively changes the type of func and that is the intention here
    func.kind = FencilSourceModuleGenerator  # type: ignore[attr-defined]
    return cast(FencilProcessorProtocol[SourceModule, FencilSourceModuleGenerator], func)


class FencilExecutor(FencilProcessorProtocol[None, "FencilExecutor"], Protocol):
    @property
    def kind(self) -> type[FencilExecutor]:
        return FencilExecutor


def fencil_executor(func: FencilProcessorFunction[None]) -> FencilExecutor:
    """
    Wrap an executor function in a ``FencilFormatter`` instance.

    Examples:
    ---------
    >>> @fencil_executor
    ... def badly_execute(fencil: FencilDefinition, *args, **kwargs) -> None:
    ...     '''A useless and incorrect fencil executor.'''
    ...     pass

    >>> ensure_processor_kind(badly_execute, FencilExecutor)
    """
    # this operation effectively changes the type of func and that is the intention here
    func.kind = FencilExecutor  # type: ignore[attr-defined]
    return cast(FencilExecutor, func)


def is_processor_kind(
    obj: Callable[..., OutputT], kind: type[ProcessorKindT]
) -> TypeGuard[FencilProcessorProtocol[OutputT, ProcessorKindT]]:
    return callable(obj) and getattr(obj, "kind", None) is kind


def ensure_processor_kind(
    obj: FencilProcessorProtocol[OutputT, ProcessorKindT], kind: type[ProcessorKindT]
) -> None:
    if not is_processor_kind(obj, kind):
        raise RuntimeError(f"{obj} is not a {kind.__name__}!")
