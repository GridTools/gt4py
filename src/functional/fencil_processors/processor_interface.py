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

Fencil processors are functions which take an ``iterator.ir.FencilDefinition`` along with input values for the fencil.
Those which execute the fencil with the given arguments (possibly by generating code along the way) are fencil executors.
Those which generate / any kind of string based on the fencil and (optionally) input values are fencil formatters.

For more information refer to ``gt4py/docs/functional/architecture/007-Fencil-Processors.md``
"""
from __future__ import annotations

from functools import update_wrapper
from typing import Protocol, TypeVar, runtime_checkable

from functional.iterator.ir import FencilDefinition

from .source_modules import SourceModule


PROCESSOR_RETURN_T = TypeVar("PROCESSOR_RETURN_T", covariant=True)
PROCESSOR_KIND_T = TypeVar("PROCESSOR_KIND_T", bound="FencilProcessorProtocol", covariant=True)


class FencilProcessorFunction(Protocol[PROCESSOR_RETURN_T]):
    def __call__(self, fencil: FencilDefinition, *args, **kwargs) -> PROCESSOR_RETURN_T:
        ...


@runtime_checkable
class FencilProcessorProtocol(
    FencilProcessorFunction[PROCESSOR_RETURN_T], Protocol[PROCESSOR_RETURN_T, PROCESSOR_KIND_T]
):
    @classmethod
    def kind(cls) -> type[PROCESSOR_KIND_T]:
        ...


class FencilFormatter(FencilProcessorProtocol[str, "FencilFormatter"], Protocol):
    @classmethod
    def kind(cls) -> type[FencilFormatter]:
        return FencilFormatter


def fencil_formatter(
    func: FencilProcessorFunction[str],
) -> FencilProcessorProtocol[str, FencilFormatter]:
    """
    Wrap a formatter function in a ``FencilFormatter`` instance.

    Examples:
    ---------
    >>> @fencil_formatter
    ... def format_foo(fencil: FencilDefinition, *args, **kwargs) -> str:
    ...     '''A very useless fencil formatter.'''
    ...     return "foo"

    >>> ensure_processor_kind(format_foo, FencilFormatter)
    """

    class _FormatterClass(FencilFormatter):
        def __call__(self, fencil: FencilDefinition, *args, **kwargs) -> str:
            return func(fencil, *args, **kwargs)

    formatter_instance = _FormatterClass()
    update_wrapper(formatter_instance, func)
    return formatter_instance


class FencilSourceModuleGenerator(
    FencilProcessorProtocol[SourceModule, "FencilSourceModuleGenerator"], Protocol
):
    @classmethod
    def kind(cls) -> type[FencilSourceModuleGenerator]:
        return FencilSourceModuleGenerator


def fencil_source_module_generator(
    func: FencilProcessorFunction[SourceModule],
) -> FencilProcessorProtocol[SourceModule, FencilSourceModuleGenerator]:
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

    class _SourceModuleGeneratorClass(FencilSourceModuleGenerator):
        def __call__(self, fencil: FencilDefinition, *args, **kwargs) -> SourceModule:
            return func(fencil, *args, **kwargs)

    generator_instance = _SourceModuleGeneratorClass()
    update_wrapper(generator_instance, func)
    return generator_instance


class FencilExecutor(FencilProcessorProtocol[None, "FencilExecutor"], Protocol):
    @classmethod
    def kind(cls) -> type[FencilExecutor]:
        return FencilExecutor


def fencil_executor(
    func: FencilProcessorFunction[None],
) -> FencilProcessorProtocol[None, FencilExecutor]:
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

    class _ExecutorClass(FencilExecutor):
        def __call__(self, fencil: FencilDefinition, *args, **kwargs) -> None:
            func(fencil, *args, **kwargs)

    executor_instance = _ExecutorClass()
    update_wrapper(executor_instance, func)
    return executor_instance


def ensure_processor_kind(processor: FencilProcessorProtocol, kind: type) -> None:
    if not isinstance(processor, FencilProcessorProtocol):
        raise RuntimeError(f"{processor} does not fulfill {FencilProcessorProtocol.__name__}")
    if processor.kind() != kind:
        raise RuntimeError(f"{processor} is not a {kind.__name__}!")
