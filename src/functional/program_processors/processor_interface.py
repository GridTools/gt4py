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

Program processors are functions which take an ``iterator.ir.itir.FencilDefinition``
along with input values for the fencil. Those which execute the fencil with
the given arguments (possibly by generating code along the way) are fencil
executors. Those which generate any kind of string based on the fencil
and (optionally) input values are fencil formatters.

For more information refer to
``gt4py/docs/functional/architecture/007-Program-Processors.md``
"""
from __future__ import annotations

from typing import Callable, Protocol, TypeGuard, TypeVar, cast

from functional.iterator import ir as itir
from functional.otf import stages


OutputT = TypeVar("OutputT", covariant=True)
ProcessorKindT = TypeVar("ProcessorKindT", bound="ProgramProcessorProtocol", covariant=True)


class ProgramProcessorFunction(Protocol[OutputT]):
    def __call__(self, fencil: itir.FencilDefinition, *args, **kwargs) -> OutputT:
        ...


class ProgramProcessorProtocol(
    ProgramProcessorFunction[OutputT], Protocol[OutputT, ProcessorKindT]
):
    @property
    def kind(self) -> type[ProcessorKindT]:
        ...


class ProgramFormatter(ProgramProcessorProtocol[str, "ProgramFormatter"], Protocol):
    @property
    def kind(self) -> type[ProgramFormatter]:
        return ProgramFormatter


def program_formatter(func: ProgramProcessorFunction[str]) -> ProgramFormatter:
    """
    Turn a formatter function into a ProgramFormatter.

    Examples:
    ---------
    >>> @program_formatter
    ... def format_foo(fencil: itir.FencilDefinition, *args, **kwargs) -> str:
    ...     '''A very useless fencil formatter.'''
    ...     return "foo"

    >>> ensure_processor_kind(format_foo, ProgramFormatter)
    """
    # this operation effectively changes the type of func and that is the intention here
    func.kind = ProgramFormatter  # type: ignore[attr-defined]
    return cast(ProgramProcessorProtocol[str, ProgramFormatter], func)


class ProgramSourceGenerator(
    ProgramProcessorProtocol[stages.ProgramSource, "ProgramSourceGenerator"], Protocol
):
    @property
    def kind(self) -> type[ProgramSourceGenerator]:
        return ProgramSourceGenerator


def program_source_generator(
    func: ProgramProcessorFunction[stages.ProgramSource],
) -> ProgramSourceGenerator:
    """
    Wrap a source module generator function in a ``ProgramSourceGenerator`` instance.

    Examples:
    ---------
    >>> from functional.otf.source import Function
    >>> @program_source_generator
    ... def generate_foo(fencil: itir.FencilDefinition, *args, **kwargs) -> stages.ProgramSource:
    ...     '''A very useless fencil formatter.'''
    ...     return stages.ProgramSource(entry_point=Function(fencil.id, []), library_deps=[], source_code="foo", language="foo")

    >>> ensure_processor_kind(generate_foo, ProgramSourceGenerator)
    """
    # this operation effectively changes the type of func and that is the intention here
    func.kind = ProgramSourceGenerator  # type: ignore[attr-defined]
    return cast(ProgramProcessorProtocol[stages.ProgramSource, ProgramSourceGenerator], func)


class ProgramExecutor(ProgramProcessorProtocol[None, "ProgramExecutor"], Protocol):
    @property
    def kind(self) -> type[ProgramExecutor]:
        return ProgramExecutor


def program_executor(func: ProgramProcessorFunction[None]) -> ProgramExecutor:
    """
    Wrap an executor function in a ``ProgramFormatter`` instance.

    Examples:
    ---------
    >>> @program_executor
    ... def badly_execute(fencil: itir.FencilDefinition, *args, **kwargs) -> None:
    ...     '''A useless and incorrect fencil executor.'''
    ...     pass

    >>> ensure_processor_kind(badly_execute, ProgramExecutor)
    """
    # this operation effectively changes the type of func and that is the intention here
    func.kind = ProgramExecutor  # type: ignore[attr-defined]
    return cast(ProgramExecutor, func)


def is_processor_kind(
    obj: Callable[..., OutputT], kind: type[ProcessorKindT]
) -> TypeGuard[ProgramProcessorProtocol[OutputT, ProcessorKindT]]:
    return callable(obj) and getattr(obj, "kind", None) is kind


def ensure_processor_kind(
    obj: ProgramProcessorProtocol[OutputT, ProcessorKindT], kind: type[ProcessorKindT]
) -> None:
    if not is_processor_kind(obj, kind):
        raise RuntimeError(f"{obj} is not a {kind.__name__}!")
