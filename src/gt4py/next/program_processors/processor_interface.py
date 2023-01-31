# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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
Interface for program processors.

Program processors are functions which operate on a program paired with the input
arguments for the program. Programs are represented by an ``iterator.ir.itir.FencilDefinition``
node. Program processors that execute the program with the given arguments (possibly by generating
code along the way) are program executors. Those that generate any kind of string based
on the program and (optionally) input values are program formatters.

For more information refer to
``gt4py/docs/functional/architecture/007-Program-Processors.md``
"""
from __future__ import annotations

from typing import Callable, Protocol, TypeGuard, TypeVar, cast

from gt4py.next.iterator import ir as itir


OutputT = TypeVar("OutputT", covariant=True)
ProcessorKindT = TypeVar("ProcessorKindT", bound="ProgramProcessor", covariant=True)


class ProgramProcessorFunction(Protocol[OutputT]):
    def __call__(self, program: itir.FencilDefinition, *args, **kwargs) -> OutputT:
        ...


class ProgramProcessor(ProgramProcessorFunction[OutputT], Protocol[OutputT, ProcessorKindT]):
    @property
    def kind(self) -> type[ProcessorKindT]:
        ...


class ProgramFormatter(ProgramProcessor[str, "ProgramFormatter"], Protocol):
    @property
    def kind(self) -> type[ProgramFormatter]:
        return ProgramFormatter


def program_formatter(func: ProgramProcessorFunction[str]) -> ProgramFormatter:
    """
    Turn a function that formats a program as a string into a ProgramFormatter.

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
    return cast(ProgramProcessor[str, ProgramFormatter], func)


class ProgramExecutor(ProgramProcessor[None, "ProgramExecutor"], Protocol):
    @property
    def kind(self) -> type[ProgramExecutor]:
        return ProgramExecutor


def program_executor(func: ProgramProcessorFunction[None]) -> ProgramExecutor:
    """
    Turn a function that executes a program into a ``ProgramExecutor``.

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
) -> TypeGuard[ProgramProcessor[OutputT, ProcessorKindT]]:
    return callable(obj) and getattr(obj, "kind", None) is kind


def ensure_processor_kind(
    obj: ProgramProcessor[OutputT, ProcessorKindT], kind: type[ProcessorKindT]
) -> None:
    if not is_processor_kind(obj, kind):
        raise TypeError(f"{obj} is not a {kind.__name__}!")
