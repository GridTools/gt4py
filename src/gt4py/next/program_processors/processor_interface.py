# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import functools
from collections.abc import Sequence
from typing import Any, Callable, Literal, Optional, Protocol, TypeGuard, TypeVar, cast

import gt4py._core.definitions as core_defs
import gt4py.next.allocators as next_allocators
import gt4py.next.iterator.ir as itir


OutputT = TypeVar("OutputT", covariant=True)
ProcessorKindT = TypeVar("ProcessorKindT", bound="ProgramProcessor", covariant=True)


class ProgramProcessorCallable(Protocol[OutputT]):
    def __call__(self, program: itir.FencilDefinition, *args, **kwargs) -> OutputT:
        ...


class ProgramProcessor(ProgramProcessorCallable[OutputT], Protocol[OutputT, ProcessorKindT]):
    @property
    def kind(self) -> type[ProcessorKindT]:
        ...


class ProgramFormatter(ProgramProcessor[str, "ProgramFormatter"], Protocol):
    @property
    def kind(self) -> type[ProgramFormatter]:
        return ProgramFormatter


def make_program_processor(
    func: ProgramProcessorCallable[OutputT],
    kind: type[ProcessorKindT],
    *,
    name: Optional[str] = None,
    accept_args: None | int | Literal["all"] = "all",
    accept_kwargs: None | Sequence[str] | Literal["all"] = "all",
) -> ProgramProcessor[OutputT, ProcessorKindT]:
    """
    Create a program processor from a callable function.

    Args:
        func: The callable function to be wrapped as a program processor.
        kind: The type of the processor.
        name: The name of the processor.
        accept_args: The number of positional arguments to accept, or "all" to accept all.
        accept_kwargs: The names of the keyword arguments to accept, or "all" to accept all.

    Returns:
        A program processor that wraps the given function.

    Raises:
        ValueError: If the value of `accept_args` or `accept_kwargs` is invalid.
    """
    args_filter: Callable[[Sequence], Sequence]
    if accept_args is None:
        args_filter = lambda args: ()  # noqa: E731  # use def instead of named lambdas
    elif accept_args == "all":
        args_filter = lambda args: args  # noqa: E731
    elif isinstance(accept_args, int):
        if accept_args < 0:
            raise ValueError(
                f"Number of accepted arguments cannot be a negative number ({accept_args})"
            )
        args_filter = lambda args: args[:accept_args]  # type: ignore[misc] # noqa: E731
    else:
        raise ValueError(f"Invalid ({accept_args}) accept_args value")

    filtered_kwargs: Callable[[dict[str, Any]], dict[str, Any]]
    if accept_kwargs is None:
        filtered_kwargs = lambda kwargs: {}  # noqa: E731  # use def instead of named lambdas
    elif accept_kwargs == "all":  # don't swap with 'isinstance(..., Sequence)'
        filtered_kwargs = lambda kwargs: kwargs  # noqa: E731
    elif isinstance(accept_kwargs, Sequence):
        if not all(isinstance(a, str) for a in accept_kwargs):
            raise ValueError(f"Provided invalid list of keyword argument names ({accept_args})")
        filtered_kwargs = lambda kwargs: {  # noqa: E731
            key: value for key, value in kwargs.items() if key in accept_kwargs  # type: ignore[operator]  # key in accept_kwargs
        }
    else:
        raise ValueError(f"Invalid ({accept_kwargs}) 'accept_kwargs' value")

    @functools.wraps(func)
    def _wrapper(program: itir.FencilDefinition, *args, **kwargs) -> OutputT:
        return func(program, *args_filter(args), **filtered_kwargs(kwargs))

    if name is not None:
        _wrapper.__name__ = name

    # this operation effectively changes the type of the returned object,
    # which is the intention here
    _wrapper.kind = kind  # type: ignore[attr-defined]

    return cast(ProgramProcessor[OutputT, ProcessorKindT], _wrapper)


def program_formatter(
    func: ProgramProcessorCallable[str],
    *,
    name: Optional[str] = None,
    accept_args: None | int | Literal["all"] = "all",
    accept_kwargs: Sequence[str] | None | Literal["all"] = "all",
) -> ProgramFormatter:
    """
    Turn a function that formats a program as a string into a ProgramFormatter.

    Examples:
        >>> @program_formatter
        ... def format_foo(fencil: itir.FencilDefinition, *args, **kwargs) -> str:
        ...     '''A very useless fencil formatter.'''
        ...     return "foo"

        >>> ensure_processor_kind(format_foo, ProgramFormatter)
    """
    return make_program_processor(
        func,
        ProgramFormatter,  # type: ignore[type-abstract]  # ProgramFormatter is abstract
        name=name,
        accept_args=accept_args,
        accept_kwargs=accept_kwargs,
    )


class ProgramExecutor(ProgramProcessor[None, "ProgramExecutor"]):
    @property
    def kind(self) -> type[ProgramExecutor]:
        return ProgramExecutor


def program_executor(
    func: ProgramProcessorCallable[None],
    *,
    name: Optional[str] = None,
    accept_args: None | int | Literal["all"] = "all",
    accept_kwargs: Sequence[str] | None | Literal["all"] = "all",
) -> ProgramExecutor:
    """
    Turn a function that executes a program into a ``ProgramExecutor``.

    Examples:
        >>> @program_executor
        ... def badly_execute(fencil: itir.FencilDefinition, *args, **kwargs) -> None:
        ...     '''A useless and incorrect fencil executor.'''
        ...     pass

        >>> ensure_processor_kind(badly_execute, ProgramExecutor)
    """
    return cast(
        ProgramExecutor,
        make_program_processor(
            func,
            ProgramExecutor,
            name=name,
            accept_args=accept_args,
            accept_kwargs=accept_kwargs,
        ),
    )


def is_processor_kind(
    obj: Callable[..., OutputT], kind: type[ProcessorKindT]
) -> TypeGuard[ProgramProcessor[OutputT, ProcessorKindT]]:
    return callable(obj) and getattr(obj, "kind", None) is kind


def ensure_processor_kind(
    obj: ProgramProcessor[OutputT, ProcessorKindT], kind: type[ProcessorKindT]
) -> None:
    if not is_processor_kind(obj, kind):
        raise TypeError(f"{obj} is not a {kind.__name__}!")


class ProgramBackend(
    ProgramProcessor[None, "ProgramExecutor"],
    next_allocators.FieldBufferAllocatorFactoryProtocol[core_defs.DeviceTypeT],
    Protocol[core_defs.DeviceTypeT],
):
    ...


def is_program_backend(obj: Callable) -> TypeGuard[ProgramBackend]:
    return is_processor_kind(
        obj, ProgramExecutor  # type: ignore[type-abstract]  # ProgramExecutor is abstract
    ) and next_allocators.is_field_allocator_factory(obj)


def is_program_backend_for(
    obj: Callable, device: core_defs.DeviceTypeT
) -> TypeGuard[ProgramBackend[core_defs.DeviceTypeT]]:
    return is_processor_kind(
        obj, ProgramExecutor  # type: ignore[type-abstract]  # ProgramExecutor is abstract
    ) and next_allocators.is_field_allocator_factory_for(obj, device)
