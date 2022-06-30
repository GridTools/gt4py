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
from dataclasses import dataclass
from functools import update_wrapper
from typing import Protocol

from functional.iterator import ir as itir


class FencilFormatterFunction(Protocol):
    def __call__(self, fencil: itir.FencilDefinition, *args, **kwargs) -> str:
        ...


class FencilExecutorFunction(Protocol):
    def __call__(self, fencil: itir.FencilDefinition, *args, **kwargs) -> None:
        ...


@dataclass
class FencilFormatter:
    """Wrap a raw formatter function and make it type-checkable as a fencil formatter at runtime."""

    formatter_function: FencilFormatterFunction

    def __call__(self, fencil: itir.FencilDefinition, *args, **kwargs) -> str:
        return self.formatter_function(fencil, *args, **kwargs)


@dataclass
class FencilExecutor:
    """Wrap a raw executor function and make it type-checkable as a fencil executor at runtime."""

    executor_function: FencilExecutorFunction

    def __call__(self, fencil: itir.FencilDefinition, *args, **kwargs) -> None:
        self.executor_function(fencil, *args, **kwargs)


def fencil_formatter(func: FencilFormatterFunction) -> FencilFormatter:
    """
    Wrap a formatter function in a ``FencilFormatter`` instance.

    Examples:
    ---------
    >>> @fencil_formatter
    ... def format_foo(fencil: itir.FencilDefinition, *args, **kwargs) -> str:
    ...     '''A very useless fencil formatter.'''
    ...     return "foo"

    >>> assert isinstance(format_foo, FencilFormatter)
    """
    wrapper = FencilFormatter(formatter_function=func)
    update_wrapper(wrapper, func)
    return wrapper


def fencil_executor(func: FencilExecutorFunction) -> FencilExecutor:
    """
    Wrap an executor function in a ``FencilFormatter`` instance.

    Examples:
    ---------
    >>> @fencil_executor
    ... def badly_execute(fencil: itir.FencilDefinition, *args, **kwargs) -> None:
    ...     '''A useless and incorrect fencil executor.'''
    ...     pass

    >>> assert isinstance(badly_execute, FencilExecutor)
    """
    wrapper = FencilExecutor(executor_function=func)
    update_wrapper(wrapper, func)
    return wrapper


def ensure_formatter(formatter: FencilFormatter) -> None:
    """Check that a formatter is an instance of ``FencilFormatter`` and raise an error if not."""
    if not isinstance(formatter, FencilFormatter):
        raise RuntimeError(f"{formatter} is not a fencil formatter!")


def ensure_executor(executor: FencilExecutor) -> None:
    """Check that an executor is an instance of ``FencilExecutor`` and raise an error if not."""
    if not isinstance(executor, FencilExecutor):
        raise RuntimeError(f"{executor} is not a fencil executor!")
