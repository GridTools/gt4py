# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Interface for program processors.

Program processors are functions which operate on a program paired with the input
arguments for the program. Programs are represented by an ``iterator.ir.Program``
node. Program processors that execute the program with the given arguments (possibly by generating
code along the way) are program executors. Those that generate any kind of string based
on the program and (optionally) input values are program formatters.

For more information refer to
``gt4py/docs/functional/architecture/007-Program-Processors.md``
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Callable

import gt4py.next.iterator.ir as itir


class ProgramFormatter:
    @abc.abstractmethod
    def __call__(self, program: itir.Program, *args: Any, **kwargs: Any) -> str: ...


@dataclasses.dataclass(frozen=True)
class WrappedProgramFormatter(ProgramFormatter):
    formatter: Callable[..., str]

    def __call__(self, program: itir.Program, *args: Any, **kwargs: Any) -> str:
        return self.formatter(program, *args, **kwargs)


def program_formatter(func: Callable[..., str]) -> ProgramFormatter:
    """
    Turn a function that formats a program as a string into a ProgramFormatter.

    Examples:
        >>> @program_formatter
        ... def format_foo(fencil: itir.Program, *args, **kwargs) -> str:
        ...     '''A very useless fencil formatter.'''
        ...     return "foo"

        >>> isinstance(format_foo, ProgramFormatter)
        True
    """

    return WrappedProgramFormatter(formatter=func)
