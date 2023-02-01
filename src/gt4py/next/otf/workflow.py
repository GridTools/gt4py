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

from __future__ import annotations

import dataclasses
from typing import Generic, Protocol, TypeVar


StartT = TypeVar("StartT")
StartT_contra = TypeVar("StartT_contra", contravariant=True)
EndT = TypeVar("EndT")
EndT_co = TypeVar("EndT_co", covariant=True)
NewEndT = TypeVar("NewEndT")
IntermediateT = TypeVar("IntermediateT")


def make_step(function: Workflow[StartT, EndT]) -> Step[StartT, EndT]:
    """
    Wrap a function in the workflow step convenience wrapper.

    Examples:
    ---------
    >>> @make_step
    ... def times_two(x: int) -> int:
    ...    return x * 2

    >>> def stringify(x: int) -> str:
    ...    return str(x)

    >>> # create a workflow int -> int -> str
    >>> times_two.chain(stringify)(3)
    '6'
    """
    return Step(function)


class Workflow(Protocol[StartT_contra, EndT_co]):
    """
    Workflow protocol.

    Anything that implements this interface can be a workflow of one or more steps.
    - callable
    - take a single input argument
    """

    def __call__(self, inp: StartT_contra) -> EndT_co:
        ...


@dataclasses.dataclass(frozen=True)
class Step(Generic[StartT, EndT]):
    """
    Workflow step convenience wrapper.

    Can wrap any callable which implements the workflow step protocol,
    adds the .chain(other_step) method for convenience.

    Examples:
    ---------
    >>> def times_two(x: int) -> int:
    ...    return x * 2

    >>> def stringify(x: int) -> str:
    ...    return str(x)

    >>> # create a workflow int -> int -> str
    >>> Step(times_two).chain(stringify)(3)
    '6'
    """

    step: Workflow[StartT, EndT]

    def __call__(self, inp: StartT) -> EndT:
        return self.step(inp)

    def chain(self, step: Workflow[EndT, NewEndT]) -> CombinedStep[StartT, EndT, NewEndT]:
        return CombinedStep(first=self.step, second=step)


@dataclasses.dataclass(frozen=True)
class CombinedStep(Generic[StartT, IntermediateT, EndT]):
    """
    Composable workflow of single input callables.

    Examples:
    ---------
    >>> def plus_one(x: int) -> int:
    ...    return x + 1

    >>> def plus_half(x: int) -> float:
    ...    return x + 0.5

    >>> def stringify(x: float) -> str:
    ...    return str(x)

    >>> CombinedStep(  # workflow (int -> float -> str)
    ...    first=CombinedStep(  # workflow (int -> int -> float)
    ...        first=plus_one,
    ...        second=plus_half
    ...    ),
    ...    second=stringify
    ... )(73)
    '74.5'

    >>> # is exactly equivalent to
    >>> CombinedStep(first=plus_one, second=plus_half).chain(stringify)(73)
    '74.5'

    """

    first: Workflow[StartT, IntermediateT]
    second: Workflow[IntermediateT, EndT]

    def __call__(self, inp: StartT) -> EndT:
        return self.second(self.first(inp))

    def chain(self, step: Workflow[EndT, NewEndT]) -> CombinedStep[StartT, EndT, NewEndT]:
        return CombinedStep(first=self, second=step)
