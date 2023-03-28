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

from __future__ import annotations

import dataclasses
from typing import Any, Callable, ClassVar, Generic, Protocol, TypeVar


StartT = TypeVar("StartT")
StartT_contra = TypeVar("StartT_contra", contravariant=True)
EndT = TypeVar("EndT")
EndT_co = TypeVar("EndT_co", covariant=True)
NewEndT = TypeVar("NewEndT")
IntermediateT = TypeVar("IntermediateT")
HashT = TypeVar("HashT")


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


WFT = TypeVar("WFT", bound=Workflow)


def replace(workflow: WFT, **kwargs: Any) -> WFT:
    """Make a copy of the workflow with changed configuration or subworkflows."""
    return dataclasses.replace(workflow, **kwargs)


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
class NamedStepSequence(Generic[StartT, EndT]):
    """
    Workflow with linear succession of named steps.

    Examples:
    ---------
    >>> import dataclasses

    >>> def parse(x: str) -> int:
    ...    return int(x)

    >>> def plus_half(x: int) -> float:
    ...    return x + 0.5

    >>> def stringify(x: float) -> str:
    ...    return str(x)

    >>> @dataclasses.dataclass(frozen=True)
    ... class ParseOpPrint(NamedStepSequence[str, str]):
    ...    parse: Workflow[str, int]
    ...    op: Workflow[int, float]
    ...    print: Workflow[float, str]
    ...    step_order = ["parse", "op", "print"]

    >>> pop = ParseOpPrint(
    ...    parse=parse,
    ...    op=plus_half,
    ...    print=stringify
    ... )

    >>> pop(73)
    '73.5'

    >>> def plus_tenth(x: int) -> float:
    ...   return x + 0.1

    >>> pop.replace(op=plus_tenth)(73)
    '73.1'
    """

    step_order: ClassVar[list[str]]

    def __call__(self, inp: StartT) -> EndT:
        """Compose the steps in the order defined in the `.step_order` class attribute."""
        step_result: Any = inp
        for step in [getattr(self, s) for s in self.step_order]:
            step_result = step(step_result)
        return step_result

    def chain(self, step: Workflow[EndT, NewEndT]) -> CombinedStep[StartT, EndT, NewEndT]:
        return CombinedStep(first=self, second=step)

    def replace(self, **kwargs: Any) -> NamedStepSequence[StartT, EndT]:
        return dataclasses.replace(self, **kwargs)


@dataclasses.dataclass(frozen=True)
class CombinedStep(NamedStepSequence, Generic[StartT, IntermediateT, EndT]):
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
    step_order = ["first", "second"]


@dataclasses.dataclass(frozen=True)
class CachedStep(Step[StartT, EndT], Generic[StartT, EndT, HashT]):
    """
    Cached workflow of single input callables.

    Examples:
    ---------
    >>> def heavy_computation(x: int) -> int:
    ...    print("This might take a while...")
    ...    return x

    >>> cached_step = CachedStep(step=heavy_computation)

    >>> cached_step(42)
    This might take a while...
    42

    The next invocation for the same argument will be cached:
    >>> cached_step(42)
    42

    >>> cached_step(1)
    This might take a while...
    1
    """

    hash_function: Callable[[StartT], HashT] = dataclasses.field(default=hash)  # type: ignore[assignment]

    _cache: dict[HashT, EndT] = dataclasses.field(repr=False, init=False, default_factory=dict)

    def __call__(self, inp: StartT) -> EndT:
        """Run the step only if the input is not cached, else return from cache."""
        hash_ = self.hash_function(inp)
        try:
            result = self._cache[hash_]
        except KeyError:
            result = self._cache[hash_] = self.step(inp)
        return result
