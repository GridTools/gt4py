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

import functools
from typing import Any, Callable, ClassVar, ParamSpec, TypeGuard, TypeVar, cast

import numpy as np

from gt4py.next import common


class RecursionGuard:
    """
    Context manager to guard against inifinite recursion.

    >>> def foo(i):
    ...    with RecursionGuard(i):
    ...        if i % 2 == 0:
    ...            foo(i)
    ...    return i
    >>> foo(3)
    3
    >>> foo(2)  # doctest:+ELLIPSIS
    Traceback (most recent call last):
        ...
    gt4py.next.utils.RecursionGuard.RecursionDetected
    """

    guarded_objects: ClassVar[set[int]] = set()

    obj: Any

    class RecursionDetected(Exception):
        pass

    def __init__(self, obj: Any):
        self.obj = obj

    def __enter__(self):
        if id(self.obj) in self.guarded_objects:
            raise self.RecursionDetected()
        self.guarded_objects.add(id(self.obj))

    def __exit__(self, *exc):
        self.guarded_objects.remove(id(self.obj))


_T = TypeVar("_T")

_P = ParamSpec("_P")
_R = TypeVar("_R")


def is_tuple_of(v: Any, t: type[_T]) -> TypeGuard[tuple[_T, ...]]:
    return isinstance(v, tuple) and all(isinstance(e, t) for e in v)


def tree_map(fun: Callable[_P, _R]) -> Callable[..., _R | tuple[_R | tuple, ...]]:
    """Apply `fun` to each entry of (possibly nested) tuples.

    Examples:
        >>> tree_map(lambda x: x + 1)(((1, 2), 3))
        ((2, 3), 4)

        >>> tree_map(lambda x, y: x + y)(((1, 2), 3), ((4, 5), 6))
        ((5, 7), 9)
    """

    @functools.wraps(fun)
    def impl(*args: Any | tuple[Any | tuple, ...]) -> _R | tuple[_R | tuple, ...]:
        if isinstance(args[0], tuple):
            assert all(isinstance(arg, tuple) and len(args[0]) == len(arg) for arg in args)
            return tuple(impl(*arg) for arg in zip(*args))

        return fun(
            *cast(_P.args, args)
        )  # mypy doesn't understand that `args` at this point is of type `_P.args`

    return impl


# TODO(havogt): consider moving to module like `field_utils`
@tree_map
def asnumpy(field: common.Field | np.ndarray) -> np.ndarray:
    return field.asnumpy() if common.is_field(field) else field  # type: ignore[return-value] # mypy doesn't understand the condition
