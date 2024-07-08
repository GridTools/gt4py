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
from typing import Any, Callable, ClassVar, Optional, ParamSpec, TypeGuard, TypeVar, cast, overload


class RecursionGuard:
    """
    Context manager to guard against inifinite recursion.

    >>> def foo(i):
    ...     with RecursionGuard(i):
    ...         if i % 2 == 0:
    ...             foo(i)
    ...     return i
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

    def __enter__(self) -> None:
        if id(self.obj) in self.guarded_objects:
            raise self.RecursionDetected()
        self.guarded_objects.add(id(self.obj))

    def __exit__(self, *exc: Any) -> None:
        self.guarded_objects.remove(id(self.obj))


_T = TypeVar("_T")
_P = ParamSpec("_P")
_R = TypeVar("_R")


def is_tuple_of(v: Any, t: type[_T]) -> TypeGuard[tuple[_T, ...]]:
    return isinstance(v, tuple) and all(isinstance(e, t) for e in v)


# TODO(havogt): remove flatten duplications in the whole codebase
def flatten_nested_tuple(value: tuple[_T | tuple, ...]) -> tuple[_T, ...]:
    if isinstance(value, tuple):
        return sum((flatten_nested_tuple(v) for v in value), start=())  # type: ignore[arg-type] # cannot properly express nesting
    else:
        return (value,)


@overload
def tree_map(fun: Callable[_P, _R], /) -> Callable[..., _R | tuple[_R | tuple, ...]]: ...


@overload
def tree_map(
    *, collection_type: type | tuple[type, ...], result_collection_type: Optional[type] = None
) -> Callable[[Callable[_P, _R]], Callable[..., _R | tuple[_R | tuple, ...]]]: ...


def tree_map(
    *args: Callable[_P, _R],
    collection_type: type | tuple[type, ...] = tuple,
    result_collection_type: Optional[type] = None,
) -> (
    Callable[..., _R | tuple[_R | tuple, ...]]
    | Callable[[Callable[_P, _R]], Callable[..., _R | tuple[_R | tuple, ...]]]
):
    """
    Apply `fun` to each entry of (possibly nested) collections (by default `tuple`s).

    Args:
        fun: Function to apply to each entry of the collection.
        collection_type: Type of the collection to be traversed. Can be a single type or a tuple of types.
        result_collection_type: Type of the collection to be returned. If `None` the same type as `collection_type` is used.

    Examples:
        >>> tree_map(lambda x: x + 1)(((1, 2), 3))
        ((2, 3), 4)

        >>> tree_map(lambda x, y: x + y)(((1, 2), 3), ((4, 5), 6))
        ((5, 7), 9)

        >>> tree_map(collection_type=list)(lambda x: x + 1)([[1, 2], 3])
        [[2, 3], 4]

        >>> tree_map(collection_type=list, result_collection_type=tuple)(lambda x: x + 1)([[1, 2], 3])
        ((2, 3), 4)
    """

    if result_collection_type is None:
        if isinstance(collection_type, tuple):
            raise TypeError(
                "tree_map() requires `result_collection_type` when `collection_type` is a tuple."
            )
        result_collection_type = collection_type

    if len(args) == 1:
        fun = args[0]

        @functools.wraps(fun)
        def impl(*args: Any | tuple[Any | tuple, ...]) -> _R | tuple[_R | tuple, ...]:
            if isinstance(args[0], collection_type):
                assert all(
                    isinstance(arg, collection_type) and len(args[0]) == len(arg) for arg in args
                )
                assert result_collection_type is not None
                return result_collection_type(impl(*arg) for arg in zip(*args))

            return fun(
                *cast(_P.args, args)
            )  # mypy doesn't understand that `args` at this point is of type `_P.args`

        return impl
    if len(args) == 0:
        return functools.partial(
            tree_map,
            collection_type=collection_type,
            result_collection_type=result_collection_type,
        )
    raise TypeError(
        "tree_map() can be used as decorator with optional kwarg `collection_type` and `result_collection_type`."
    )
