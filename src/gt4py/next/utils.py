# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
def flatten_nested_tuple(
    value: tuple[
        _T | tuple, ...
    ],  # `_T` omitted on purpose as type of `value`, to properly deduce `_T` on the user-side
) -> tuple[_T, ...]:
    if isinstance(value, tuple):
        return sum((flatten_nested_tuple(v) for v in value), start=())  # type: ignore[arg-type] # cannot properly express nesting
    else:
        return (value,)


@overload
def tree_map(
    fun: Callable[_P, _R],
    *,
    collection_type: type | tuple[type, ...] = tuple,
    result_collection_constructor: Optional[type | Callable] = None,
) -> Callable[..., _R | tuple[_R | tuple, ...]]: ...


@overload
def tree_map(
    *,
    collection_type: type | tuple[type, ...] = tuple,
    result_collection_constructor: Optional[type | Callable] = None,
) -> Callable[
    [Callable[_P, _R]], Callable[..., Any]
]: ...  # TODO(havogt): if result_collection_constructor is Callable, improve typing


def tree_map(
    fun: Optional[Callable[_P, _R]] = None,
    *,
    collection_type: type | tuple[type, ...] = tuple,
    result_collection_constructor: Optional[type | Callable] = None,
) -> Callable[..., _R | tuple[_R | tuple, ...]] | Callable[[Callable[_P, _R]], Callable[..., Any]]:
    """
    Apply `fun` to each entry of (possibly nested) collections (by default `tuple`s).

    Args:
        fun: Function to apply to each entry of the collection.
        collection_type: Type of the collection to be traversed. Can be a single type or a tuple of types.
        result_collection_constructor: Type of the collection to be returned. If `None` the same type as `collection_type` is used.

    Examples:
        >>> tree_map(lambda x: x + 1)(((1, 2), 3))
        ((2, 3), 4)

        >>> tree_map(lambda x, y: x + y)(((1, 2), 3), ((4, 5), 6))
        ((5, 7), 9)

        >>> tree_map(collection_type=list)(lambda x: x + 1)([[1, 2], 3])
        [[2, 3], 4]

        >>> tree_map(collection_type=list, result_collection_constructor=tuple)(lambda x: x + 1)(
        ...     [[1, 2], 3]
        ... )
        ((2, 3), 4)

        >>> @tree_map
        ... def impl(x):
        ...     return x + 1
        >>> impl(((1, 2), 3))
        ((2, 3), 4)
    """

    if result_collection_constructor is None:
        if isinstance(collection_type, tuple):
            raise TypeError(
                "tree_map() requires `result_collection_constructor` when `collection_type` is a tuple of types."
            )
        result_collection_constructor = collection_type

    if fun:

        @functools.wraps(fun)
        def impl(*args: Any | tuple[Any | tuple, ...]) -> _R | tuple[_R | tuple, ...]:
            if isinstance(args[0], collection_type):
                assert all(
                    isinstance(arg, collection_type) and len(args[0]) == len(arg) for arg in args
                )
                assert result_collection_constructor is not None
                return result_collection_constructor(impl(*arg) for arg in zip(*args))

            return fun(  # type: ignore[call-arg, misc] # mypy not smart enough
                *cast(_P.args, args)
            )  # mypy doesn't understand that `args` at this point is of type `_P.args`

        return impl
    else:
        return functools.partial(
            tree_map,
            collection_type=collection_type,
            result_collection_constructor=result_collection_constructor,
        )
