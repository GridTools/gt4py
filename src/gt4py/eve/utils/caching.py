# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Caching decorators."""

from __future__ import annotations

import functools
import typing
from collections.abc import Callable
from typing import Any, Literal, Optional, ParamSpec, TypeVar, Union, overload

from .hashing import EqualityBy, HashableBy


_P = ParamSpec("_P")
_T = TypeVar("_T")


@overload
def optional_lru_cache(
    func: Literal[None] = None, *, maxsize: Optional[int] = 128, typed: bool = False
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


@overload
def optional_lru_cache(
    func: Callable[_P, _T], *, maxsize: Optional[int] = 128, typed: bool = False
) -> Callable[_P, _T]: ...


def optional_lru_cache(
    func: Optional[Callable[_P, _T]] = None, *, maxsize: Optional[int] = 128, typed: bool = False
) -> Union[Callable[_P, _T], Callable[[Callable[_P, _T]], Callable[_P, _T]]]:
    """Wrap :func:`functools.lru_cache` to fall back to the original function if arguments are not hashable.

    Examples:
        >>> @optional_lru_cache(typed=True)
        ... def func(a, b):
        ...     print(f"Inside func({a}, {b})")
        ...     return a + b
        >>> print(func(1, 3))
        Inside func(1, 3)
        4
        >>> print(func(1, 3))
        4
        >>> print(func([1], [3]))
        Inside func([1], [3])
        [1, 3]
        >>> print(func([1], [3]))
        Inside func([1], [3])
        [1, 3]

    Notes:
        Based on :func:`typing._tp_cache`.
    """

    def _decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
        if maxsize is None and not typed:
            cached = functools.cache(func)
        else:
            cached = functools.lru_cache(maxsize=maxsize, typed=typed)(func)

        @functools.wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Any:
            try:
                return cached(*args, **kwargs)
            except TypeError as error:
                if error.args and error.args[0].startswith("unhashable"):
                    # Catch errors due to non-hashable arguments and fallback to original function
                    return func(*args, **kwargs)
                else:
                    raise error

        return inner

    return _decorator(func) if func is not None else _decorator


# TODO(egparedes): it would be more efficient to implement the caching logic
# here instead of relying on `functools.lru_cache` and wrapping/unwrapping the
# arguments.
def lru_cache(
    func: Optional[Callable[_P, _T]] = None,
    *,
    key: Optional[Callable[_P, int]] = None,
    maxsize: Optional[int] = 128,
    typed: bool = False,
) -> Union[Callable[_P, _T], Callable[[Callable[_P, _T]], Callable[_P, _T]]]:
    """
    Wrap :func:`functools.lru_cache` but allow customizing the cache key.

    Be careful, with custom `key` functions, `key(obj1) == key(obj2)` automatically
    implies `obj1 == obj2`, i.e. they are considered equal.

    >>> @lru_cache(key=id)
    ... def func(x):
    ...     print("called")
    ...     return x

    >>> obj = object()
    >>> func(obj) is obj
    called
    True
    >>> func(obj) is obj
    True
    """

    def _decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
        if key:
            if typed:
                raise ValueError("Cannot use both 'key' and 'typed'")

            @functools.lru_cache(maxsize=maxsize, typed=False)
            def cached_func(*args: HashableBy, **kwargs: HashableBy) -> _T:
                return func(*(arg.value for arg in args), **{k: v.value for k, v in kwargs.items()})

            @functools.wraps(func)
            def inner(*args, **kwargs):  # type: ignore[no-untyped-def]  # cast below restores type info
                return cached_func(
                    *(EqualityBy(key, arg) for arg in args),
                    **{k: EqualityBy(key, arg) for k, arg in kwargs.items()},
                )

            inner.cache_parameters = cached_func.cache_parameters  # type: ignore[attr-defined]  # mypy not aware of functools.lru_cache behavior
            inner.cache_info = cached_func.cache_info  # type: ignore[attr-defined]  # mypy not aware of functools.lru_cache behavior
            inner.cache_clear = cached_func.cache_clear  # type: ignore[attr-defined]  # mypy not aware of functools.cache_clear behavior

            return typing.cast(Callable[_P, _T], inner)

        return typing.cast(
            Callable[_P, _T], functools.lru_cache(maxsize=maxsize, typed=typed)(func)
        )

    return _decorator(func) if func is not None else _decorator
