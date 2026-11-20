# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for building and composing callables."""

from __future__ import annotations

import dataclasses
import functools
import operator
from collections.abc import Callable, Iterable
from typing import Any, Generic, Literal, Optional, ParamSpec, TypeVar, Union, overload

from boltons.iterutils import is_collection

from ..type_definitions import NOTHING
from ..xtyping import ArgsOnlyCallable


_P = ParamSpec("_P")
_S = TypeVar("_S")
_T = TypeVar("_T")


def isinstancechecker(type_info: Union[type[Any], Iterable[type[Any]]]) -> Callable[[Any], bool]:
    """Return a callable object that checks if operand is an instance of `type_info`.

    Examples:
        >>> checker = isinstancechecker((int, str))
        >>> checker(3)
        True
        >>> checker("3")
        True
        >>> checker(3.3)
        False

    """
    types: tuple[type[Any], ...] = tuple()
    if isinstance(type_info, type):
        types = (type_info,)
    elif not isinstance(type_info, tuple) and is_collection(type_info):
        types = tuple(type_info)
    else:
        types = type_info  # type:ignore  # it is checked at run-time

    if not isinstance(types, tuple) or not all(isinstance(t, type) for t in types):
        raise ValueError(f"Invalid type(s) definition: '{types}'.")

    return lambda obj: isinstance(obj, types)


def attrchecker(*names: str) -> Callable[[Any], bool]:
    """Return a callable object that checks if operand has all `names` attributes.

    Examples:
        >>> from collections import namedtuple
        >>> Point = namedtuple("Point", ["x", "y"])
        >>> point = Point(1.0, 2.0)
        >>> checker = attrchecker("x")
        >>> checker(point)
        True

        >>> checker = attrchecker("x", "y")
        >>> checker(point)
        True

        >>> checker = attrchecker("z")
        >>> checker(point)
        False

    """
    if not all(isinstance(name, str) for name in names):
        raise TypeError(f"Arguments with invalid attribute names: '{names}'.")
    return lambda obj: all(hasattr(obj, name) for name in names)


def attrgetter_(*names: str, default: Any = NOTHING) -> Callable[[Any], Any]:
    """Return a callable object that gets `names` attributes from its operand.

    Similar to :func:`operator.attrgetter()` but accepts a default value.

    Examples:
        >>> from collections import namedtuple
        >>> Point = namedtuple("Point", ["x", "y"])
        >>> point = Point(1.0, 2.0)
        >>> getter = attrgetter_("x")
        >>> getter(point)
        1.0

        >>> import math
        >>> getter = attrgetter_("z", default=math.nan)
        >>> getter(point)
        nan

        >>> import math
        >>> getter = attrgetter_("x", "y", "z", default=math.nan)
        >>> getter(point)
        (1.0, 2.0, nan)

    """
    if any(not isinstance(name, str) for name in names):
        raise TypeError(f"Attribute names must be strings (provided: {names})")

    if default is NOTHING:
        return operator.attrgetter(*names)
    else:
        if len(names) == 1:
            name = names[0]

            def _getter_with_defaults(obj: Any) -> Any:
                return getattr(obj, name, default)

        else:

            def _getter_with_defaults(obj: Any) -> Any:
                return tuple(getattr(obj, name, default) for name in names)

        return _getter_with_defaults


def getitem_(obj: Any, key: Any, default: Any = NOTHING) -> Any:
    """Return the value of `obj` at index `key`.

    Similar to :func:`operator.getitem()` but accepts a default value.

    Examples:
        >>> d = {"a": 1}
        >>> getitem_(d, "a")
        1

        >>> d = {"a": 1}
        >>> getitem_(d, "b", "default")
        'default'

    """
    if default is NOTHING:
        result = obj[key]
    else:
        try:
            result = obj[key]
        except (KeyError, IndexError):
            result = default

    return result


def itemgetter_(key: Any, default: Any = NOTHING) -> Callable[[Any], Any]:
    """Return a callable object that gets `key` item from its operand.

    Similar to :func:`operator.itemgetter()` but accepts a default value.

    Examples:
        >>> d = {"a": 1}
        >>> getter = itemgetter_("a")
        >>> getter(d)
        1

        >>> d = {"a": 1}
        >>> getter = itemgetter_("b", "default")
        >>> getter(d)
        'default'

    """
    return lambda obj: getitem_(obj, key, default=default)


@dataclasses.dataclass(frozen=True)
class IndexerCallable(Generic[_S, _T]):
    """
    An indexer class applying the wrapped function to the index arguments.

    Examples:
        >>> indexer = IndexerCallable(lambda x: x**2)
        >>> indexer[3]
        9

        >>> indexer = IndexerCallable(lambda a, b: a + b)
        >>> indexer[3, 4]
        7
    """

    func: ArgsOnlyCallable[_S, _T]

    def __getitem__(self, key: _S | tuple[_S, ...]) -> _T:
        return self.func(*key) if isinstance(key, tuple) else self.func(key)


class fluid_partial(functools.partial):
    """Create a `functools.partial` with support for multiple applications calling `.partial()`."""

    def partial(self, *args: Any, **kwargs: Any) -> fluid_partial:
        return fluid_partial(self, *args, **kwargs)


@overload
def with_fluid_partial(
    func: Literal[None] = None, *args: Any, **kwargs: Any
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


@overload
def with_fluid_partial(func: Callable[_P, _T], *args: Any, **kwargs: Any) -> Callable[_P, _T]: ...


def with_fluid_partial(
    func: Optional[Callable[..., Any]] = None, *args: Any, **kwargs: Any
) -> Union[Callable[..., Any], Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """Add a `partial` attribute to the decorated function.

    The `partial` attribute is a function that behaves like `functools.partial`,
    but also supports partial application of the decorated function. It can be
    used both as a bare or a parameterized decorator.

    Arguments:
        func: The function to decorate.

    Returns:
        Returns the decorated function with an extra `.partial()` attribute.

    Example:
        >>> @with_fluid_partial
        ... def add(a, b):
        ...     return a + b
        >>> add.partial(1)(2)
        3
    """

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func.partial = fluid_partial(functools.partial, func, *args, **kwargs)  # type: ignore[attr-defined]  # add attribute
        return func

    return _decorator(func) if func is not None else _decorator
