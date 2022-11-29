# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
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

"""General utility functions. Some functionalities are directly imported from dependencies."""


from __future__ import annotations

import collections.abc
import dataclasses
import enum
import functools
import hashlib
import itertools
import operator
import pickle
import pprint
import re
import sys
import types
import typing
import uuid
import warnings

import deepdiff  # type: ignore[import]
import xxhash  # type: ignore[import]
from boltons.iterutils import flatten as flatten  # type: ignore[import]  # noqa: F401
from boltons.iterutils import flatten_iter as flatten_iter  # type: ignore[import]  # noqa: F401
from boltons.iterutils import is_collection as is_collection  # type: ignore[import]  # noqa: F401
from boltons.strutils import a10n as a10n  # type: ignore[import]  # noqa: F401
from boltons.strutils import asciify as asciify  # type: ignore[import]  # noqa: F401
from boltons.strutils import (  # type: ignore[import]  # noqa: F401
    format_int_list as format_int_list,
)
from boltons.strutils import (  # type: ignore[import]  # noqa: F401
    iter_splitlines as iter_splitlines,
)
from boltons.strutils import parse_int_list as parse_int_list  # type: ignore[import]  # noqa: F401
from boltons.strutils import slugify as slugify  # type: ignore[import]  # noqa: F401
from boltons.strutils import unwrap_text as unwrap_text  # type: ignore[import]  # noqa: F401

from . import extended_typing as xtyping
from .extended_typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    ParamSpec,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from .type_definitions import NOTHING, NothingType


try:
    # For perfomance reasons, try to use cytoolz when possible (using cython)
    import cytoolz as toolz  # type: ignore[import]
except ModuleNotFoundError:
    # Fall back to pure Python toolz
    import toolz  # type: ignore[import]  # noqa: F401


T = TypeVar("T")


def isinstancechecker(type_info: Union[Type, Iterable[Type]]) -> Callable[[Any], bool]:
    """Return a callable object that checks if operand is an instance of `type_info`.

    Examples:
        >>> checker = isinstancechecker((int, str))
        >>> checker(3)
        True
        >>> checker('3')
        True
        >>> checker(3.3)
        False

    """
    types: Tuple[Type, ...] = tuple()
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
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> point = Point(1.0, 2.0)
        >>> checker = attrchecker('x')
        >>> checker(point)
        True

        >>> checker = attrchecker('x', 'y')
        >>> checker(point)
        True

        >>> checker = attrchecker('z')
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
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> point = Point(1.0, 2.0)
        >>> getter = attrgetter_('x')
        >>> getter(point)
        1.0

        >>> import math
        >>> getter = attrgetter_('z', default=math.nan)
        >>> getter(point)
        nan

        >>> import math
        >>> getter = attrgetter_('x', 'y', 'z', default=math.nan)
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
        >>> d = {'a': 1}
        >>> getitem_(d, 'a')
        1

        >>> d = {'a': 1}
        >>> getitem_(d, 'b', 'default')
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
        >>> d = {'a': 1}
        >>> getter = itemgetter_('a')
        >>> getter(d)
        1

        >>> d = {'a': 1}
        >>> getter = itemgetter_('b', 'default')
        >>> getter(d)
        'default'

    """
    return lambda obj: getitem_(obj, key, default=default)


_P = ParamSpec("_P")


@overload
def optional_lru_cache(
    func: Literal[None] = None, *, maxsize: Optional[int] = 128, typed: bool = False
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    ...


@overload
def optional_lru_cache(  # noqa: F811  # redefinition of unused function
    func: Callable[_P, _T], *, maxsize: Optional[int] = 128, typed: bool = False
) -> Callable[_P, _T]:
    ...


def optional_lru_cache(  # noqa: F811  # redefinition of unused function
    func: Optional[Callable[_P, _T]] = None, *, maxsize: Optional[int] = 128, typed: bool = False
) -> Union[Callable[_P, _T], Callable[[Callable[_P, _T]], Callable[_P, _T]]]:
    """Wrap :func:`functools.lru_cache` to fall back to the original function if arguments are not hashable.

    Examples:
        >>> @optional_lru_cache(typed=True)
        ... def func(a, b):
        ...     print(f"Inside func({a}, {b})")
        ...     return a + b
        ...
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


def register_subclasses(*subclasses: Type) -> Callable[[Type], Type]:
    """Class decorator to automatically register virtual subclasses.

    Examples:
        >>> import abc
        >>> class MyVirtualSubclassA:
        ...     pass
        ...
        >>> class MyVirtualSubclassB:
        ...    pass
        ...
        >>> @register_subclasses(MyVirtualSubclassA, MyVirtualSubclassB)
        ... class MyBaseClass(abc.ABC):
        ...    pass
        ...
        >>> issubclass(MyVirtualSubclassA, MyBaseClass) and issubclass(MyVirtualSubclassB, MyBaseClass)
        True

    """

    def _decorator(base_cls: Type) -> Type:
        for s in subclasses:
            base_cls.register(s)
        return base_cls

    return _decorator


_T = TypeVar("_T")


def noninstantiable(cls: Type[_T]) -> Type[_T]:
    """Make a class without abstract method non-instantiable (subclasses should be instantiable)."""
    if not isinstance(cls, type):
        raise ValueError(f"Non-type value ({cls}) passed to 'noninstantiable()' class decorator.")

    original_init = cls.__init__

    def _noninstantiable_init(self: _T, *args: Any, **kwargs: Any) -> None:
        if self.__class__ is cls:
            raise TypeError(f"Trying to instantiate '{cls.__name__}' non-instantiable class.")
        else:
            original_init(self, *args, **kwargs)

    cls.__init__ = _noninstantiable_init  # type: ignore[assignment]
    cls.__noninstantiable__ = True  # type: ignore[attr-defined]

    return cls


def is_noninstantiable(cls: Type[_T]) -> bool:
    """Return True if `model` is a non-instantiable class."""
    return "__noninstantiable__" in cls.__dict__


def content_hash(*args: Any, hash_algorithm: str | xtyping.HashlibAlgorithm | None = None) -> str:
    """Stable content-based hash function using instance serialization data.

    It provides a customizable hash function for any kind of data.
    Unlike the builtin `hash` function, it is stable (same hash value across
    interpreter reboots) and it does not use hash customizations on user
    classes (it uses `pickle` internally to get a byte stream).

    Arguments:
        hash_algorithm: object implementing the `hash algorithm` interface
            from :mod:`hashlib` or canonical name (`str`) of the
            hash algorithm as defined in :mod:`hashlib`.
            Defaults to :class:`xxhash.xxh64`.

    """
    if hash_algorithm is None:
        hash_algorithm = xxhash.xxh64()
    elif isinstance(hash_algorithm, str):
        hash_algorithm = hashlib.new(hash_algorithm)  # type: ignore[assignment]

    hash_algorithm.update(pickle.dumps(args))  # type: ignore[union-attr]
    result = hash_algorithm.hexdigest()  # type: ignore[union-attr]
    assert isinstance(result, str)

    return result


ddiff = deepdiff.DeepDiff
"""Shortcut for deepdiff.DeepDiff.

Check https://zepworks.com/deepdiff/current/diff.html for more info.
"""


def dhash(obj: Any, **kwargs: Any) -> str:
    """Shortcut for deepdiff.deephash.DeepHash.

    Check https://zepworks.com/deepdiff/current/deephash.html for more info.
    """
    return deepdiff.deephash.DeepHash(obj)[obj]


def pprint_ddiff(
    old: Any,
    new: Any,
    *,
    pprint_opts: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """Pretty printing of deepdiff.DeepDiff objects.

    Keyword Arguments:
        pprint_opts: kwargs dict with options for pprint.pprint.
    """
    pprint_opts = pprint_opts or {"indent": 2}
    pprint.pprint(deepdiff.DeepDiff(old, new, **kwargs), **pprint_opts)


AnyWordsIterable = Union[str, Iterable[str]]


class CaseStyleConverter:
    """Utility class to convert name strings to different case styles.

    Functionality exposed through :meth:`split()`, :meth:`join()` and
    :meth:`convert()` methods.

    """

    class CASE_STYLE(enum.Enum):
        CONCATENATED = "concatenated"
        CANONICAL = "canonical"
        CAMEL = "camel"
        PASCAL = "pascal"
        SNAKE = "snake"
        KEBAB = "kebab"

    @classmethod
    def split(cls, name: str, case_style: Union[CASE_STYLE, str]) -> List[str]:
        if isinstance(case_style, str):
            case_style = cls.CASE_STYLE(case_style)
        assert isinstance(case_style, cls.CASE_STYLE)
        if case_style == cls.CASE_STYLE.CONCATENATED:
            raise ValueError("Impossible to split a simply concatenated string")

        splitter: Callable[[str], List[str]] = getattr(cls, f"split_{case_style.value}_case")
        return splitter(name)

    @classmethod
    def join(cls, words: AnyWordsIterable, case_style: Union[CASE_STYLE, str]) -> str:
        if isinstance(case_style, str):
            case_style = cls.CASE_STYLE(case_style)
        assert isinstance(case_style, cls.CASE_STYLE)
        if isinstance(words, str):
            words = [words]
        if not isinstance(words, collections.abc.Iterable):
            raise TypeError(f"'{words}' type is not a valid sequence of words")

        joiner: Callable[[AnyWordsIterable], str] = getattr(cls, f"join_{case_style.value}_case")
        return joiner(words)

    @classmethod
    def convert(
        cls, name: str, source_style: Union[CASE_STYLE, str], target_style: Union[CASE_STYLE, str]
    ) -> str:
        return cls.join(cls.split(name, source_style), target_style)

    # Following `join_...`` functions are based on:
    #    https://blog.kangz.net/posts/2016/08/31/code-generation-the-easier-way/
    #
    @staticmethod
    def join_concatenated_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else words
        return "".join(words).lower()

    @staticmethod
    def join_canonical_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else words
        return (" ".join(words)).lower()

    @staticmethod
    def join_camel_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else list(words)
        return words[0].lower() + "".join(word.title() for word in words[1:])

    @staticmethod
    def join_pascal_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else words
        return "".join(word.title() for word in words)

    @staticmethod
    def join_snake_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else words
        return "_".join(words).lower()

    @staticmethod
    def join_kebab_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else words
        return "-".join(words).lower()

    # Following `split_...`` functions are based on:
    #    https://stackoverflow.com/a/29920015/7232525
    #
    @staticmethod
    def split_canonical_case(name: str) -> List[str]:
        return name.split()

    @staticmethod
    def split_camel_case(name: str) -> List[str]:
        matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", name)
        return [m.group(0) for m in matches]

    split_pascal_case = split_camel_case

    @staticmethod
    def split_snake_case(name: str) -> List[str]:
        return name.split("_")

    @staticmethod
    def split_kebab_case(name: str) -> List[str]:
        return name.split("-")


class Namespace(types.SimpleNamespace, Generic[T]):
    """A `types.SimpleNamespace`-like class with additional dict-like interface.

    Examples:
        >>> ns = Namespace(a=10, b="hello")
        >>> ns.a
        10
        >>> ns.b = 20
        >>> ns.b
        20

        >>> ns = Namespace(a=10, b="hello")
        >>> list(ns.keys())
        ['a', 'b']

        >>> list(ns.values())
        [10, 'hello']

        >>> list(ns.items())
        [('a', 10), ('b', 'hello')]

    """

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def items(self) -> Iterable[Tuple[str, T]]:
        return self.__dict__.items()

    def keys(self) -> Iterable[str]:
        return self.__dict__.keys()

    def values(self) -> Iterable[T]:
        return self.__dict__.values()

    def reset(self, data: Optional[Dict[str, Any]] = None) -> None:
        self.__dict__.clear()
        if data:
            self.__dict__.update(data)

    def as_dict(self) -> Dict[str, T]:
        return {**self.__dict__}

    asdict = as_dict


class FrozenNamespace(Namespace[T]):
    """An immutable version of :class:`Namespace`.

    Examples:
        >>> ns = FrozenNamespace(a=10, b="hello")
        >>> ns.a
        10
        >>> ns.a = 20
        Traceback (most recent call last):
           ...
        TypeError: Trying to modify immutable 'FrozenNamespace' instance.

        >>> ns = FrozenNamespace(a=10, b="hello")
        >>> list(ns.items())
        [('a', 10), ('b', 'hello')]

        >>> ns = FrozenNamespace(a=10, b="hello")
        >>> hashed = hash(ns)
        >>> assert isinstance(hashed, int)
        >>> hashed == hash(ns) == ns.__cached_hash_value__
        True
    """

    __slots__ = "__cached_hash_value__"  # This slot is used to avoid polluting the namespace

    def __setattr__(self, __name: str, __value: Any) -> None:
        raise TypeError(f"Trying to modify immutable '{self.__class__.__name__}' instance.")

    def __delattr__(self, __name: str) -> None:
        raise TypeError(f"Trying to modify immutable '{self.__class__.__name__}' instance.")

    def __hash__(self) -> int:  # type: ignore[override]
        if not hasattr(self, "__cached_hash_value__"):
            object.__setattr__(self, "__cached_hash_value__", hash(tuple(self.__dict__.items())))

        return self.__cached_hash_value__


@dataclasses.dataclass
class UIDGenerator:
    """Simple unique id generator using different methods."""

    prefix: Optional[str] = (
        dataclasses.field(default=None, kw_only=True)  # type: ignore[call-overload]
        if sys.version_info >= (3, 10)
        else dataclasses.field(default=None)
    )
    width: Optional[int] = (
        dataclasses.field(default=None, kw_only=True)  # type: ignore[call-overload]
        if sys.version_info >= (3, 10)
        else dataclasses.field(default=None)
    )
    warn_unsafe: Optional[bool] = (
        dataclasses.field(default=None, kw_only=True)  # type: ignore[call-overload]
        if sys.version_info >= (3, 10)
        else dataclasses.field(default=None)
    )

    _counter: Iterator[int] = dataclasses.field(
        default_factory=functools.partial(itertools.count, 1), init=False
    )
    """Constantly increasing counter for generation of sequential unique ids."""

    def random_id(self, *, prefix: Optional[str] = None, width: Optional[int] = None) -> str:
        """Generate a random globally unique id."""
        width = width or self.width or 8
        if width <= 4:
            raise ValueError(f"Width must be a positive number > 4 ({width} provided).")
        prefix = prefix or self.prefix
        u = uuid.uuid4()
        s = str(u).replace("-", "")[:width]
        return f"{prefix}_{s}" if prefix else f"{s}"

    def sequential_id(self, *, prefix: Optional[str] = None, width: Optional[int] = None) -> str:
        """Generate a sequential unique id (for the current session)."""
        width = width or self.width
        if width is not None and width < 1:
            raise ValueError(f"Width must be a positive number ({width} provided).")
        prefix = prefix or self.prefix
        count = next(self._counter)
        s = f"{count:0{width}}" if width else f"{count}"
        return f"{prefix}_{s}" if prefix else f"{s}"

    def reset_sequence(self, start: int = 1, *, warn_unsafe: Optional[bool] = None) -> UIDGenerator:
        """Reset generator counter.

        It returns the same instance to allow resetting at initialization:

        Example:
            >>> generator = UIDGenerator().reset_sequence(3)

        Notes:
            If the new start value is lower than the last generated UID, new
            IDs are not longer guaranteed to be unique.

        """
        if start < 0:
            raise ValueError(f"Starting value must be a positive number ({start} provided).")
        if warn_unsafe is None:
            warn_unsafe = self.warn_unsafe
        if warn_unsafe and start < next(self._counter):
            warnings.warn("Unsafe reset of UIDGenerator ({self})")
        self._counter = itertools.count(start)

        return self


UIDs = UIDGenerator()

# -- Iterators --
S = TypeVar("S")
K = TypeVar("K")

P = ParamSpec("P")


def as_xiter(iterator_func: Callable[P, Iterable[T]]) -> Callable[P, XIterable[T]]:
    """Wrap the provided callable to convert its output in a :class:`XIterable`."""

    @functools.wraps(iterator_func)
    def _xiterator(*args: Any, **keywords: Any) -> XIterable[T]:
        return xiter(iterator_func(*args, **keywords))

    return _xiterator


xenumerate = as_xiter(enumerate)


class XIterable(Iterable[T]):
    """Iterable wrapper supporting method chaining for extra functionality."""

    iterator: Iterator[T]

    def __init__(self, it: Union[Iterable[T], Iterator[T]]) -> None:
        object.__setattr__(self, "iterator", iter(it))

    def __getattr__(self, name: str) -> Any:
        # Forward special methods to wrapped iterator
        if name.startswith("__") and name.endswith("__"):
            return getattr(self.iterator, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError(f"{type(self).__name__} is immutable.")

    def __iter__(self) -> Iterator[T]:
        return self.iterator

    def map(self, func: Callable) -> XIterable[Any]:  # noqa  # A003: shadowing a python builtin
        """Apply a callable to every iterator element.

        Equivalent to ``map(func, self)``.

        For detailed information check :func:`map` reference.

        Examples:
            >>> it = xiter(range(3))
            >>> list(it.map(str))
            ['0', '1', '2']

            >>> it = xiter(range(3))
            >>> list(it.map(lambda x: -x).map(str))
            ['0', '-1', '-2']

            If the callable requires additional arguments, ``lambda`` of :func:`functools.partial`
            functions can be used:

            >>> def times(a, b):
            ...     return a * b
            >>> times_2 = functools.partial(times, 2)
            >>> it = xiter(range(4))
            >>> list(it.map(lambda x: x + 1).map(times_2))
            [2, 4, 6, 8]

            Curried functions generated by :func:`toolz.curry` will also work as expected:

            >>> @toolz.curry
            ... def mul(x, y):
            ...     return x * y
            >>> it = xiter(range(4))
            >>> list(it.map(lambda x: x + 1).map(mul(5)))
            [5, 10, 15, 20]

        """
        if not callable(func):
            raise ValueError(f"Invalid function or callable: '{func}'.")
        return XIterable(map(func, self.iterator))

    def filter(  # noqa  # A003: shadowing a python builtin
        self, func: Callable[..., bool]
    ) -> XIterable[T]:
        """Filter elements with callables.

        Equivalent to ``filter(func, self)``.

        For detailed information check :func:`filter` reference.

        Examples:
            >>> it = xiter(range(4))
            >>> list(it.filter(lambda x: x % 2 == 0))
            [0, 2]

            >>> it = xiter(range(4))
            >>> list(it.filter(lambda x: x % 2 == 0).filter(lambda x: x > 1))
            [2]


        Notes:
            `lambdas`, `partial` and `curried` functions are supported (see :meth:`map`).

        """
        if not callable(func):
            raise TypeError(f"Invalid function or callable: '{func}'.")
        return XIterable(filter(func, self.iterator))

    def if_isinstance(self, *types: Type) -> XIterable[T]:
        """Filter elements using :func:`isinstance` checks.

        Equivalent to ``xiter(item for item in self if isinstance(item, types))``.

        Examples:
            >>> it = xiter([1, '2', 3.3, [4, 5], {6, 7}])
            >>> list(it.if_isinstance(int, float))
            [1, 3.3]

        """
        return XIterable(filter(isinstancechecker([*types]), self.iterator))

    def if_not_isinstance(self, *types: Type) -> XIterable[T]:
        """Filter elements using negated :func:`isinstance` checks.

        Equivalent to ``xiter(item for item in self if not isinstance(item, types))``.

        Examples:
            >>> it = xiter([1, '2', 3.3, [4, 5], {6, 7}])
            >>> list(it.if_not_isinstance(int, float))
            ['2', [4, 5], {6, 7}]

        """
        return XIterable(
            filter(toolz.functoolz.complement(isinstancechecker([*types])), self.iterator)
        )

    def if_is(self, obj: Any) -> XIterable[T]:
        """Filter elements using :func:`operator.is_` checks.

        Equivalent to ``xiter(item for item in self if item is obj)``.

        Examples:
            >>> it = xiter([1, None, 1, 123456789, None, 123456789])
            >>> list(it.if_is(None))
            [None, None]

            >>> it = xiter([1, None, 1, 123456789, None, 123456789])
            >>> list(it.if_is(1))
            [1, 1]

            >>> it = xiter([1, None, 1, 123456789, None, 123456789])
            >>> list(it.if_is(123456789))
            []

        """
        return XIterable(filter(lambda x: operator.is_(x, obj), self.iterator))

    def if_is_not(self, obj: Any) -> XIterable[T]:
        """Filter elements using negated  :func:`operator.is_` checks.

        Equivalent to ``xiter(item for item in self if item is not obj)``.

        Examples:
            >>> it = xiter([1, None, 1, 123456789, None, 123456789])
            >>> list(it.if_is_not(None))
            [1, 1, 123456789, 123456789]

            >>> it = xiter([1, None, 1, 123456789, None, 123456789])
            >>> list(it.if_is_not(1))
            [None, 123456789, None, 123456789]

            >>> it = xiter([1, None, 1, 123456789, None, 123456789])
            >>> list(it.if_is_not(123456789))
            [1, None, 1, 123456789, None, 123456789]

        """
        return XIterable(filter(lambda x: not operator.is_(x, obj), self.iterator))

    def if_in(self, collection: Collection[T]) -> XIterable[T]:
        """Filter elements using :func:`operator.contains` checks.

        Equivalent to ``xiter(item for item in self if item in collection)``.

        Examples:
            >>> it = xiter(range(8))
            >>> list(it.if_in([0, 2, 4, 6]))
            [0, 2, 4, 6]

        """
        return XIterable(filter(lambda x: operator.contains(collection, x), self.iterator))

    def if_not_in(self, collection: Collection[T]) -> XIterable[T]:
        """Filter elements using negated :func:`operator.contains` checks.

        Equivalent to ``xiter(item for item in self if item not in collection)``.

        Examples:
            >>> it = xiter(range(8))
            >>> list(it.if_not_in([0, 2, 4, 6]))
            [1, 3, 5, 7]

        """
        return XIterable(filter(lambda x: not operator.contains(collection, x), self.iterator))

    def if_contains(self, *values: Any) -> XIterable[T]:
        """Filter elements using :func:`operator.contains` checks.

        Equivalent to ``xiter(item for item in self if all(v in item for v in values))``.

        Examples:
            >>> it = xiter([None, (0, 1, 2), "a", (1, 2, 3), (2, 3, 4), "b"])
            >>> list(it.if_contains(2))
            [(0, 1, 2), (1, 2, 3), (2, 3, 4)]

            >>> it = xiter([None, (0, 1, 2), "a", (1, 2, 3), (2, 3, 4), "b"])
            >>> list(it.if_contains(1, 2))
            [(0, 1, 2), (1, 2, 3)]

        """

        def _contains(a: Any, collection: Tuple) -> bool:
            try:
                return all(operator.contains(a, v) for v in collection)
            except Exception:
                return False

        return XIterable(filter((lambda x: _contains(x, values)), self.iterator))

    def if_hasattr(self, *names: str) -> XIterable[T]:
        """Filter elements using :func:`hasattr` checks.

        Equivalent to ``filter(attrchecker(names), self)``.

        Examples:
            >>> it = xiter([1, '2', 3.3, [4, 5], {6, 7}])
            >>> list(it.if_hasattr('__len__'))
            ['2', [4, 5], {6, 7}]

            >>> it = xiter([1, '2', 3.3, [4, 5], {6, 7}])
            >>> list(it.if_hasattr('__len__', 'index'))
            ['2', [4, 5]]

        """
        return XIterable(filter(attrchecker(*names), self.iterator))

    def getattr(  # noqa  # A003: shadowing a python builtin
        self, *names: str, default: Any = NOTHING
    ) -> XIterable[Any]:
        """Get provided attributes from each item in a sequence.

        Equivalent to ``map(attrgetter_(*names, default=default), self)``.

        Keyword Arguments:
            default: returned value if the item does not contain an attribute
                with the provided `name`.

        For detailed information check :func:`attrgetter_` and :func:`operator.attrgetter` reference.

        Examples:
            >>> from collections import namedtuple
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> it = xiter([Point(1.0, -1.0), Point(2.0, -2.0), Point(3.0, -3.0)])
            >>> list(it.getattr('y'))
            [-1.0, -2.0, -3.0]

            >>> it = xiter([Point(1.0, -1.0), Point(2.0, -2.0), Point(3.0, -3.0)])
            >>> list(it.getattr('x', 'z', default=None))
            [(1.0, None), (2.0, None), (3.0, None)]

        """
        return XIterable(map(attrgetter_(*names, default=default), self.iterator))

    def getitem(self, *indices: Union[int, str], default: Any = NOTHING) -> XIterable[Any]:
        """Get provided indices data from each item in a sequence.

        Equivalent to ``toolz.itertoolz.pluck(indices, self)``.

        Keyword Arguments:
            default: returned value if the item does not contain an `index`
                with the provided value.

        For detailed information check :func:`toolz.itertoolz.pluck` reference.

            >>> it = xiter([('a', 1), ('b', 2), ('c', 3)])
            >>> list(it.getitem(0))
            ['a', 'b', 'c']

            >>> it = xiter([
            ...     dict(name="AA", age=20, country="US"),
            ...     dict(name="BB", age=30, country="UK"),
            ...     dict(name="CC", age=40, country="EU"),
            ...     dict(country="CH")
            ... ])
            >>> list(it.getitem("name", "age", default=None))
            [('AA', 20), ('BB', 30), ('CC', 40), (None, None)]

        """
        ind: Any  # a hashable item or a list of hashable items
        if len(indices) == 1:
            ind = indices[0]
        else:
            ind = [*indices]
        if default is NOTHING:
            return XIterable(toolz.itertoolz.pluck(ind, self.iterator))
        else:
            return XIterable(toolz.itertoolz.pluck(ind, self.iterator, default))

    def chain(self, *others: Iterable) -> XIterable[Union[T, S]]:
        """Chain iterators.

        Equivalent to ``itertools.chain(self, *others)``.

        For detailed information check :func:`itertools.chain` reference.

        Examples:
            >>> it_a, it_b = xiter(range(2)), xiter(['a', 'b'])
            >>> list(it_a.chain(it_b))
            [0, 1, 'a', 'b']

            >>> it_a = xiter(range(2))
            >>> list(it_a.chain(['a', 'b'], ['A', 'B']))
            [0, 1, 'a', 'b', 'A', 'B']

        """
        iterators = [it.iterator if isinstance(it, XIterable) else it for it in others]
        return XIterable(itertools.chain(self.iterator, *iterators))

    def diff(
        self,
        *others: Iterable,
        default: Any = NOTHING,
        key: Union[NOTHING, Callable] = NOTHING,
    ) -> XIterable[Tuple[T, S]]:
        """Diff iterators.

        Equivalent to ``toolz.itertoolz.diff(self, *others)``.

        Keyword Arguments:
            default: returned value for missing items.
            key: callable computing the key to use per item in the sequence.

        For detailed information check :func:`toolz.itertoolz.diff` reference.

        Examples:
            >>> it_a, it_b = xiter([1, 2, 3]), xiter([1, 3, 5])
            >>> list(it_a.diff(it_b))
            [(2, 3), (3, 5)]

            >>> it_a, it_b, it_c = xiter([1, 2, 3]), xiter([1, 3, 5]), xiter([1, 2, 5])
            >>> list(it_a.diff(it_b, it_c))
            [(2, 3, 2), (3, 5, 5)]

            Adding missing values:

            >>> it_a = xiter([1, 2, 3, 4])
            >>> list(it_a.diff([1, 3, 5], default=None))
            [(2, 3), (3, 5), (4, None)]

            Use a key function:

            >>> it_a, it_b = xiter(["Apples", "Bananas"]), xiter(["apples", "oranges"])
            >>> list(it_a.diff(it_b, key=str.lower))
            [('Bananas', 'oranges')]

        """
        kwargs: Dict[str, Any] = {}
        if default is not NOTHING:
            kwargs["default"] = default
        if key is not NOTHING:
            kwargs["key"] = key

        iterators = [it.iterator if isinstance(it, XIterable) else it for it in others]
        return XIterable(toolz.itertoolz.diff(self.iterator, *iterators, **kwargs))

    def product(
        self, other: Union[Iterable[S], int]
    ) -> Union[XIterable[Tuple[T, S]], XIterable[Tuple[T, T]]]:
        """Product of iterators.

        Equivalent to ``itertools.product(it_a, it_b)``.

        For detailed information check :func:`itertools.product` reference.

        Examples:
            >>> it_a, it_b = xiter([0, 1]), xiter(['a', 'b'])
            >>> list(it_a.product(it_b))
            [(0, 'a'), (0, 'b'), (1, 'a'), (1, 'b')]

            Product of an iterator with itself:

            >>> it_a = xiter([0, 1])
            >>> list(it_a.product(3))
            [(0, 0, 0), (1, 1, 1)]

        """
        if isinstance(other, int):
            if other < 0:
                raise ValueError(
                    f"Only non-negative integer numbers are accepted (provided: {other})."
                )
            return XIterable(map(lambda item: tuple([item] * other), self.iterator))  # type: ignore  # mypy gets confused with `other`
        else:
            if not isinstance(other, XIterable):
                other = xiter(other)
        return XIterable(itertools.product(self.iterator, other.iterator))

    def partition(
        self, n: int, *, exact: bool = False, fill: Any = NOTHING
    ) -> XIterable[Tuple[T, ...]]:
        """Partition iterator into tuples of length `n` (``exact=True``) or at most `n` (``exact=False``).

        Equivalent to ``toolz.itertoolz.partition(n, self)`` or
        ``toolz.itertoolz.partition_all(n, self)``.

        For detailed information check :func:`toolz.itertoolz.partition` and
        :func:`toolz.itertoolz.partition_all` reference.

        Keyword Arguments:
            exact: if `True`, it will return tuples of length `n`. If `False`, the last
                tuple could have a smaller size if there are not enough items in the sequence.
            fill: if provided together with ``exact=True``, this value will be used to fill
                the last tuple until it has exactly length `n`.

        Examples:
            >>> it = xiter(range(7))
            >>> list(it.partition(3))
            [(0, 1, 2), (3, 4, 5), (6,)]

            >>> it = xiter(range(7))
            >>> list(it.partition(3, exact=True))
            [(0, 1, 2), (3, 4, 5)]

            >>> it = xiter(range(7))
            >>> list(it.partition(3, exact=True, fill=None))
            [(0, 1, 2), (3, 4, 5), (6, None, None)]

        """
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"Only positive integer numbers are accepted (provided: {n}).")

        if exact:
            if fill is NOTHING:
                iterator = toolz.itertoolz.partition(n, self.iterator)
            else:
                iterator = toolz.itertoolz.partition(n, self.iterator, pad=fill)
        else:
            iterator = toolz.itertoolz.partition_all(n, self.iterator)

        return XIterable(iterator)

    def take_nth(self, n: int) -> XIterable[T]:
        """Take every nth item in sequence.

        Equivalent to ``toolz.itertoolz.take_nth(n, self)``.

        For detailed information check :func:`toolz.itertoolz.take_nth` reference.

        Examples:
            >>> it = xiter(range(7))
            >>> list(it.take_nth(3))
            [0, 3, 6]

        """
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"Only positive integer numbers are accepted (provided: {n}).")
        return XIterable(toolz.itertoolz.take_nth(n, self.iterator))

    def zip(  # noqa  # A003: shadowing a python builtin
        self, *others: Iterable, fill: Any = NOTHING
    ) -> XIterable[Tuple[T, S]]:
        """Zip iterators.

        Equivalent to ``zip(self, *others)`` or ``itertools.zip_longest(self, *others, fillvalue=fill)``.

        Keyword Arguments:
            fill: value used to fill the result for sequences with fewer items.

        For detailed information check :func:`zip` and :func:`itertools.zip_longest` reference.

        Examples:
            >>> it_a = xiter(range(3))
            >>> it_b = ['a', 'b', 'c']
            >>> list(it_a.zip(it_b))
            [(0, 'a'), (1, 'b'), (2, 'c')]

            >>> it = xiter(range(3))
            >>> list(it.zip(['a', 'b', 'c'], ['A', 'B', 'C']))
            [(0, 'a', 'A'), (1, 'b', 'B'), (2, 'c', 'C')]

            >>> it = xiter(range(5))
            >>> list(it.zip(['a', 'b', 'c'], ['A', 'B', 'C'], fill=None))
            [(0, 'a', 'A'), (1, 'b', 'B'), (2, 'c', 'C'), (3, None, None), (4, None, None)]

        """
        iterators = [it.iterator if isinstance(it, XIterable) else it for it in others]
        if fill is NOTHING:
            return XIterable(zip(self.iterator, *iterators))
        else:
            return XIterable(itertools.zip_longest(self.iterator, *iterators, fillvalue=fill))

    def unzip(self) -> XIterable[Tuple[Any, ...]]:
        """Unzip iterator.

        Equivalent to ``zip(*self)``.

        For detailed information check :func:`zip` reference.

        Examples:
            >>> it = xiter([('a', 1), ('b', 2), ('c', 3)])
            >>> list(it.unzip())
            [('a', 'b', 'c'), (1, 2, 3)]

        """
        return XIterable(zip(*self.iterator))  # type: ignore  # mypy gets confused with *args

    @typing.overload
    def islice(self, __stop: int) -> XIterable[T]:
        ...

    @typing.overload
    def islice(self, __start: int, __stop: int, __step: int = 1) -> XIterable[T]:
        ...

    def islice(
        self,
        __start_or_stop: int,
        __stop_or_nothing: Union[int, NothingType] = NOTHING,
        step: int = 1,
    ) -> XIterable[T]:
        """Select elements from an iterable.

        Equivalent to ``itertools.islice(iterator, start, stop, step)``.

        For detailed information check :func:`itertools.islice` reference.

        Examples:
            >>> it = xiter(range(10))
            >>> list(it.islice(2))
            [0, 1]

            >>> it = xiter(range(10))
            >>> list(it.islice(2, 8))
            [2, 3, 4, 5, 6, 7]

            >>> it = xiter(range(10))
            >>> list(it.islice(2, 8, 2))
            [2, 4, 6]

        """
        if __stop_or_nothing is NOTHING:
            start = 0
            stop = __start_or_stop
        else:
            assert isinstance(__stop_or_nothing, int)
            start = __start_or_stop
            stop = __stop_or_nothing
        return XIterable(itertools.islice(self.iterator, start, stop, step))

    def select(self, selectors: Iterable[bool]) -> XIterable[T]:
        """Return only the elements which have a corresponding element in selectors that evaluates to True.

        Equivalent to ``itertools.compress(self, selectors)``.

        For detailed information check :func:`itertools.compress` reference.

        Examples:
            >>> it = xiter([1, 2, 3, 1, 3])
            >>> list(it.select([True, False, 1, False, 0]))
            [1, 3]

        """
        if not isinstance(selectors, collections.abc.Iterable):
            raise TypeError(f"Non-iterable 'selectors' value: '{selectors}'.")
        return XIterable(itertools.compress(self.iterator, selectors))

    def unique(self, *, key: Union[NOTHING, Callable] = NOTHING) -> XIterable[T]:
        """Return only unique elements of a sequence.

        Equivalent to ``toolz.itertoolz.unique(self)``.

        Keyword Arguments:
            key: callable computing the key to use per item in the sequence.

        For detailed information check :func:`toolz.itertoolz.unique` reference.

        Examples:
            >>> it = xiter([1, 2, 3, 1, 3])
            >>> list(it.unique())
            [1, 2, 3]

            >>> it = xiter(['cat', 'mouse', 'dog', 'hen'])
            >>> list(it.unique(key=len))
            ['cat', 'mouse']

        """
        if key is NOTHING:
            return XIterable(toolz.itertoolz.unique(self.iterator))
        else:
            return XIterable(toolz.itertoolz.unique(self.iterator, key=key))

    @typing.overload
    def groupby(
        self, key: str, *other_keys: str, as_dict: bool = False
    ) -> XIterable[Tuple[Any, List[T]]]:
        ...

    @typing.overload
    def groupby(self, key: List[Any], *, as_dict: bool = False) -> XIterable[Tuple[Any, List[T]]]:
        ...

    @typing.overload
    def groupby(
        self, key: Callable[[T], Any], *, as_dict: bool = False
    ) -> XIterable[Tuple[Any, List[T]]]:
        ...

    def groupby(
        self,
        key: Union[str, List[Any], Callable[[T], Any]],
        *attr_keys: str,
        as_dict: bool = False,
    ) -> Union[XIterable[Tuple[Any, List[T]]], Dict]:
        """Group a sequence by a given key.

        More or less equivalent to ``toolz.itertoolz.groupby(key, self)`` with some caveats.

        The `key` argument is used in the following way:

            - if `key` is a callable, it will be passed directly to :func:`toolz.itertoolz.groupby`
              to compute an actual `key` value for each item in the sequence.
            - if `key` is a ``str`` or (multiple ``str`` args) they will be used as
              attributes names (:func:`operator.attrgetter`).
            - if `key` is a ``list`` of values, they will be used as index values
              for :func:`operator.itemgetter`.

        Keyword Arguments:
            as_dict: if `True`, it will return the groups ``dict`` instead of a :class:`XIterable`
                instance over `groups.items()`.

        For detailed information check :func:`toolz.itertoolz.groupby` reference.

        Examples:
            >>> it = xiter([(1.0, -1.0), (1.0,-2.0), (2.2, -3.0)])
            >>> list(it.groupby([0]))
            [(1.0, [(1.0, -1.0), (1.0, -2.0)]), (2.2, [(2.2, -3.0)])]

            >>> data = [
            ...     {'x': 1.0, 'y': -1.0, 'z': 1.0},
            ...     {'x': 1.0, 'y': -2.0, 'z': 1.0},
            ...     {'x': 2.2, 'y': -3.0, 'z': 2.2}
            ... ]
            >>> list(xiter(data).groupby(['x']))
            [(1.0, [{'x': 1.0, 'y': -1.0, 'z': 1.0}, {'x': 1.0, 'y': -2.0, 'z': 1.0}]), (2.2, [{'x': 2.2, 'y': -3.0, 'z': 2.2}])]
            >>> list(xiter(data).groupby(['x', 'z']))
            [((1.0, 1.0), [{'x': 1.0, 'y': -1.0, 'z': 1.0}, {'x': 1.0, 'y': -2.0, 'z': 1.0}]), ((2.2, 2.2), [{'x': 2.2, 'y': -3.0, 'z': 2.2}])]

            >>> from collections import namedtuple
            >>> Point = namedtuple('Point', ['x', 'y', 'z'])
            >>> data = [Point(1.0, -2.0, 1.0), Point(1.0, -2.0, 1.0), Point(2.2, 3.0, 2.0)]
            >>> list(xiter(data).groupby('x'))
            [(1.0, [Point(x=1.0, y=-2.0, z=1.0), Point(x=1.0, y=-2.0, z=1.0)]), (2.2, [Point(x=2.2, y=3.0, z=2.0)])]
            >>> list(xiter(data).groupby('x', 'z'))
            [((1.0, 1.0), [Point(x=1.0, y=-2.0, z=1.0), Point(x=1.0, y=-2.0, z=1.0)]), ((2.2, 2.0), [Point(x=2.2, y=3.0, z=2.0)])]

            >>> it = xiter(['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank'])
            >>> list(it.groupby(len))
            [(5, ['Alice', 'Edith', 'Frank']), (3, ['Bob', 'Dan']), (7, ['Charlie'])]

        """
        if (not callable(key) and not isinstance(key, (int, str, list))) or not all(
            isinstance(i, str) for i in attr_keys
        ):
            raise TypeError(f"Invalid 'key' function or attribute name: '{key}'.")
        if callable(key):
            groupby_key = key
        elif isinstance(key, list):
            groupby_key = operator.itemgetter(*key)
        else:
            assert isinstance(key, str)
            groupby_key = operator.attrgetter(key, *attr_keys)

        groups = toolz.itertoolz.groupby(groupby_key, self.iterator)
        return groups if as_dict else xiter(groups.items())

    def accumulate(
        self, func: Callable[[Any, T], Any] = operator.add, *, init: Any = None
    ) -> XIterable:
        """Reduce an iterator using a callable.

        Equivalent to ``itertools.accumulate(self, func, init)``.

        Keyword Arguments:
            init: initial value for the accumulation.

        For detailed information check :func:`itertools.accumulate` reference.

        Examples:
            >>> it = xiter(range(5))
            >>> list(it.accumulate())
            [0, 1, 3, 6, 10]

            >>> it = xiter(range(5))
            >>> list(it.accumulate(init=10))
            [10, 10, 11, 13, 16, 20]

            >>> it = xiter(range(1, 5))
            >>> list(it.accumulate((lambda x, y: x * y), init=-1))
            [-1, -1, -2, -6, -24]

        """
        return XIterable(itertools.accumulate(self.iterator, func, initial=init))

    def reduce(self, bin_op_func: Callable[[Any, T], Any], *, init: Any = None) -> Any:
        """Reduce an iterator using a callable.

        Equivalent to ``functools.reduce(bin_op_func, self, init)``.

        Keyword Arguments:
            init: initial value for the reduction.

        For detailed information check :func:`functools.reduce` reference.

        Examples:
            >>> it = xiter(range(5))
            >>> it.reduce((lambda accu, i: accu + i), init=0)
            10

            >>> it = xiter(['a', 'b', 'c', 'd', 'e'])
            >>> sorted(it.reduce((lambda accu, item: (accu or set()) | {item} if item in 'aeiou' else accu)))
            ['a', 'e']

        """
        return functools.reduce(bin_op_func, self.iterator, init)

    @typing.overload
    def reduceby(
        self,
        bin_op_func: Callable[[S, T], S],
        key: str,
        *,
        as_dict: Literal[False],
        init: Union[S, NothingType],
    ) -> XIterable[Tuple[str, S]]:
        ...

    @typing.overload
    def reduceby(
        self,
        bin_op_func: Callable[[S, T], S],
        key: str,
        __attr_keys1: str,
        *attr_keys: str,
        as_dict: Literal[False],
        init: Union[S, NothingType],
    ) -> XIterable[Tuple[Tuple[str, ...], S]]:
        ...

    @typing.overload
    def reduceby(
        self,
        bin_op_func: Callable[[S, T], S],
        key: str,
        *,
        as_dict: Literal[True],
        init: Union[S, NothingType],
    ) -> Dict[str, S]:
        ...

    @typing.overload
    def reduceby(
        self,
        bin_op_func: Callable[[S, T], S],
        key: str,
        __attr_keys1: str,
        *attr_keys: str,
        as_dict: Literal[True],
        init: Union[S, NothingType],
    ) -> Dict[Tuple[str, ...], S]:
        ...

    @typing.overload
    def reduceby(
        self,
        bin_op_func: Callable[[S, T], S],
        key: List[K],
        *,
        as_dict: Literal[False],
        init: Union[S, NothingType],
    ) -> XIterable[Tuple[K, S]]:
        ...

    @typing.overload
    def reduceby(
        self,
        bin_op_func: Callable[[S, T], S],
        key: List[K],
        *,
        as_dict: Literal[True],
        init: Union[S, NothingType],
    ) -> Dict[K, S]:
        ...

    @typing.overload
    def reduceby(
        self,
        bin_op_func: Callable[[S, T], S],
        key: Callable[[T], K],
        *,
        as_dict: Literal[False],
        init: Union[S, NothingType],
    ) -> XIterable[Tuple[K, S]]:
        ...

    @typing.overload
    def reduceby(
        self,
        bin_op_func: Callable[[S, T], S],
        key: Callable[[T], K],
        *,
        as_dict: Literal[True],
        init: Union[S, NothingType],
    ) -> Dict[K, S]:
        ...

    def reduceby(  # type: ignore[misc] # signatures 2 and 4 are not satified due to inconsistencies with type variables
        self,
        bin_op_func: Callable[[S, T], S],
        key: Union[str, List[K], Callable[[T], K]],
        *attr_keys: str,
        as_dict: bool = False,
        init: Union[S, NothingType] = NOTHING,
    ) -> Union[
        XIterable[Tuple[str, S]],
        Dict[str, S],
        XIterable[Tuple[Tuple[str, ...], S]],
        Dict[Tuple[str, ...], S],
        XIterable[Tuple[K, S]],
        Dict[K, S],
    ]:
        """Group a sequence by a given key and simultaneously perform a reduction inside the groups.

        More or less equivalent to ``toolz.itertoolz.reduceby(key, bin_op_func, self, init)``
        with some caveats.

        The `key` argument is used in the following way:

            - if `key` is a callable, it will be passed directly to :func:`toolz.itertoolz.reduceby`
              to compute an actual `key` value for each item in the sequence.
            - if `key` is a ``str`` or (multiple ``str`` args) they will be used as
              attributes names (:func:`operator.attrgetter`).
            - if `key` is a ``list`` of values, they will be used as index values
              for :func:`operator.itemgetter`.

        Keyword Arguments:
            init: initial value for the reduction.
            as_dict: if `True`, it will return the groups ``dict`` instead of a :class:`XIterable`
                instance over `groups.items()`.

        For detailed information check :func:`toolz.itertoolz.reduceby` reference.

        Examples:
            >>> it = xiter([(1.0, -1.0), (1.0,-2.0), (2.2, -3.0)])
            >>> list(it.reduceby((lambda accu, _: accu + 1), [0], init=0))
            [(1.0, 2), (2.2, 1)]

            >>> data = [
            ...     {'x': 1.0, 'y': -1.0, 'z': 1.0},
            ...     {'x': 1.0, 'y': -2.0, 'z': 1.0},
            ...     {'x': 2.2, 'y': -3.0, 'z': 2.2}
            ... ]
            >>> list(xiter(data).reduceby((lambda accu, _: accu + 1), ['x'], init=0))
            [(1.0, 2), (2.2, 1)]
            >>> list(xiter(data).reduceby((lambda accu, _: accu + 1), ['x', 'z'], init=0))
            [((1.0, 1.0), 2), ((2.2, 2.2), 1)]

            >>> from collections import namedtuple
            >>> Point = namedtuple('Point', ['x', 'y', 'z'])
            >>> data = [Point(1.0, -2.0, 1.0), Point(1.0, -2.0, 1.0), Point(2.2, 3.0, 2.0)]
            >>> list(xiter(data).reduceby((lambda accu, _: accu + 1), 'x', init=0))
            [(1.0, 2), (2.2, 1)]
            >>> list(xiter(data).reduceby((lambda accu, _: accu + 1), 'x', 'z', init=0))
            [((1.0, 1.0), 2), ((2.2, 2.0), 1)]

            >>> it = xiter(['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank'])
            >>> list(it.reduceby(lambda nvowels, name: nvowels + sum(i in 'aeiou' for i in name), len, init=0))
            [(5, 4), (3, 2), (7, 3)]

        """  # noqa: RST203, RST301  # sphinx.napoleon conventions confuse RST validator
        if (not callable(key) and not isinstance(key, (int, str, list))) or not all(
            isinstance(i, str) for i in attr_keys
        ):
            raise TypeError(f"Invalid 'key' function or attribute name: '{key}'.")
        if callable(key):
            groupby_key = key
        elif isinstance(key, list):
            groupby_key = typing.cast(Callable[[T], K], operator.itemgetter(*key))
        else:
            assert isinstance(key, str)
            groupby_key = typing.cast(Callable[[T], K], operator.attrgetter(key, *attr_keys))

        if init is not NOTHING:
            groups = toolz.itertoolz.reduceby(groupby_key, bin_op_func, self.iterator, init=init)
        else:
            groups = toolz.itertoolz.reduceby(groupby_key, bin_op_func, self.iterator)
        return groups if as_dict else xiter(groups.items())

    def to_list(self) -> List[T]:
        """Expand iterator into a ``list``.

        Equivalent to ``list(self)``.

        Examples:
            >>> it = xiter(range(5))
            >>> it.to_list()
            [0, 1, 2, 3, 4]

        """
        return list(self.iterator)

    def to_set(self) -> Set[T]:
        """Expand iterator into a ``set``.

        Equivalent to ``set(self)``.

        Examples:
            >>> it = xiter([1, 2, 3, 1, 3, -1])
            >>> it.to_set()
            {1, 2, 3, -1}

        """
        return set(self.iterator)


xiter = XIterable
