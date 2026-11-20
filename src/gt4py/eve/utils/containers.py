# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom mappings and namespaces."""

from __future__ import annotations

import abc
import collections
import collections.abc
import types
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar


T = TypeVar("T")
_K = TypeVar("_K")
_V = TypeVar("_V")


class CustomDefaultDictBase(collections.defaultdict[_K, _V]):
    """
    Base dict-like class using a value factory to compute default values per key.

    This class is not intended to be used directly, but as a base for other classes.

    Examples:
        >>> class MyDefaultDict(CustomDefaultDictBase):
        ...     def value_factory(self, key):
        ...         return key * 2
        >>> d = MyDefaultDict()
        >>> d[1]
        2
        >>> d[2]
        4
        >>> d[1] = 10
        >>> d[1]
        10

    """

    __slots__ = ()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __missing__(self, key: _K) -> _V:
        self[key] = value = self.value_factory(key)
        return value

    @abc.abstractmethod
    def value_factory(self, key: _K) -> _V:
        raise NotImplementedError


class CustomMapping(collections.abc.MutableMapping[_K, _V]):
    """
    A custom mapping class that uses a key function to map keys to values.

    This class allows for custom key functions to be used for indexing and
    retrieving values, while maintaining a mapping-like interface.

    Examples:
        >>> mapping = CustomMapping(lambda x: hash(repr(x)))
        >>> mapping[[1, 2]] = "first"
        >>> mapping[[1, 2]]
        'first'
        >>> mapping[{1, 2}] = "second"
        >>> mapping[{1, 2}]
        'second'
        >>> len(mapping)
        2
    """

    __slots__ = ("key_func", "key_map", "value_map")

    key_func: Callable[[_K], int]
    key_map: dict[int, _K]
    value_map: dict[int, _V]

    def __init__(self, key: Callable[[_K], int]) -> None:
        self.key_func = key
        self.key_map = {}
        self.value_map = {}

    def __getitem__(self, key: _K) -> _V:
        return self.value_map[self.key_func(key)]

    def __setitem__(self, key: _K, value: _V) -> None:
        data_key = self.key_func(key)
        self.key_map[data_key] = key
        self.value_map[data_key] = value

    def __delitem__(self, key: _K) -> None:
        custom_key = self.key_func(key)
        del self.key_map[custom_key]
        del self.value_map[custom_key]

    def __len__(self) -> int:
        return len(self.key_map)

    def __iter__(self) -> Iterator[_K]:
        for custom_key in self.key_map:
            yield self.key_map[custom_key]

    def __repr__(self) -> str:
        return (
            "{"
            + ", ".join(f"{self.key_map[k]!r}: {self.value_map[k]!r}" for k in self.key_map)
            + "}"
        )

    def internal_key(self, key: _K) -> int:
        """Return the internal key used to store the value associated with `key`."""
        return self.key_func(key)


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

    def items(self) -> Iterable[tuple[str, T]]:
        return self.__dict__.items()

    def keys(self) -> Iterable[str]:
        return self.__dict__.keys()

    def values(self) -> Iterable[T]:
        return self.__dict__.values()

    def reset(self, data: Optional[dict[str, Any]] = None) -> None:
        self.__dict__.clear()
        if data:
            self.__dict__.update(data)

    def as_dict(self) -> dict[str, T]:
        return {**self.__dict__}

    asdict = as_dict

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> T: ...


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
