# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""General-purpose classes and class utilities.

Includes custom mappings, namespaces, indexers and ID generators, as well as
helpers to register, inspect and restrict classes.
"""

from __future__ import annotations

import abc
import collections
import collections.abc
import dataclasses
import itertools
import types
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

from ..xtyping import ArgsOnlyCallable


_T = TypeVar("_T")
_S = TypeVar("_S")
T = TypeVar("T")
_K = TypeVar("_K")
_V = TypeVar("_V")


def get_fully_qualified_name(obj: type | types.FunctionType | types.ModuleType) -> str:
    """
    Get the fully qualified name of an object.

    This is useful for creating unique identifiers for objects that can be used
    in fingerprinting or other identification scenarios.
    """
    if isinstance(obj, types.ModuleType):
        return obj.__name__
    return f"{obj.__module__}.{obj.__qualname__}"


def register_subclasses(*subclasses: type[Any]) -> Callable[[type[Any]], type[Any]]:
    """Class decorator to automatically register virtual subclasses.

    Examples:
        >>> import abc
        >>> class MyVirtualSubclassA:
        ...     pass
        >>> class MyVirtualSubclassB:
        ...     pass
        >>> @register_subclasses(MyVirtualSubclassA, MyVirtualSubclassB)
        ... class MyBaseClass(abc.ABC):
        ...     pass
        >>> issubclass(MyVirtualSubclassA, MyBaseClass) and issubclass(
        ...     MyVirtualSubclassB, MyBaseClass
        ... )
        True

    """

    def _decorator(base_cls: type[Any]) -> type[Any]:
        for s in subclasses:
            base_cls.register(s)
        return base_cls

    return _decorator


def noninstantiable(cls: type[_T]) -> type[_T]:
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


def is_noninstantiable(cls: type[_T]) -> bool:
    """Return True if `model` is a non-instantiable class."""
    return "__noninstantiable__" in cls.__dict__


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


@dataclasses.dataclass(frozen=True)
class SequentialIDGenerator:
    """Simple sequential ID generator."""

    prefix: str = ""
    counter: Iterator[int] = dataclasses.field(default_factory=itertools.count)
    #: A string to be used as template for the new ids.
    #: It should contain the `{prefix}`and `{id}` format keys.
    format: str = "{prefix}_{id}"

    def __next__(self) -> str:
        return self.format.format(prefix=self.prefix, id=next(self.counter))

    def next(self) -> str:
        return self.__next__()
