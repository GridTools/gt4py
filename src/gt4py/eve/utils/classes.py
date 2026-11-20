# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for classes and other named objects."""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any, TypeVar


_T = TypeVar("_T")


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
