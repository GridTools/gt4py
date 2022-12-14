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

"""Definitions of useful field and general types."""


from __future__ import annotations

import abc
import re
import sys
from enum import Enum as Enum  # noqa: F401  # imported but unused
from enum import IntEnum as IntEnum  # noqa: F401  # imported but unused

from boltons.typeutils import classproperty as classproperty  # type: ignore[import]  # noqa: F401
from frozendict import frozendict as _frozendict  # type: ignore[attr-defined]  # noqa: F401

from .extended_typing import (
    Any,
    ClassVar,
    Generic,
    NoReturn,
    Optional,
    Tuple,
    TypeAlias,
    TypeVar,
    final,
)


# -- Frozen collections --
_T = TypeVar("_T")
_Tc = TypeVar("_Tc", covariant=True)


class FrozenList(Tuple[_Tc, ...], metaclass=abc.ABCMeta):  # noqa: B024   # no abstract methods
    """Tuple subtype which works as an alias of ``Tuple[_Tc, ...]``."""

    __slots__ = ()

    @classmethod
    def __subclasshook__(cls, C: type) -> bool:
        return tuple in C.__mro__


if sys.version_info >= (3, 9):
    frozendict: TypeAlias = _frozendict
else:
    _KeyT = TypeVar("_KeyT")

    @final
    class frozendict(_frozendict, Generic[_KeyT, _T]):
        __slots__ = ()


# -- Sentinels --


@final
class NothingType(type):
    """Metaclass of :class:`NOTHING` setting its bool value to False."""

    def __bool__(cls) -> bool:
        return False


@final
class NOTHING(metaclass=NothingType):
    """Marker to avoid confusion with `None` in contexts where `None` could be a valid value."""

    def __new__(cls: type) -> NoReturn:  # type: ignore[misc]  # should return an instance
        raise TypeError(f"{cls.__name__} is used as a sentinel value and cannot be instantiated.")


# -- Others --
class StrEnum(str, Enum):
    """:class:`enum.Enum` subclass whose members are considered as real strings."""

    def __str__(self) -> str:
        assert isinstance(self.value, str)
        return self.value


class ConstrainedStr(str):
    """Base string subclass allowing to restrict values to those satisfying a regular expression.

    Subclasses should define the specific constraint pattern in the ``regex``
    class keyword argument or as class variable.

    Examples:
        >>> class OnlyLetters(ConstrainedStr, regex=re.compile(r"^[a-zA-Z]*$")): pass
        >>> OnlyLetters("aabbCC")
        OnlyLetters('aabbCC')

        >>> OnlyLetters("aabbCC33")
        Traceback (most recent call last):
            ...
        ValueError: OnlyLetters('aabbCC33') does not satisfies RE constraint re.compile('^[a-zA-Z]*$').

    """

    __slots__ = ()

    regex: ClassVar[re.Pattern]

    def __new__(cls, value: str) -> ConstrainedStr:
        if cls is ConstrainedStr:
            raise TypeError(f"{cls} cannot be directly instantiated, it should be subclassed.")
        if not isinstance(value, str) or not cls.regex.fullmatch(value):
            raise ValueError(
                f"{cls.__name__}('{value}') does not satisfies RE constraint {cls.regex}."
            )
        return super().__new__(cls, value)

    def __init_subclass__(cls, *, regex: Optional[re.Pattern] = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if regex is None and "regex" in cls.__dict__:
            # regex has been defined as a class var either in this class or in the parents
            assert isinstance(cls.regex, re.Pattern)
            return
        if not isinstance(regex, re.Pattern):
            raise TypeError(
                f"Invalid regex pattern ({regex}) for '{cls.__name__}' ConstrainedStr subclass."
            )
        cls.regex = regex

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"
