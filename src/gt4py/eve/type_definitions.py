# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions of useful field and general types."""

from __future__ import annotations

import abc
import re
from enum import Enum as Enum, IntEnum as IntEnum

from boltons.typeutils import classproperty as classproperty

from .extended_typing import Any, ClassVar, NoReturn, Optional, Tuple, TypeVar, final


# -- Frozen collections --
_Tc = TypeVar("_Tc", covariant=True)


class FrozenList(Tuple[_Tc, ...], metaclass=abc.ABCMeta):
    """Tuple subtype which works as an alias of ``Tuple[_Tc, ...]``."""

    __slots__ = ()

    @classmethod
    def __subclasshook__(cls, C: type) -> bool:
        return tuple in C.__mro__


# -- Sentinels --


@final
class NothingType(type):
    """Metaclass of :class:`NOTHING` setting its bool value to False."""

    def __bool__(cls) -> bool:
        return False


@final
class NOTHING(metaclass=NothingType):
    """Marker to avoid confusion with `None` in contexts where `None` could be a valid value."""

    def __new__(cls: type) -> NoReturn:
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
        >>> class OnlyLetters(ConstrainedStr, regex=re.compile(r"^[a-zA-Z]*$")):
        ...     pass
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
