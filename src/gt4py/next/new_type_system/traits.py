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

from __future__ import annotations

import abc
import dataclasses
import typing
from collections.abc import Sequence
from typing import Optional

from .base import FunctionArgument, Trait, Type


__all__ = ["Trait"]


class SignednessTrait(Trait, abc.ABC):
    """Specifies the signedness of a type."""

    @abc.abstractmethod
    def is_signed(self) -> bool: ...


class FromTrait(Trait, abc.ABC):
    """
    Specifies if a type is constructible FROM another type.

    This trait represents explicit conversions and an implementation should return
    True if the conversion is possible, even if it is losing arithmetic precision.
    """

    @abc.abstractmethod
    def is_constructible_from(self, ty: Type) -> bool: ...


class ToTrait(Trait, abc.ABC):
    """
    Specifies if a type is convertible TO another type.

    This trait represents explicit conversions and an implementation should return
    True if the conversion is possible, even if it is losing arithmetic precision.
    """

    @abc.abstractmethod
    def is_convertible_to(self, ty: Type) -> bool: ...


class FromImplicitTrait(Trait, abc.ABC):
    """
    Specifies if a type is IMPLICITLY constructible FROM another type.

    This trait represents implicit conversions and an implementation should return
    True if and only if the conversion is possible AND does not lose arithmetic
    precision.
    """

    @abc.abstractmethod
    def is_implicitly_constructible_from(self, ty: Type) -> bool: ...


class ToImplicitTrait(Trait, abc.ABC):
    """
    Specifies if a type is IMPLICITLY convertible TO another type.

    This trait represents implicit conversions and an implementation should return
    True if and only if the conversion is possible AND does not lose arithmetic
    precision.
    """

    @abc.abstractmethod
    def is_implicitly_convertible_to(self, ty: Type) -> bool: ...


class ArithmeticTrait(Trait):
    """
    Specifies if a type supports arithmetic operations (i.e. +, -).

    A type that implements this trait does NOT necessarily support arithmetic.
    Actual support is determined via the member functions.
    """

    def supports_arithmetic(self) -> bool:
        """Check if the type supports arithmetic operations."""
        return True

    def common_arithmetic_type(self, other: Type) -> Optional[Type]:
        """
        Find a common type that can store the result without losing precision.

        If no such type exists or the operation cannot be performed, None is
        returned.
        """
        return self if self == other and isinstance(self, Type) else None


class BitwiseTrait(Trait):
    """
    Specifies if a type supports bitwise operations (i.e. &, |).

    A type that implements this trait does NOT necessarily support bitwise ops.
    Actual support is determined via the member functions.
    """

    def supports_bitwise(self) -> bool:
        """Check if the type supports bitwise operations."""
        return True

    def common_bitwise_type(self, other: Type) -> Optional[Type]:
        """
        Find a common type that can store the result without losing precision.

        If no such type exists or the operation cannot be performed, None is
        returned.
        """
        return self if self == other and isinstance(self, Type) else None


@dataclasses.dataclass(frozen=True)
class CallValidity:
    """Holds information about the validity of a function call."""

    _value: Optional[Type] | list[str]
    """
    Either the returned type or the list of errors.

    In case the function call is valid, it contains the result produced by the
    call. In case the function call is NOT valid, it contains the list of
    error diagnostics that make the function call invalid, such as incorrect
    argument types.
    """

    def __bool__(self):
        """Check if the function call is valid."""
        return isinstance(self._value, Type) or self._value is None

    @property
    def result(self) -> Optional[Type]:
        """If the call is valid, return the result's type."""
        assert bool(self)
        return typing.cast(Optional[Type], self._value)

    @property
    def errors(self) -> list[str]:
        """If the call is invalid, return the error diagnostics."""
        assert not bool(self)
        return typing.cast(list[str], self._value)


class CallableTrait(Trait, abc.ABC):
    """Specifies if a type is callable with a given set of arguments."""

    @abc.abstractmethod
    def is_callable(self, args: Sequence[FunctionArgument]) -> CallValidity:
        """Determine if the type is callable with a given set of arguments."""
        ...


def is_convertible(from_: Type, to: Type) -> bool:
    """
    Check if a type is convertible to another.

    Check if the types implement the conversion traits and queries if either
    type reports conversion compatibility with the other.
    """
    if isinstance(to, FromTrait):
        return to.is_constructible_from(from_)
    elif isinstance(from_, ToTrait):
        return from_.is_convertible_to(to)
    else:
        return False


def is_implicitly_convertible(from_: Type, to: Type) -> bool:
    """
    Check if a type is IMPLICITLY convertible to another.

    Check if the types implement the conversion traits and queries if either
    type reports IMPLICIT conversion compatibility with the other.
    """
    if isinstance(to, FromImplicitTrait):
        return to.is_implicitly_constructible_from(from_)
    elif isinstance(from_, ToImplicitTrait):
        return from_.is_implicitly_convertible_to(to)
    else:
        return False


def common_type(lhs: Type, rhs: Type) -> Optional[Type]:
    """
    Find a type that both arguments can be converted to.

    Note that the result type candidates are only the two provided types.
    If a common type cannot be found, None is returned.
    """
    if is_implicitly_convertible(lhs, rhs):
        return rhs
    elif is_implicitly_convertible(rhs, lhs):
        return lhs
    return None


def common_arithmetic_type(lhs: Type, rhs: Type) -> Optional[Type]:
    """
    Find a type that can store the result of an arithmetic operation.

    The returned type can represent the result without precision loss.
    If not such type is found, None is returned.
    """
    if isinstance(lhs, ArithmeticTrait):
        ty = lhs.common_arithmetic_type(rhs)
        if ty is not None:
            return ty
    if isinstance(rhs, ArithmeticTrait):
        return rhs.common_arithmetic_type(lhs)
    return None


def common_bitwise_type(lhs: Type, rhs: Type) -> Optional[Type]:
    """
    Find a type that can store the result of a bitwise operation.

    The returned type can represent the result without precision loss.
    If not such type is found, None is returned.
    """
    if isinstance(lhs, BitwiseTrait):
        ty = lhs.common_bitwise_type(rhs)
        if ty is not None:
            return ty
    if isinstance(rhs, BitwiseTrait):
        return rhs.common_bitwise_type(lhs)
    return None
