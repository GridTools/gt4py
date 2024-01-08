from __future__ import annotations

import abc
from typing import Any, Optional
from collections.abc import Sequence

import dataclasses


class Trait:
    ...


class SignednessTrait(Trait):
    @abc.abstractmethod
    def is_signed(self) -> bool:
        ...


class FromTrait(Trait):
    @abc.abstractmethod
    def is_constructible_from(self, ty: Any) -> bool:
        ...


class ToTrait(Trait):
    @abc.abstractmethod
    def is_convertible_to(self, ty: Any) -> bool:
        ...


class FromImplicitTrait(Trait):
    @abc.abstractmethod
    def is_implicitly_constructible_from(self, ty: Any) -> bool:
        ...


class ToImplicitTrait(Trait):
    @abc.abstractmethod
    def is_implicitly_convertible_to(self, ty: Any) -> bool:
        ...


class ArithmeticTrait(Trait):
    def supports_arithmetic(self) -> bool:
        return True

    def common_arithmetic_type(self, other: Any) -> Optional[Any]:
        return self if self == other else None


class BitwiseTrait(Trait):
    def supports_bitwise(self) -> bool:
        return True

    def common_bitwise_type(self, other: Any) -> Optional[Any]:
        return self if self == other else None


@dataclasses.dataclass(frozen=True)
class FunctionArgument:
    ty: Any
    location: int | str


class CallableTrait(Trait):
    @abc.abstractmethod
    def is_callable(self, args: Sequence[FunctionArgument]) -> tuple[bool, str | Any]:
        ...


def is_convertible(from_: Any, to: Any) -> bool:
    if isinstance(to, FromTrait):
        return to.is_constructible_from(from_)
    elif isinstance(from_, ToTrait):
        return from_.is_convertible_to(to)
    else:
        return False


def is_implicitly_convertible(from_: Any, to: Any) -> bool:
    if isinstance(to, FromImplicitTrait):
        return to.is_implicitly_constructible_from(from_)
    elif isinstance(from_, ToImplicitTrait):
        return from_.is_implicitly_convertible_to(to)
    else:
        return False


def common_type(lhs: Any, rhs: Any) -> Optional[Any]:
    if is_implicitly_convertible(lhs, rhs):
        return rhs
    elif is_implicitly_convertible(rhs, lhs):
        return lhs
    return None


def common_arithmetic_type(lhs: Any, rhs: Any) -> Optional[Any]:
    if isinstance(lhs, ArithmeticTrait):
        ty = lhs.common_arithmetic_type(rhs)
        if ty is not None:
            return ty
    if isinstance(rhs, ArithmeticTrait):
        return rhs.common_arithmetic_type(lhs)
    return None


def common_bitwise_type(lhs: Any, rhs: Any) -> Optional[Any]:
    if isinstance(lhs, BitwiseTrait):
        ty = lhs.common_bitwise_type(rhs)
        if ty is not None:
            return ty
    if isinstance(rhs, BitwiseTrait):
        return rhs.common_bitwise_type(lhs)
    return None
