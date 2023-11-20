from __future__ import annotations
from typing import Any, Callable, Optional


class Trait:
    def satisfied_by(self, _: Trait) -> bool:
        return False


class SignednessTrait(Trait):
    t_signed: bool

    def __init__(self, signed: bool):
        self.t_signed = signed

    def satisfied_by(self, other: Trait) -> bool:
        return isinstance(other, SignednessTrait) and self.t_signed == other.t_signed


class SignedTrait(SignednessTrait):
    def __init__(self):
        SignednessTrait.__init__(self, True)


class UnsignedTrait(SignednessTrait):
    def __init__(self):
        SignednessTrait.__init__(self, False)


class FromTrait(Trait):
    ty: Any
    is_convertible_from: Optional[Callable[[Any], bool]]

    def __init__(self, ty: Any, is_convertible_from: Optional[Callable[[Any], bool]] = None):
        self.ty = ty
        self.is_convertible_from = is_convertible_from

    def satisfied_by(self, other: Trait) -> bool:
        if isinstance(other, FromTrait):
            if other.is_convertible_from is not None:
                return other.is_convertible_from(other.ty)
        return False


class ToTrait(Trait):
    ty: Any
    is_convertible_to: Optional[Callable[[Any], bool]]

    def __init__(self, ty: Any, is_convertible_to: Optional[Callable[[Any], bool]] = None):
        self.ty = ty
        self.is_convertible_to = is_convertible_to

    def satisfied_by(self, other: Trait) -> bool:
        if isinstance(other, FromTrait):
            if self.is_convertible_to is not None:
                return self.is_convertible_to(other.ty)
        return False


def is_convertible(from_: Any, to: Any):
    if isinstance(to, FromTrait):
        return to.satisfied_by(from_)
    elif isinstance(from_, ToTrait):
        return from_.satisfied_by(to)