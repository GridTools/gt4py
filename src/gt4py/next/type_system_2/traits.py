from __future__ import annotations


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