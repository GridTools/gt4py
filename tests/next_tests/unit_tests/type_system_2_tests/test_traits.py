from gt4py.next.type_system_2 import traits
from gt4py.next.type_system_2 import types


def test_signedness_trait_same():
    assert traits.SignedTrait().satisfied_by(traits.SignedTrait())
    assert traits.UnsignedTrait().satisfied_by(traits.UnsignedTrait())

def test_signedness_trait_different():
    assert not traits.SignedTrait().satisfied_by(traits.UnsignedTrait())
    assert not traits.UnsignedTrait().satisfied_by(traits.SignedTrait())

def test_type_implements():
    assert types.IntegerType(32, True).implements(traits.SignedTrait())
    assert types.IntegerType(32, False).implements(traits.UnsignedTrait())