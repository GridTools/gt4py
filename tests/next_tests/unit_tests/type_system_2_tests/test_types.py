from gt4py.next.type_system_2 import types
from gt4py.next.type_system_2 import traits

int32 = types.IntegerType(32, True)
int16 = types.IntegerType(16, True)
float32 = types.FloatType(32)


def test_conversion_int_from_int():
    assert int32.implements(traits.FromTrait(int16))


def test_conversion_int_from_float():
    assert int32.implements(traits.FromTrait(float32))


def test_is_convertible():
    assert traits.is_convertible(int32, int16)
