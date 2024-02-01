import pytest

from gt4py.next.new_type_system import types
from gt4py.next.new_type_system import traits

f16 = types.Float16Type()
f32 = types.Float32Type()
f64 = types.Float64Type()

scalar_types = [
    types.BoolType(),
    types.Int8Type(),
    types.Int16Type(),
    types.Int32Type(),
    types.Int64Type(),
    types.Uint8Type(),
    types.Uint16Type(),
    types.Uint32Type(),
    types.Uint64Type(),
    types.Float16Type(),
    types.Float32Type(),
    types.Float64Type(),
]

implicit_conversion_matrix = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # to BoolType
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # to Int8Type
    [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # to Int16Type
    [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],  # to Int32Type
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # to Int64Type
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # to Uint8Type
    [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # to Uint16Type
    [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # to Uint32Type
    [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],  # to Uint64Type
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # to Float16Type
    [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],  # to Float32Type
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],  # to Float64Type
]


def test_integer_subtypes():
    assert types.BoolType().width == 1
    assert types.BoolType().signed == False

    assert types.Int8Type().width == 8
    assert types.Int8Type().signed == True
    assert types.Int16Type().width == 16
    assert types.Int16Type().signed == True
    assert types.Int32Type().width == 32
    assert types.Int32Type().signed == True
    assert types.Int64Type().width == 64
    assert types.Int64Type().signed == True

    assert types.Uint8Type().width == 8
    assert types.Uint8Type().signed == False
    assert types.Uint16Type().width == 16
    assert types.Uint16Type().signed == False
    assert types.Uint32Type().width == 32
    assert types.Uint32Type().signed == False
    assert types.Uint64Type().width == 64
    assert types.Uint64Type().signed == False


def test_integer_invalid_size():
    with pytest.raises(AssertionError):
        types.IntegerType(2, False)
    with pytest.raises(AssertionError):
        types.IntegerType(128, False)  # May be added in the future


def test_integer_invalid_bool():
    with pytest.raises(AssertionError):
        types.IntegerType(1, True)


def test_integer_str():
    assert str(types.IntegerType(1, False)) == "bool"
    assert str(types.IntegerType(8, False)) == "uint8"
    assert str(types.IntegerType(64, True)) == "int64"


def test_integer_comparison():
    assert types.IntegerType(8, True) == types.Int8Type()
    assert types.Int8Type() == types.Int8Type()
    assert types.Int8Type() != types.Uint8Type()
    assert types.Int8Type() != types.Int16Type()
    assert types.Int32Type() != types.Float32Type()


def test_float_subtypes():
    assert types.Float16Type().width == 16
    assert types.Float32Type().width == 32
    assert types.Float64Type().width == 64


def test_float_invalid_size():
    with pytest.raises(AssertionError):
        types.FloatType(2)
    with pytest.raises(AssertionError):
        types.FloatType(128)  # May be added in the future.


def test_float_str():
    assert str(types.FloatType(16)) == "float16"
    assert str(types.FloatType(32)) == "float32"
    assert str(types.FloatType(64)) == "float64"


def test_float_comparison():
    assert types.FloatType(16) == types.Float16Type()
    assert types.Float16Type() == types.Float16Type()
    assert types.Float16Type() != types.Float32Type()
    assert types.Float16Type() != types.Int16Type()


def test_explicit_conversion_scalar():
    for from_ty in scalar_types:
        for to_ty in scalar_types:
            assert traits.is_convertible(from_ty, to_ty)


def test_implicit_conversion_scalar():
    for from_idx, from_ty in enumerate(scalar_types):
        for to_idx, to_ty in enumerate(scalar_types):
            result = traits.is_implicitly_convertible(from_ty, to_ty)
            expected = implicit_conversion_matrix[to_idx][from_idx] != 0
            assert result == expected


def test_tuple_str():
    assert str(types.TupleType([types.Int16Type(), types.Float16Type()])) == "tuple[int16, float16]"


def test_explicit_conversion_tuple():
    i32 = types.Int32Type()
    f32 = types.Float32Type()

    # Other type is not a tuple.
    assert not traits.is_convertible(
        types.TupleType([i32, f32]),
        i32,
    )

    # Different tuple length.
    assert not traits.is_convertible(
        types.TupleType([i32, f32]),
        types.TupleType([i32]),
    )

    # All elements convertible.
    assert traits.is_convertible(
        types.TupleType([i32, f32]),
        types.TupleType([i32, f32]),
    )

    # Some elements not convertible.
    assert not traits.is_convertible(
        types.TupleType([i32, f32]),
        types.TupleType([i32, types.TupleType([f32])]),
    )


def test_implicit_conversion_tuple():
    i64 = types.Int64Type()
    i32 = types.Int32Type()
    f32 = types.Float32Type()
    f16 = types.Float16Type()

    # Other type is not a tuple.
    assert not traits.is_implicitly_convertible(
        types.TupleType([i32, f32]),
        i32,
    )

    # Different tuple length.
    assert not traits.is_implicitly_convertible(
        types.TupleType([i32, f32]),
        types.TupleType([i32]),
    )

    # All elements convertible.
    assert traits.is_implicitly_convertible(
        types.TupleType([i32, f32]),
        types.TupleType([i64, f32]),
    )

    # Some elements not convertible.
    assert not traits.is_implicitly_convertible(
        types.TupleType([i32, f32]),
        types.TupleType([i32, f16]),
    )


def test_struct_str():
    ty = types.StructType([("a", types.Int16Type()), ("b", types.Float16Type())])
    assert str(ty) == "struct[a: int16, b: float16]"


def test_struct_eq():
    ty = types.StructType([("a", types.Int16Type()), ("b", types.Float16Type())])
    diff_name = types.StructType([("a", types.Int16Type()), ("c", types.Float16Type())])
    diff_ty = types.StructType([("a", types.Int16Type()), ("b", types.Float32Type())])
    diff_size = types.StructType([("a", types.Int16Type())])
    assert ty == ty
    assert ty != diff_name
    assert ty != diff_ty
    assert ty != diff_size


def test_function_type_str():
    ty = types.FunctionType(
        parameters=[
            types.FunctionParameter(f32, "a", True, False),
            types.FunctionParameter(f32, "b", False, True),
        ],
        result=f32,
    )
    assert str(ty) == "(float32, float32) -> float32"


def test_function_type_callability():
    ty = types.FunctionType(
        parameters=[
            types.FunctionParameter(f32, "a", True, False),
            types.FunctionParameter(f32, "b", False, True),
        ],
        result=f32,
    )

    assert ty.is_callable([
        types.FunctionArgument(f16, 0),
        types.FunctionArgument(f16, "b"),
    ])

    assert not ty.is_callable([
        types.FunctionArgument(f64, 0),
        types.FunctionArgument(f16, "b"),
    ])

    assert not ty.is_callable([
        types.FunctionArgument(f16, 0),
        types.FunctionArgument(f64, "b"),
    ])


def test_common_bitwise_type():
    # Same signed: no extension.
    assert traits.common_bitwise_type(types.Int16Type(), types.Int16Type()) == types.Int16Type()

    # Same sign, different width: extension.
    assert traits.common_bitwise_type(types.Int16Type(), types.Int32Type()) == types.Int32Type()

    # Same unsigned: no extension.
    assert traits.common_bitwise_type(types.Uint16Type(), types.Uint16Type()) == types.Uint16Type()

    # Same sign, different width: extension.
    assert traits.common_bitwise_type(types.Uint16Type(), types.Uint32Type()) == types.Uint32Type()

    # Same width mixed sign: extension + signed.
    assert traits.common_bitwise_type(types.Int16Type(), types.Uint16Type()) == types.Int32Type()

    # Cannot extend above 64 bits width.
    assert traits.common_bitwise_type(types.Int64Type(), types.Uint64Type()) is None


def test_common_artihmetic_type_integer():
    # Same signed: no extension.
    assert traits.common_arithmetic_type(types.Int16Type(), types.Int16Type()) == types.Int16Type()

    # Same sign, different width: extension.
    assert traits.common_arithmetic_type(types.Int16Type(), types.Int32Type()) == types.Int32Type()

    # Same unsigned: no extension.
    assert traits.common_arithmetic_type(types.Uint16Type(), types.Uint16Type()) == types.Uint16Type()

    # Same sign, different width: extension.
    assert traits.common_arithmetic_type(types.Uint16Type(), types.Uint32Type()) == types.Uint32Type()

    # Same width mixed sign: extension + signed.
    assert traits.common_arithmetic_type(types.Int16Type(), types.Uint16Type()) == types.Int32Type()

    # Cannot extend above 64 bits width.
    assert traits.common_arithmetic_type(types.Int64Type(), types.Uint64Type()) is None


def test_common_artihmetic_type_float():
    assert traits.common_arithmetic_type(types.Float16Type(), types.Float16Type()) == types.Float16Type()
    assert traits.common_arithmetic_type(types.Float16Type(), types.Float64Type()) == types.Float64Type()


def test_common_artihmetic_type_mixed():
    # Float mantissa can store entire integer.
    assert traits.common_arithmetic_type(types.Float16Type(), types.Int8Type()) == types.Float16Type()
    assert traits.common_arithmetic_type(types.Float16Type(), types.Uint8Type()) == types.Float16Type()

    # Float mantissa cannot store entire integer.
    assert traits.common_arithmetic_type(types.Float16Type(), types.Int16Type()) == types.Float32Type()
    assert traits.common_arithmetic_type(types.Float16Type(), types.Uint16Type()) == types.Float32Type()

    # Integer is larger than float
    assert traits.common_arithmetic_type(types.Float16Type(), types.Int32Type()) == types.Float64Type()
    assert traits.common_arithmetic_type(types.Float16Type(), types.Uint32Type()) == types.Float64Type()

    # Result would exceed 64 bit precision
    assert traits.common_arithmetic_type(types.Float64Type(), types.Int64Type()) is None
    assert traits.common_arithmetic_type(types.Float64Type(), types.Uint64Type()) is None
