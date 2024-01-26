import pytest

from gt4py.next.new_type_system import utils
from gt4py.next.new_type_system import types

f16 = types.Float16Type()
f32 = types.Float32Type()


def test_flatten_tuples_not_tuple():
    assert utils.flatten_tuples(f16) == [f16]


def test_flatten_tuples_empty():
    ty = types.TupleType([])
    assert utils.flatten_tuples(ty) == []


def test_flatten_tuples_simple():
    ty = types.TupleType([f16, f32])
    assert utils.flatten_tuples(ty) == [f16, f32]


def test_flatten_tuples_nested():
    ty = types.TupleType([f16, types.TupleType([f32, f16])])
    assert utils.flatten_tuples(ty) == [f16, f32, f16]


def test_unflatten_tuples_not_tuple():
    marker = types.Type()
    tys = [f16]
    assert utils.unflatten_tuples(tys, marker) == f16


def test_unflatten_tuples_not_empty():
    tys = []
    structure = types.TupleType([])
    assert utils.unflatten_tuples(tys, structure) == types.TupleType([])


def test_unflatten_tuples_simple():
    marker = types.Type()
    tys = [f16, f32]
    structure = types.TupleType([marker, marker])
    expected = types.TupleType([f16, f32])
    assert utils.unflatten_tuples(tys, structure) == expected


def test_unflatten_tuples_nested():
    marker = types.Type()
    tys = [f16, f32, f16]
    structure = types.TupleType([marker, types.TupleType([marker, marker])])
    expected = types.TupleType([f16, types.TupleType([f32, f16])])
    assert utils.unflatten_tuples(tys, structure) == expected


def test_unflatten_tuples_underflow():
    marker = types.Type()
    tys = [f16, f32]
    structure = types.TupleType([marker, marker, marker])
    with pytest.raises(ValueError):
        utils.unflatten_tuples(tys, structure)


def test_unflatten_tuples_overflow():
    marker = types.Type()
    tys = [f16, f32, f16]
    structure = types.TupleType([marker, marker])
    with pytest.raises(ValueError):
        utils.unflatten_tuples(tys, structure)


