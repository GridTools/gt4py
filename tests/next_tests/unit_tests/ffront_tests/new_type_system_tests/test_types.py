import pytest

from gt4py.next.ffront.new_type_system import types as types_f
from gt4py.next.new_type_system import types, traits
from gt4py.next import Dimension
from gt4py.next import FieldOffset


i32 = types.Int32Type()
f32 = types.Float32Type()
f64 = types.Float64Type()
IDim = Dimension("IDim")
JDim = Dimension("JDim")
IOff = FieldOffset("IOff", IDim, (IDim,))


def test_dimension_type_str():
    assert str(types_f.DimensionType(IDim)) == "Dimension[IDim]"


def test_field_offset_type_str():
    assert str(types_f.FieldOffsetType(IOff)) == "FieldOffset[IOff]"


def test_field_operator_type_callability():
    # Should only test the semantics of field operator calls.
    # The actual logic is delegated to FunctionType, which is tested extensively.
    # The semantics are the same as functions, hence the short test.
    ty = types_f.FieldOperatorType(
        [
            types.FunctionParameter(f32, "a", True, True),
        ],
        f32
    )
    assert ty.is_callable([types.FunctionArgument(f32, 0)])


@pytest.fixture
def simple_scanop():
    return types_f.ScanOperatorType(
        dimension=IDim,
        carry=f32,
        parameters=[
            types.FunctionParameter(f32, "a", True, True),
            types.FunctionParameter(f32, "b", True, True),
        ],
        result=f32,
    )


@pytest.fixture
def tuple_scanop():
    return types_f.ScanOperatorType(
        dimension=IDim,
        carry=f32,
        parameters=[
            types.FunctionParameter(types.TupleType([f32, f32]), "a", True, True),
        ],
        result=f32,
    )


def test_scan_operator_type_callability_dimensions(simple_scanop):
    field_ty = types_f.FieldType(f32, {IDim})
    args = [
        types.FunctionArgument(field_ty, 0),
        types.FunctionArgument(field_ty, 1),
    ]

    assert simple_scanop.is_callable(args)
    assert simple_scanop.is_callable(args).result == field_ty


def test_scan_operator_type_callability_dimensions_combine(simple_scanop):
    i_field_ty = types_f.FieldType(f32, {IDim})
    j_field_ty = types_f.FieldType(f32, {JDim})
    ij_field_ty = types_f.FieldType(f32, {IDim, JDim})
    args = [
        types.FunctionArgument(i_field_ty, 0),
        types.FunctionArgument(j_field_ty, 1),
    ]

    assert simple_scanop.is_callable(args)
    assert simple_scanop.is_callable(args).result == ij_field_ty


def test_scan_operator_type_callability_dimensions_empty(simple_scanop):
    args = [
        types.FunctionArgument(f32, 0),
        types.FunctionArgument(f32, 1),
    ]

    assert simple_scanop.is_callable(args)
    assert simple_scanop.is_callable(args).result == f32


def test_scan_operator_type_callability_tuple(tuple_scanop):
    i_field_ty = types_f.FieldType(f32, {IDim})
    j_field_ty = types_f.FieldType(f32, {JDim})
    ij_field_ty = types_f.FieldType(f32, {IDim, JDim})

    args = [
        types.FunctionArgument(types.TupleType([i_field_ty, j_field_ty]), 0),
    ]

    assert tuple_scanop.is_callable(args)
    assert tuple_scanop.is_callable(args).result == ij_field_ty


def test_field_type_str():
    ty = types_f.FieldType(f32, {IDim, JDim})
    assert str(ty) == "Field[[IDim, JDim], float32]"


def test_field_dimension_upcast():
    i_field_ty = types_f.FieldType(f32, {IDim})
    j_field_ty = types_f.FieldType(f32, {JDim})
    ij_field_ty = types_f.FieldType(f32, {IDim, JDim})

    assert traits.is_convertible(j_field_ty, j_field_ty)
    assert traits.is_convertible(j_field_ty, ij_field_ty)
    assert not traits.is_convertible(ij_field_ty, j_field_ty)
    assert not traits.is_convertible(i_field_ty, j_field_ty)

    assert traits.is_implicitly_convertible(j_field_ty, j_field_ty)
    assert traits.is_implicitly_convertible(j_field_ty, ij_field_ty)
    assert not traits.is_implicitly_convertible(ij_field_ty, j_field_ty)
    assert not traits.is_implicitly_convertible(i_field_ty, j_field_ty)


def test_field_element_cast():
    f32_field = types_f.FieldType(f32, {IDim})
    f64_field = types_f.FieldType(f64, {IDim})

    assert traits.is_convertible(f32_field, f64_field)
    assert traits.is_convertible(f64_field, f32_field)
    assert traits.is_implicitly_convertible(f32_field, f64_field)
    assert not traits.is_implicitly_convertible(f64_field, f32_field)