import pytest

from gt4py.next.ffront.new_type_system import types as types_f
from gt4py.next.new_type_system import types, traits
from gt4py.next import Dimension, DimensionKind
from gt4py.next import FieldOffset
from gt4py.next.ffront import fbuiltins


i32 = types.Int32Type()
f32 = types.Float32Type()
f64 = types.Float64Type()
IDim = Dimension("IDim")
JDim = Dimension("JDim")
LDim = Dimension("LDim", kind=DimensionKind.LOCAL)
IOff = FieldOffset("IOff", IDim, (IDim,))
i_field_f32 = types_f.FieldType(f32, {IDim})
j_field_f32 = types_f.FieldType(f32, {JDim})
i_field_f64 = types_f.FieldType(f64, {IDim})
ij_field_f32 = types_f.FieldType(f32, {IDim, JDim})


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


def test_field_arithmetic():
    f32_field = types_f.FieldType(f32, {IDim})
    tuple_field = types_f.FieldType(types.TupleType([]), {IDim})
    assert f32_field.supports_arithmetic()
    assert not tuple_field.supports_arithmetic()


def test_field_bitwise():
    i32_field = types_f.FieldType(i32, {IDim})
    tuple_field = types_f.FieldType(types.TupleType([]), {IDim})
    assert i32_field.supports_bitwise()
    assert not tuple_field.supports_bitwise()


def test_field_callability():
    i_field = types_f.FieldType(i32, {IDim})
    jl_field = types_f.FieldType(i32, {JDim, LDim})
    i2i = types_f.FieldOffsetType(FieldOffset("_", IDim, (IDim,)))
    j2j = types_f.FieldOffsetType(FieldOffset("_", JDim, (JDim,)))
    i2jl = types_f.FieldOffsetType(FieldOffset("_", IDim, (JDim, LDim)))
    assert i_field.is_callable([types.FunctionArgument(i2i, 0)])
    assert i_field.is_callable([types.FunctionArgument(i2i, 0)]).result == i_field

    assert not i_field.is_callable([types.FunctionArgument(j2j, 0)])

    assert i_field.is_callable([types.FunctionArgument(i2jl, 0)])
    assert i_field.is_callable([types.FunctionArgument(i2jl, 0)]).result == jl_field


def test_cast_function_str():
    ty = types_f.CastFunctionType(f32)
    assert str(ty) == "(To[float32]) -> float32"


def test_cast_function_callability():
    ty = types_f.CastFunctionType(f32)
    good_ty = f64
    bad_ty = types.TupleType([])

    assert ty.is_callable([types.FunctionArgument(good_ty, 0)])
    assert ty.is_callable([types.FunctionArgument(good_ty, 0)]).result == f32

    assert not ty.is_callable([types.FunctionArgument(bad_ty, 0)])


def test_builtin_broadcast_callability():
    ty = types_f.BuiltinFunctionType(fbuiltins.broadcast)
    superset_dims = types.TupleType([
        types_f.DimensionType(IDim),
        types_f.DimensionType(JDim),
    ])
    subset_dims = types.TupleType([])
    independent_dims = types.TupleType([types_f.DimensionType(JDim)])

    assert ty.is_callable([types.FunctionArgument(i_field_f32, 0), types.FunctionArgument(superset_dims, 1)])
    assert ty.is_callable([types.FunctionArgument(i_field_f32, 0), types.FunctionArgument(superset_dims, 1)]).result == ij_field_f32

    assert not ty.is_callable([types.FunctionArgument(i_field_f32, 0), types.FunctionArgument(subset_dims, 1)])
    assert not ty.is_callable([types.FunctionArgument(i_field_f32, 0), types.FunctionArgument(independent_dims, 1)])


def test_builtin_as_type_callability():
    ty = types_f.BuiltinFunctionType(fbuiltins.astype)
    tuple_ty = types.TupleType([f32])
    field_ty = i_field_f32
    scalar_ty = f32
    target_ty = f64
    failure_ty = types_f.FieldType(types.TupleType([]), {IDim})

    def make_args(source_ty, target_ty):
        return [
            types.FunctionArgument(source_ty, 0),
            types.FunctionArgument(types_f.CastFunctionType(target_ty), 1),
        ]

    assert ty.is_callable(make_args(scalar_ty, target_ty))
    assert ty.is_callable(make_args(scalar_ty, target_ty)).result == f64

    assert ty.is_callable(make_args(tuple_ty, target_ty))
    assert ty.is_callable(make_args(tuple_ty, target_ty)).result == types.TupleType([f64])

    assert ty.is_callable(make_args(field_ty, target_ty))
    assert ty.is_callable(make_args(field_ty, target_ty)).result == i_field_f64

    assert not ty.is_callable(make_args(failure_ty, target_ty))


def test_builtin_reduction_callability():
    ty = types_f.BuiltinFunctionType(fbuiltins.neighbor_sum)
    field_ty = types_f.FieldType(f32, {IDim, LDim})
    idim_ty = types_f.DimensionType(IDim)
    jdim_ty = types_f.DimensionType(JDim)
    ldim_ty = types_f.DimensionType(LDim)

    def make_args(field_ty, dim_ty):
        return [
            types.FunctionArgument(field_ty, 0),
            types.FunctionArgument(dim_ty, "axis"),
        ]

    assert ty.is_callable(make_args(field_ty, ldim_ty))
    assert ty.is_callable(make_args(field_ty, ldim_ty)).result == i_field_f32

    assert not ty.is_callable(make_args(field_ty, idim_ty))
    assert not ty.is_callable(make_args(field_ty, jdim_ty))
    assert not ty.is_callable(make_args(i_field_f32, ldim_ty))


def test_builtin_arithmetic_binop_callability():
    ty = types_f.BuiltinFunctionType(getattr(fbuiltins, "maximum"))

    assert ty.is_callable([types.FunctionArgument(f32, 0), types.FunctionArgument(f64, 1)])
    assert ty.is_callable([types.FunctionArgument(f32, 0), types.FunctionArgument(f64, 1)]).result == f64

    assert not ty.is_callable([types.FunctionArgument(types.TupleType([]), 0), types.FunctionArgument(f32, 1)])


def test_builtin_where_callability():
    ty = types_f.BuiltinFunctionType(fbuiltins.where)

    j_cond_ty = types_f.FieldType(types.BoolType(), {JDim})
    ij_cond_ty = types_f.FieldType(types.BoolType(), {IDim, JDim})
    invalid_cond_ty = types_f.FieldType(types.TupleType([]), {IDim})

    def make_args(cond_ty, lhs_ty, rhs_ty):
        return [
            types.FunctionArgument(cond_ty, 0),
            types.FunctionArgument(lhs_ty, 1),
            types.FunctionArgument(rhs_ty, 2),
        ]

    assert ty.is_callable(make_args(ij_cond_ty, i_field_f32, j_field_f32))
    assert ty.is_callable(make_args(ij_cond_ty, i_field_f32, j_field_f32)).result == ij_field_f32

    assert ty.is_callable(make_args(j_cond_ty, i_field_f32, j_field_f32))
    assert ty.is_callable(make_args(j_cond_ty, i_field_f32, j_field_f32)).result == ij_field_f32

    assert ty.is_callable(make_args(j_cond_ty, i_field_f32, i_field_f32))
    assert ty.is_callable(make_args(j_cond_ty, i_field_f32, i_field_f32)).result == ij_field_f32

    assert not ty.is_callable(make_args(invalid_cond_ty, i_field_f32, i_field_f32))


