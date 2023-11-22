from gt4py.next.type_system_2 import types
from gt4py.next.type_system_2 import traits


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
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # to BoolType
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # to Int8Type
    [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], # to Int16Type
    [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0], # to Int32Type
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], # to Int64Type
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # to Uint8Type
    [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # to Uint16Type
    [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], # to Uint32Type
    [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], # to Uint64Type
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], # to Float16Type
    [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0], # to Float32Type
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1], # to Float64Type
]


def test_explicit_conversion():
    for from_ty in scalar_types:
        for to_ty in scalar_types:
            assert traits.is_convertible(from_ty, to_ty)


def test_implicit_conversion():
    for from_idx, from_ty in enumerate(scalar_types):
        for to_idx, to_ty in enumerate(scalar_types):
            result = traits.is_implicitly_convertible(from_ty, to_ty)
            expected = implicit_conversion_matrix[to_idx][from_idx] != 0
            assert result == expected