import pytest

from functional.common import Dimension, DimensionKind, GridType, GTTypeError
from functional.ffront.fbuiltins import FieldOffset
from functional.ffront.past_to_itir import _deduce_grid_type
from functional.ffront.symbol_makers import make_symbol_type_from_value


Dim = Dimension("Dim")
LocalDim = Dimension("LocalDim", kind=DimensionKind.LOCAL)

CartesianOffset = FieldOffset("CartesianOffset", source=Dim, target=(Dim,))
UnstructuredOffset = FieldOffset("UnstructuredOffset", source=Dim, target=(Dim, LocalDim))


@pytest.mark.parametrize(
    "input,expected",
    [
        ({CartesianOffset}, GridType.CARTESIAN),
        ({Dim}, GridType.CARTESIAN),
        ({UnstructuredOffset}, GridType.UNSTRUCTURED),
        ({LocalDim}, GridType.UNSTRUCTURED),
    ],
)
def test_domain_deduction(input: set[Dimension | FieldOffset], expected: GridType):
    inputs_as_type = {make_symbol_type_from_value(el) for el in input}
    assert _deduce_grid_type(inputs_as_type) == expected
