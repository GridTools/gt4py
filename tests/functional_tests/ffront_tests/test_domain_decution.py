import pytest

from functional.common import Dimension, GridType, GTTypeError
from functional.ffront.decorator import Program
from functional.ffront.fbuiltins import FieldOffset


Dim = Dimension("Dim")
LocalDim = Dimension("LocalDim", local=True)

CartesianOffset = FieldOffset("CartesianOffset", source=Dim, target=(Dim,))
UnstructuredOffset = FieldOffset("UnstructuredOffset", source=Dim, target=(Dim, LocalDim))


def test_domain_deduction():
    assert Program._deduce_grid_type(GridType.CARTESIAN, {CartesianOffset}) == GridType.CARTESIAN
    assert (
        Program._deduce_grid_type(GridType.UNSTRUCTURED, {UnstructuredOffset})
        == GridType.UNSTRUCTURED
    )
    assert Program._deduce_grid_type(None, {CartesianOffset}) == GridType.CARTESIAN
    assert Program._deduce_grid_type(None, {Dim}) == GridType.CARTESIAN
    assert Program._deduce_grid_type(None, {UnstructuredOffset}) == GridType.UNSTRUCTURED
    assert Program._deduce_grid_type(None, {LocalDim}) == GridType.UNSTRUCTURED

    # unstructured is ok, even if we don't have unstructured offsets
    assert (
        Program._deduce_grid_type(GridType.UNSTRUCTURED, {CartesianOffset}) == GridType.UNSTRUCTURED
    )

    with pytest.raises(GTTypeError, match="unstructured.*FieldOffset.*found"):
        Program._deduce_grid_type(GridType.CARTESIAN, {UnstructuredOffset})
        Program._deduce_grid_type(GridType.CARTESIAN, {LocalDim})
