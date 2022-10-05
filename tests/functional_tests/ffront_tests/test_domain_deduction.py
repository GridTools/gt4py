import pytest

from functional.common import Dimension, DimensionKind, GridType, GTTypeError
from functional.ffront.decorator import _deduce_grid_type
from functional.ffront.fbuiltins import FieldOffset


Dim = Dimension("Dim")
LocalDim = Dimension("LocalDim", kind=DimensionKind.LOCAL)

CartesianOffset = FieldOffset("CartesianOffset", source=Dim, target=(Dim,))
UnstructuredOffset = FieldOffset("UnstructuredOffset", source=Dim, target=(Dim, LocalDim))


def test_domain_deduction_cartesian():
    assert _deduce_grid_type(None, {CartesianOffset}) == GridType.CARTESIAN
    assert _deduce_grid_type(None, {Dim}) == GridType.CARTESIAN


def test_domain_deduction_unstructured():
    assert _deduce_grid_type(None, {UnstructuredOffset}) == GridType.UNSTRUCTURED
    assert _deduce_grid_type(None, {LocalDim}) == GridType.UNSTRUCTURED


def test_domain_complies_with_request_cartesian():
    assert _deduce_grid_type(GridType.CARTESIAN, {CartesianOffset}) == GridType.CARTESIAN
    with pytest.raises(GTTypeError, match="unstructured.*FieldOffset.*found"):
        _deduce_grid_type(GridType.CARTESIAN, {UnstructuredOffset})
        _deduce_grid_type(GridType.CARTESIAN, {LocalDim})


def test_domain_complies_with_request_unstructured():
    assert _deduce_grid_type(GridType.UNSTRUCTURED, {UnstructuredOffset}) == GridType.UNSTRUCTURED
    # unstructured is ok, even if we don't have unstructured offsets
    assert _deduce_grid_type(GridType.UNSTRUCTURED, {CartesianOffset}) == GridType.UNSTRUCTURED
