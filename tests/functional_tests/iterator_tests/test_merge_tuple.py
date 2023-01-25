import pytest

from functional.iterator import ir
from functional.iterator.transforms.merge_tuple import MergeTuple


def _tuple_get(i: int, t: ir.Expr):
    return ir.FunCall(fun=ir.SymRef(id="tuple_get"), args=[ir.Literal(value=str(i), type="int"), t])


@pytest.fixture
def tup():
    return ir.SymRef(id="foo")


def test_simple(tup):
    testee = ir.FunCall(
        fun=ir.SymRef(id="make_tuple"), args=[_tuple_get(0, tup), _tuple_get(1, tup)]
    )
    expected = tup
    actual = MergeTuple().visit(testee)
    assert actual == expected


def test_incompatible_order(tup):
    testee = ir.FunCall(
        fun=ir.SymRef(id="make_tuple"), args=[_tuple_get(1, tup), _tuple_get(0, tup)]
    )
    actual = MergeTuple().visit(testee)
    assert actual == testee  # did nothing
