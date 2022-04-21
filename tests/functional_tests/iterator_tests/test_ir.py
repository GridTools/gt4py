import pytest

from functional.iterator import ir


def test_noninstantiable():
    with pytest.raises(TypeError, match="non-instantiable"):
        ir.Node()
    with pytest.raises(TypeError, match="non-instantiable"):
        ir.Expr()


def test_str():
    testee = ir.Lambda(params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
    expected = "λ(x) → x"
    actual = str(testee)
    assert actual == expected
