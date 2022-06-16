from functional.iterator import ir
from functional.iterator.transforms.eta_reduction import EtaReduction


def test_simple():
    testee = ir.Lambda(
        params=[ir.Sym(id="x")],
        expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x")]),
    )
    expected = ir.SymRef(id="deref")
    actual = EtaReduction().visit(testee)
    assert actual == expected
