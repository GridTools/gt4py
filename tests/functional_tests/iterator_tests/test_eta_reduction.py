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


def _make_scan(*args: list[str]):
    return ir.FunCall(
        fun=ir.SymRef(id="scan"),
        args=[
            ir.Lambda(
                params=[ir.Sym(id="state")] + [ir.Sym(id=f"{arg}") for arg in args],
                expr=ir.SymRef(id="foo"),
            ),
            ir.Literal(value="0.0", type="float"),
            ir.Literal(value="True", type="bool"),
        ],
    )


def test_extended_eta_reduction_for_scan():
    testee = ir.Lambda(
        params=[ir.Sym(id="x"), ir.Sym(id="y")],
        expr=ir.FunCall(
            fun=_make_scan("param_y", "param_x"),
            args=[ir.SymRef(id="y"), ir.SymRef(id="x")],
        ),
    )
    expected = _make_scan("param_x", "param_y")
    actual = EtaReduction().visit(testee)
    assert actual == expected
