from functional.iterator import ir
from functional.iterator.transforms.prune_closure_inputs import PruneClosureInputs


def test_simple():
    testee = ir.StencilClosure(
        domain=ir.SymRef(id="d"),
        stencil=ir.Lambda(
            params=[ir.Sym(id="x"), ir.Sym(id="y"), ir.Sym(id="z")],
            expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="y")]),
        ),
        output=ir.SymRef(id="out"),
        inputs=[ir.SymRef(id="foo"), ir.SymRef(id="bar"), ir.SymRef(id="baz")],
    )
    expected = ir.StencilClosure(
        domain=ir.SymRef(id="d"),
        stencil=ir.Lambda(
            params=[ir.Sym(id="y")],
            expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="y")]),
        ),
        output=ir.SymRef(id="out"),
        inputs=[ir.SymRef(id="bar")],
    )
    actual = PruneClosureInputs().visit(testee)
    assert actual == expected


def test_shadowing():
    testee = ir.StencilClosure(
        domain=ir.SymRef(id="d"),
        stencil=ir.Lambda(
            params=[ir.Sym(id="x"), ir.Sym(id="y"), ir.Sym(id="z")],
            expr=ir.FunCall(
                fun=ir.Lambda(
                    params=[ir.SymRef(id="z")],
                    expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="z")]),
                ),
                args=[ir.SymRef(id="y")],
            ),
        ),
        output=ir.SymRef(id="out"),
        inputs=[ir.SymRef(id="foo"), ir.SymRef(id="bar"), ir.SymRef(id="baz")],
    )
    expected = ir.StencilClosure(
        domain=ir.SymRef(id="d"),
        stencil=ir.Lambda(
            params=[ir.Sym(id="y")],
            expr=ir.FunCall(
                fun=ir.Lambda(
                    params=[ir.SymRef(id="z")],
                    expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="z")]),
                ),
                args=[ir.SymRef(id="y")],
            ),
        ),
        output=ir.SymRef(id="out"),
        inputs=[ir.SymRef(id="bar")],
    )
    actual = PruneClosureInputs().visit(testee)
    assert actual == expected
