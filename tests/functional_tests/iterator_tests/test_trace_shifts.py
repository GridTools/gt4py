from functional.iterator import ir
from functional.iterator.transforms.trace_shifts import ALL_NEIGHBORS, TraceShifts


def test_trivial():
    testee = ir.StencilClosure(
        stencil=ir.SymRef(id="deref"),
        inputs=[ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.SymRef(id="dom"),
    )
    expected = {"inp": [()]}

    actual = dict()
    TraceShifts().visit(testee, shifts=actual)
    assert actual == expected


def test_shift():
    testee = ir.StencilClosure(
        stencil=ir.Lambda(
            expr=ir.FunCall(
                fun=ir.SymRef(id="deref"),
                args=[
                    ir.FunCall(
                        fun=ir.FunCall(
                            fun=ir.SymRef(id="shift"),
                            args=[ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1)],
                        ),
                        args=[ir.SymRef(id="x")],
                    )
                ],
            ),
            params=[ir.Sym(id="x")],
        ),
        inputs=[ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.SymRef(id="dom"),
    )
    expected = {"inp": [(ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1))]}

    actual = dict()
    TraceShifts().visit(testee, shifts=actual)
    assert actual == expected


def test_lift():
    testee = ir.StencilClosure(
        stencil=ir.Lambda(
            expr=ir.FunCall(
                fun=ir.SymRef(id="deref"),
                args=[
                    ir.FunCall(
                        fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="deref")]),
                        args=[
                            ir.FunCall(
                                fun=ir.FunCall(
                                    fun=ir.SymRef(id="shift"),
                                    args=[ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1)],
                                ),
                                args=[ir.SymRef(id="x")],
                            )
                        ],
                    )
                ],
            ),
            params=[ir.Sym(id="x")],
        ),
        inputs=[ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.SymRef(id="dom"),
    )
    expected = {"inp": [(ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1))]}

    actual = dict()
    TraceShifts().visit(testee, shifts=actual)
    assert actual == expected


def test_reduce():
    testee = ir.StencilClosure(
        stencil=ir.FunCall(
            fun=ir.SymRef(id="reduce"),
            args=[ir.SymRef(id="plus"), ir.SymRef(id="init")],
        ),
        inputs=[ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.SymRef(id="dom"),
    )
    expected = {"inp": [(ALL_NEIGHBORS,)]}

    actual = dict()
    TraceShifts().visit(testee, shifts=actual)
    assert actual == expected
