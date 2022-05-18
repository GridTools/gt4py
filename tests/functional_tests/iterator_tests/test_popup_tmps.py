from functional.iterator import ir
from functional.iterator.transforms.popup_tmps import PopupTmps


def test_trivial_single_lift():
    testee = ir.FunCall(
        fun=ir.Lambda(
            params=[ir.Sym(id="bar_inp")],
            expr=ir.FunCall(
                fun=ir.SymRef(id="deref"),
                args=[
                    ir.FunCall(
                        fun=ir.FunCall(
                            fun=ir.SymRef(id="lift"),
                            args=[
                                ir.Lambda(
                                    params=[ir.Sym(id="foo_inp")],
                                    expr=ir.FunCall(
                                        fun=ir.SymRef(id="deref"),
                                        args=[ir.SymRef(id="foo_inp")],
                                    ),
                                )
                            ],
                        ),
                        args=[ir.SymRef(id="bar_inp")],
                    )
                ],
            ),
        ),
        args=[ir.SymRef(id="inp")],
    )
    expected = ir.FunCall(
        fun=ir.Lambda(
            params=[ir.Sym(id="bar_inp"), ir.Sym(id="t0")],
            expr=ir.FunCall(
                fun=ir.SymRef(id="deref"),
                args=[ir.SymRef(id="t0")],
            ),
        ),
        args=[
            ir.SymRef(id="inp"),
            ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="lift"),
                    args=[
                        ir.Lambda(
                            params=[ir.Sym(id="foo_inp")],
                            expr=ir.FunCall(
                                fun=ir.SymRef(id="deref"),
                                args=[ir.SymRef(id="foo_inp")],
                            ),
                        )
                    ],
                ),
                args=[ir.SymRef(id="inp")],
            ),
        ],
    )
    actual = PopupTmps().visit(testee)
    assert actual == expected


def test_trivial_multiple_lifts():
    testee = ir.FunCall(
        fun=ir.Lambda(
            params=[ir.Sym(id="baz_inp")],
            expr=ir.FunCall(
                fun=ir.SymRef(id="deref"),
                args=[
                    ir.FunCall(
                        fun=ir.FunCall(
                            fun=ir.SymRef(id="lift"),
                            args=[
                                ir.Lambda(
                                    params=[ir.Sym(id="bar_inp")],
                                    expr=ir.FunCall(
                                        fun=ir.SymRef(id="deref"),
                                        args=[
                                            ir.FunCall(
                                                fun=ir.FunCall(
                                                    fun=ir.SymRef(id="lift"),
                                                    args=[
                                                        ir.Lambda(
                                                            params=[ir.Sym(id="foo_inp")],
                                                            expr=ir.FunCall(
                                                                fun=ir.SymRef(id="deref"),
                                                                args=[ir.SymRef(id="foo_inp")],
                                                            ),
                                                        )
                                                    ],
                                                ),
                                                args=[ir.SymRef(id="bar_inp")],
                                            )
                                        ],
                                    ),
                                )
                            ],
                        ),
                        args=[ir.SymRef(id="baz_inp")],
                    )
                ],
            ),
        ),
        args=[ir.SymRef(id="inp")],
    )
    expected = ir.FunCall(
        fun=ir.Lambda(
            params=[ir.Sym(id="baz_inp"), ir.Sym(id="t1")],
            expr=ir.FunCall(
                fun=ir.SymRef(id="deref"),
                args=[ir.SymRef(id="t1")],
            ),
        ),
        args=[
            ir.SymRef(id="inp"),
            ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="lift"),
                    args=[
                        ir.Lambda(
                            params=[
                                ir.Sym(id="bar_inp"),
                                ir.Sym(id="t0"),
                            ],
                            expr=ir.FunCall(
                                fun=ir.SymRef(id="deref"),
                                args=[
                                    ir.SymRef(id="t0"),
                                ],
                            ),
                        )
                    ],
                ),
                args=[
                    ir.SymRef(id="inp"),
                    ir.FunCall(
                        fun=ir.FunCall(
                            fun=ir.SymRef(id="lift"),
                            args=[
                                ir.Lambda(
                                    params=[ir.Sym(id="foo_inp")],
                                    expr=ir.FunCall(
                                        fun=ir.SymRef(id="deref"),
                                        args=[ir.SymRef(id="foo_inp")],
                                    ),
                                )
                            ],
                        ),
                        args=[ir.SymRef(id="inp")],
                    ),
                ],
            ),
        ],
    )
    actual = PopupTmps().visit(testee)
    assert actual == expected
