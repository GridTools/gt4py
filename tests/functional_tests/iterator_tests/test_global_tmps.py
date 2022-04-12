from functional.iterator import ir
from functional.iterator.runtime import CartesianAxis
from functional.iterator.transforms.global_tmps import (
    collect_tmps_info,
    split_closures,
    update_cartesian_domains,
)


def test_split_closures():
    testee = ir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[ir.Sym(id="d"), ir.Sym(id="inp"), ir.Sym(id="out")],
        closures=[
            ir.StencilClosure(
                domain=ir.SymRef(id="d"),
                stencil=ir.Lambda(
                    params=[ir.SymRef(id="baz_inp")],
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
                                                                        args=[
                                                                            ir.SymRef(id="foo_inp")
                                                                        ],
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
                output=ir.Sym(id="out"),
                inputs=[ir.Sym(id="inp")],
            )
        ],
    )

    expected = ir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[
            ir.Sym(id="d"),
            ir.Sym(id="inp"),
            ir.Sym(id="out"),
            ir.Sym(id="_gtmp_0"),
            ir.Sym(id="_gtmp_1"),
            ir.Sym(id="_gtmp_auto_domain"),
        ],
        closures=[
            ir.StencilClosure(
                domain=ir.SymRef(id="_gtmp_auto_domain"),
                stencil=ir.Lambda(
                    params=[ir.Sym(id="foo_inp")],
                    expr=ir.FunCall(
                        fun=ir.SymRef(id="deref"),
                        args=[ir.SymRef(id="foo_inp")],
                    ),
                ),
                output=ir.SymRef(id="_gtmp_1"),
                inputs=[ir.SymRef(id="inp")],
            ),
            ir.StencilClosure(
                domain=ir.SymRef(id="_gtmp_auto_domain"),
                stencil=ir.Lambda(
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
                ),
                output=ir.SymRef(id="_gtmp_0"),
                inputs=[ir.SymRef(id="inp"), ir.SymRef(id="_gtmp_1")],
            ),
            ir.StencilClosure(
                domain=ir.SymRef(id="d"),
                stencil=ir.Lambda(
                    params=[ir.SymRef(id="baz_inp"), ir.SymRef(id="t1")],
                    expr=ir.FunCall(
                        fun=ir.SymRef(id="deref"),
                        args=[ir.SymRef(id="t1")],
                    ),
                ),
                output=ir.SymRef(id="out"),
                inputs=[ir.SymRef(id="inp"), ir.SymRef(id="_gtmp_0")],
            ),
        ],
    )
    actual, tmps = split_closures(testee)
    assert tmps == ["_gtmp_0", "_gtmp_1"]
    assert actual == expected


def test_update_cartesian_domains():
    testee = ir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[
            ir.Sym(id="i"),
            ir.Sym(id="j"),
            ir.Sym(id="k"),
            ir.Sym(id="inp"),
            ir.Sym(id="out"),
            ir.Sym(id="_gtmp_0"),
            ir.Sym(id="_gtmp_1"),
            ir.Sym(id="_gtmp_auto_domain"),
        ],
        closures=[
            ir.StencilClosure(
                domain=ir.SymRef(id="_gtmp_auto_domain"),
                stencil=ir.Lambda(
                    params=[ir.Sym(id="foo_inp")],
                    expr=ir.FunCall(
                        fun=ir.SymRef(id="deref"),
                        args=[ir.SymRef(id="foo_inp")],
                    ),
                ),
                output=ir.SymRef(id="_gtmp_1"),
                inputs=[ir.SymRef(id="inp")],
            ),
            ir.StencilClosure(
                domain=ir.SymRef(id="_gtmp_auto_domain"),
                stencil=ir.SymRef(id="deref"),
                output=ir.SymRef(id="_gtmp_0"),
                inputs=[ir.SymRef(id="_gtmp_1")],
            ),
            ir.StencilClosure(
                domain=ir.FunCall(
                    fun=ir.SymRef(id="domain"),
                    args=[
                        ir.FunCall(
                            fun=ir.SymRef(id="named_range"),
                            args=[
                                ir.AxisLiteral(value=a),
                                ir.IntLiteral(value=0),
                                ir.SymRef(id=s),
                            ],
                        )
                        for a, s in (("IDim", "i"), ("JDim", "j"), ("KDim", "k"))
                    ],
                ),
                stencil=ir.Lambda(
                    params=[ir.SymRef(id="baz_inp"), ir.SymRef(id="t1")],
                    expr=ir.FunCall(
                        fun=ir.SymRef(id="deref"),
                        args=[
                            ir.FunCall(
                                fun=ir.FunCall(
                                    fun=ir.SymRef(id="shift"),
                                    args=[ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1)],
                                ),
                                args=[ir.SymRef(id="t1")],
                            )
                        ],
                    ),
                ),
                output=ir.SymRef(id="out"),
                inputs=[ir.SymRef(id="inp"), ir.SymRef(id="_gtmp_0")],
            ),
        ],
    )
    expected = ir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[
            ir.Sym(id="i"),
            ir.Sym(id="j"),
            ir.Sym(id="k"),
            ir.Sym(id="inp"),
            ir.Sym(id="out"),
            ir.Sym(id="_gtmp_0"),
            ir.Sym(id="_gtmp_1"),
        ],
        closures=[
            ir.StencilClosure(
                domain=ir.FunCall(
                    fun=ir.SymRef(id="domain"),
                    args=[
                        ir.FunCall(
                            fun=ir.SymRef(id="named_range"),
                            args=[
                                ir.AxisLiteral(value="IDim"),
                                ir.IntLiteral(value=0),
                                ir.FunCall(
                                    fun=ir.SymRef(id="plus"),
                                    args=[ir.SymRef(id="i"), ir.IntLiteral(value=1)],
                                ),
                            ],
                        )
                    ]
                    + [
                        ir.FunCall(
                            fun=ir.SymRef(id="named_range"),
                            args=[
                                ir.AxisLiteral(value=a),
                                ir.IntLiteral(value=0),
                                ir.SymRef(id=s),
                            ],
                        )
                        for a, s in (("JDim", "j"), ("KDim", "k"))
                    ],
                ),
                stencil=ir.Lambda(
                    params=[ir.Sym(id="foo_inp")],
                    expr=ir.FunCall(
                        fun=ir.SymRef(id="deref"),
                        args=[ir.SymRef(id="foo_inp")],
                    ),
                ),
                output=ir.SymRef(id="_gtmp_1"),
                inputs=[ir.SymRef(id="inp")],
            ),
            ir.StencilClosure(
                domain=ir.FunCall(
                    fun=ir.SymRef(id="domain"),
                    args=[
                        ir.FunCall(
                            fun=ir.SymRef(id="named_range"),
                            args=[
                                ir.AxisLiteral(value="IDim"),
                                ir.IntLiteral(value=0),
                                ir.FunCall(
                                    fun=ir.SymRef(id="plus"),
                                    args=[ir.SymRef(id="i"), ir.IntLiteral(value=1)],
                                ),
                            ],
                        )
                    ]
                    + [
                        ir.FunCall(
                            fun=ir.SymRef(id="named_range"),
                            args=[
                                ir.AxisLiteral(value=a),
                                ir.IntLiteral(value=0),
                                ir.SymRef(id=s),
                            ],
                        )
                        for a, s in (("JDim", "j"), ("KDim", "k"))
                    ],
                ),
                stencil=ir.SymRef(id="deref"),
                output=ir.SymRef(id="_gtmp_0"),
                inputs=[ir.SymRef(id="_gtmp_1")],
            ),
            ir.StencilClosure(
                domain=ir.FunCall(
                    fun=ir.SymRef(id="domain"),
                    args=[
                        ir.FunCall(
                            fun=ir.SymRef(id="named_range"),
                            args=[
                                ir.AxisLiteral(value=a),
                                ir.IntLiteral(value=0),
                                ir.SymRef(id=s),
                            ],
                        )
                        for a, s in (("IDim", "i"), ("JDim", "j"), ("KDim", "k"))
                    ],
                ),
                stencil=ir.Lambda(
                    params=[ir.SymRef(id="baz_inp"), ir.SymRef(id="t1")],
                    expr=ir.FunCall(
                        fun=ir.SymRef(id="deref"),
                        args=[
                            ir.FunCall(
                                fun=ir.FunCall(
                                    fun=ir.SymRef(id="shift"),
                                    args=[ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1)],
                                ),
                                args=[ir.SymRef(id="t1")],
                            )
                        ],
                    ),
                ),
                output=ir.SymRef(id="out"),
                inputs=[ir.SymRef(id="inp"), ir.SymRef(id="_gtmp_0")],
            ),
        ],
    )
    actual = update_cartesian_domains(testee, {"I": CartesianAxis("IDim")})
    assert actual == expected


def test_collect_tmps_info():
    tmp_domain = ir.FunCall(
        fun=ir.SymRef(id="domain"),
        args=[
            ir.FunCall(
                fun=ir.SymRef(id="named_range"),
                args=[
                    ir.AxisLiteral(value="IDim"),
                    ir.IntLiteral(value=0),
                    ir.FunCall(
                        fun=ir.SymRef(id="plus"),
                        args=[ir.SymRef(id="i"), ir.IntLiteral(value=1)],
                    ),
                ],
            )
        ]
        + [
            ir.FunCall(
                fun=ir.SymRef(id="named_range"),
                args=[
                    ir.AxisLiteral(value=a),
                    ir.IntLiteral(value=0),
                    ir.SymRef(id=s),
                ],
            )
            for a, s in (("JDim", "j"), ("KDim", "k"))
        ],
    )
    testee = ir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[
            ir.Sym(id="i"),
            ir.Sym(id="j"),
            ir.Sym(id="k"),
            ir.Sym(id="inp"),
            ir.Sym(id="out"),
            ir.Sym(id="_gtmp_0"),
            ir.Sym(id="_gtmp_1"),
        ],
        closures=[
            ir.StencilClosure(
                domain=tmp_domain,
                stencil=ir.Lambda(
                    params=[ir.Sym(id="foo_inp")],
                    expr=ir.FunCall(
                        fun=ir.SymRef(id="deref"),
                        args=[ir.SymRef(id="foo_inp")],
                    ),
                ),
                output=ir.SymRef(id="_gtmp_1"),
                inputs=[ir.SymRef(id="inp")],
            ),
            ir.StencilClosure(
                domain=tmp_domain,
                stencil=ir.SymRef(id="deref"),
                output=ir.SymRef(id="_gtmp_0"),
                inputs=[ir.SymRef(id="_gtmp_1")],
            ),
            ir.StencilClosure(
                domain=ir.FunCall(
                    fun=ir.SymRef(id="domain"),
                    args=[
                        ir.FunCall(
                            fun=ir.SymRef(id="named_range"),
                            args=[
                                ir.AxisLiteral(value=a),
                                ir.IntLiteral(value=0),
                                ir.SymRef(id=s),
                            ],
                        )
                        for a, s in (("IDim", "i"), ("JDim", "j"), ("KDim", "k"))
                    ],
                ),
                stencil=ir.Lambda(
                    params=[ir.SymRef(id="baz_inp"), ir.SymRef(id="t1")],
                    expr=ir.FunCall(
                        fun=ir.SymRef(id="deref"),
                        args=[
                            ir.FunCall(
                                fun=ir.FunCall(
                                    fun=ir.SymRef(id="shift"),
                                    args=[ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1)],
                                ),
                                args=[ir.SymRef(id="t1")],
                            )
                        ],
                    ),
                ),
                output=ir.SymRef(id="out"),
                inputs=[ir.SymRef(id="inp"), ir.SymRef(id="_gtmp_0")],
            ),
        ],
    )
    expected = {"_gtmp_0": (tmp_domain, 3), "_gtmp_1": (tmp_domain, 3)}
    actual = collect_tmps_info(testee, ["_gtmp_0", "_gtmp_1"])
    assert actual == expected
