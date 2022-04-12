from functional.iterator import ir
from functional.iterator import type_inference as ti


def test_sym_ref():
    testee = ir.SymRef(id="x")
    expected = ti.Var(0)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "T₀"


def test_bool_literal():
    testee = ir.BoolLiteral(value=False)
    expected = ti.Val(ti.Value(), ti.Primitive("bool"), ti.Var(0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "bool⁰"


def test_int_literal():
    testee = ir.IntLiteral(value=False)
    expected = ti.Val(ti.Value(), ti.Primitive("int"), ti.Var(0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "int⁰"


def test_float_literal():
    testee = ir.FloatLiteral(value=False)
    expected = ti.Val(ti.Value(), ti.Primitive("float"), ti.Var(0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "float⁰"


def test_deref():
    testee = ir.SymRef(id="deref")
    expected = ti.Fun(
        ti.Tuple((ti.Val(ti.Iterator(), ti.Var(0), ti.Var(1)),)),
        ti.Val(ti.Value(), ti.Var(0), ti.Var(1)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(It[T₀¹]) → T₀¹"


def test_deref_call():
    testee = ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x")])
    expected = ti.Val(ti.Value(), ti.Var(0), ti.Var(1))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "T₀¹"


def test_lambda():
    testee = ir.Lambda(params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
    expected = ti.Fun(ti.Tuple((ti.Var(0),)), ti.Var(0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(T₀) → T₀"


def test_plus():
    testee = ir.SymRef(id="plus")
    t = ti.Val(ti.Value(), ti.Var(0), ti.Var(1))
    expected = ti.Fun(ti.Tuple((t, t)), t)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(T₀¹, T₀¹) → T₀¹"


def test_eq():
    testee = ir.SymRef(id="eq")
    t = ti.Val(ti.Value(), ti.Var(0), ti.Var(1))
    expected = ti.Fun(ti.Tuple((t, t)), ti.Val(ti.Value(), ti.Primitive("bool"), ti.Var(1)))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(T₀¹, T₀¹) → bool¹"


def test_if():
    testee = ir.SymRef(id="if_")
    c = ti.Val(ti.Value(), ti.Primitive("bool"), ti.Var(0))
    t = ti.Val(ti.Value(), ti.Var(1), ti.Var(0))
    expected = ti.Fun(ti.Tuple((c, t, t)), t)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(bool⁰, T₁⁰, T₁⁰) → T₁⁰"


def test_not():
    testee = ir.SymRef(id="not_")
    t = ti.Val(ti.Value(), ti.Primitive("bool"), ti.Var(0))
    expected = ti.Fun(ti.Tuple((t,)), t)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(bool⁰) → bool⁰"


def test_and():
    testee = ir.SymRef(id="and_")
    t = ti.Val(ti.Value(), ti.Primitive("bool"), ti.Var(0))
    expected = ti.Fun(ti.Tuple((t, t)), t)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(bool⁰, bool⁰) → bool⁰"


def test_lift():
    testee = ir.SymRef(id="lift")
    expected = ti.Fun(
        ti.Tuple(
            (
                ti.Fun(
                    ti.ValTuple(ti.Iterator(), ti.Var(0), ti.Var(1)),
                    ti.Val(ti.Value(), ti.Var(2), ti.Var(1)),
                ),
            )
        ),
        ti.Fun(
            ti.ValTuple(ti.Iterator(), ti.Var(0), ti.Var(1)),
            ti.Val(ti.Iterator(), ti.Var(2), ti.Var(1)),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "((It[T¹], …)₀ → T₂¹) → (It[T¹], …)₀ → It[T₂¹]"


def test_lift_application():
    testee = ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="deref")])
    expected = ti.Fun(
        ti.Tuple((ti.Val(ti.Iterator(), ti.Var(0), ti.Var(1)),)),
        ti.Val(ti.Iterator(), ti.Var(0), ti.Var(1)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(It[T₀¹]) → It[T₀¹]"


def test_lifted_call():
    testee = ir.FunCall(
        fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="deref")]),
        args=[ir.SymRef(id="x")],
    )
    expected = ti.Val(ti.Iterator(), ti.Var(0), ti.Var(1))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "It[T₀¹]"


def test_make_tuple():
    testee = ir.FunCall(
        fun=ir.SymRef(id="make_tuple"),
        args=[ir.BoolLiteral(value=True), ir.FloatLiteral(value=42.0), ir.SymRef(id="x")],
    )
    expected = ti.Val(
        ti.Value(), ti.Tuple((ti.Primitive("bool"), ti.Primitive("float"), ti.Var(0))), ti.Var(1)
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(bool, float, T₀)¹"


def test_tuple_get():
    testee = ir.FunCall(
        fun=ir.SymRef(id="tuple_get"),
        args=[
            ir.IntLiteral(value=1),
            ir.FunCall(
                fun=ir.SymRef(id="make_tuple"),
                args=[ir.BoolLiteral(value=True), ir.FloatLiteral(value=42.0)],
            ),
        ],
    )
    expected = ti.Val(ti.Value(), ti.Primitive("float"), ti.Var(0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "float⁰"


def test_tuple_get_in_lambda():
    testee = ir.Lambda(
        params=[ir.Sym(id="x")],
        expr=ir.FunCall(
            fun=ir.SymRef(id="tuple_get"), args=[ir.IntLiteral(value=1), ir.SymRef(id="x")]
        ),
    )
    expected = ti.Fun(
        ti.Tuple((ti.Val(ti.Var(0), ti.PartialTupleVar(2, ((1, ti.Var(1)),)), ti.Var(3)),)),
        ti.Val(ti.Var(0), ti.Var(1), ti.Var(3)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(ItOrVal₀[(…, T₁, …)₂³]) → ItOrVal₀[T₁³]"


def test_reduce():
    reduction_f = ir.Lambda(
        params=[ir.Sym(id="acc"), ir.Sym(id="x"), ir.Sym(id="y")],
        expr=ir.FunCall(
            fun=ir.SymRef(id="plus"),
            args=[
                ir.SymRef(id="acc"),
                ir.FunCall(
                    fun=ir.SymRef(id="multiplies"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")]
                ),
            ],
        ),
    )
    testee = ir.FunCall(fun=ir.SymRef(id="reduce"), args=[reduction_f, ir.IntLiteral(value=0)])
    expected = ti.Fun(
        ti.Tuple(
            (
                ti.Val(ti.Iterator(), ti.Primitive("int"), ti.Var(0)),
                ti.Val(ti.Iterator(), ti.Primitive("int"), ti.Var(0)),
            )
        ),
        ti.Val(ti.Value(), ti.Primitive("int"), ti.Var(0)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(It[int⁰], It[int⁰]) → int⁰"


def test_scan():
    scan_f = ir.Lambda(
        params=[ir.Sym(id="acc"), ir.Sym(id="x"), ir.Sym(id="y")],
        expr=ir.FunCall(
            fun=ir.SymRef(id="plus"),
            args=[
                ir.SymRef(id="acc"),
                ir.FunCall(
                    fun=ir.SymRef(id="multiplies"),
                    args=[
                        ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x")]),
                        ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="y")]),
                    ],
                ),
            ],
        ),
    )
    testee = ir.FunCall(
        fun=ir.SymRef(id="scan"),
        args=[scan_f, ir.BoolLiteral(value=True), ir.IntLiteral(value=0)],
    )
    expected = ti.Fun(
        ti.Tuple(
            (
                ti.Val(ti.Iterator(), ti.Primitive("int"), ti.Column()),
                ti.Val(ti.Iterator(), ti.Primitive("int"), ti.Column()),
            )
        ),
        ti.Val(ti.Value(), ti.Primitive("int"), ti.Column()),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(It[intᶜ], It[intᶜ]) → intᶜ"


def test_shift():
    testee = ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.SymRef(id="i"), ir.IntLiteral(value=1)])
    expected = ti.Fun(
        ti.Tuple((ti.Val(ti.Iterator(), ti.Var(0), ti.Var(1)),)),
        ti.Val(ti.Iterator(), ti.Var(0), ti.Var(1)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(It[T₀¹]) → It[T₀¹]"


def test_function_definition():
    testee = ir.FunctionDefinition(id="f", params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
    expected = ti.FunDef("f", ti.Fun(ti.Tuple((ti.Var(0),)), ti.Var(0)))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "f :: (T₀) → T₀"


CARTESIAN_DOMAIN = ir.FunCall(
    fun=ir.SymRef(id="domain"),
    args=[
        ir.FunCall(
            fun=ir.SymRef(id="named_range"),
            args=[ir.AxisLiteral(value="IDim"), ir.IntLiteral(value=0), ir.SymRef(id="i")],
        ),
        ir.FunCall(
            fun=ir.SymRef(id="named_range"),
            args=[ir.AxisLiteral(value="JDim"), ir.IntLiteral(value=0), ir.SymRef(id="j")],
        ),
        ir.FunCall(
            fun=ir.SymRef(id="named_range"),
            args=[ir.AxisLiteral(value="KDim"), ir.IntLiteral(value=0), ir.SymRef(id="k")],
        ),
    ],
)


def test_stencil_closure():
    testee = ir.StencilClosure(
        domain=CARTESIAN_DOMAIN,
        stencil=ir.SymRef(id="deref"),
        output=ir.SymRef(id="out"),
        inputs=[ir.SymRef(id="inp")],
    )
    expected = ti.Closure(
        ti.Val(ti.Iterator(), ti.Var(0), ti.Column()),
        ti.Tuple((ti.Val(ti.Iterator(), ti.Var(0), ti.Column()),)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(It[T₀ᶜ]) ⇒ It[T₀ᶜ]"


def test_fencil_definition():
    testee = ir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[
            ir.Sym(id="i"),
            ir.Sym(id="j"),
            ir.Sym(id="k"),
            ir.Sym(id="a"),
            ir.Sym(id="b"),
            ir.Sym(id="c"),
            ir.Sym(id="d"),
        ],
        closures=[
            ir.StencilClosure(
                domain=CARTESIAN_DOMAIN,
                stencil=ir.SymRef(id="deref"),
                output=ir.SymRef(id="b"),
                inputs=[ir.SymRef(id="a")],
            ),
            ir.StencilClosure(
                domain=CARTESIAN_DOMAIN,
                stencil=ir.SymRef(id="deref"),
                output=ir.SymRef(id="d"),
                inputs=[ir.SymRef(id="c")],
            ),
        ],
    )
    expected = ti.Fencil(
        "f",
        ti.Tuple(()),
        ti.Tuple(
            (
                ti.Val(ti.Value(), ti.Primitive("int"), ti.Scalar()),
                ti.Val(ti.Value(), ti.Primitive("int"), ti.Scalar()),
                ti.Val(ti.Value(), ti.Primitive("int"), ti.Scalar()),
                ti.Val(ti.Iterator(), ti.Var(0), ti.Column()),
                ti.Val(ti.Iterator(), ti.Var(0), ti.Column()),
                ti.Val(ti.Iterator(), ti.Var(1), ti.Column()),
                ti.Val(ti.Iterator(), ti.Var(1), ti.Column()),
            )
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "{f(intˢ, intˢ, intˢ, It[T₀ᶜ], It[T₀ᶜ], It[T₁ᶜ], It[T₁ᶜ])}"


def test_fencil_definition_with_function_definitions():
    fundefs = [
        ir.FunctionDefinition(id="f", params=[ir.Sym(id="x")], expr=ir.SymRef(id="x")),
        ir.FunctionDefinition(
            id="g",
            params=[ir.Sym(id="x")],
            expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x")]),
        ),
    ]
    testee = ir.FencilDefinition(
        id="foo",
        function_definitions=fundefs,
        params=[
            ir.Sym(id="i"),
            ir.Sym(id="j"),
            ir.Sym(id="k"),
            ir.Sym(id="a"),
            ir.Sym(id="b"),
            ir.Sym(id="c"),
            ir.Sym(id="d"),
            ir.Sym(id="x"),
            ir.Sym(id="y"),
        ],
        closures=[
            ir.StencilClosure(
                domain=CARTESIAN_DOMAIN,
                stencil=ir.SymRef(id="g"),
                output=ir.SymRef(id="b"),
                inputs=[ir.SymRef(id="a")],
            ),
            ir.StencilClosure(
                domain=CARTESIAN_DOMAIN,
                stencil=ir.SymRef(id="deref"),
                output=ir.SymRef(id="d"),
                inputs=[ir.SymRef(id="c")],
            ),
            ir.StencilClosure(
                domain=CARTESIAN_DOMAIN,
                stencil=ir.Lambda(
                    params=[ir.Sym(id="y")],
                    expr=ir.FunCall(
                        fun=ir.SymRef(id="g"),
                        args=[ir.FunCall(fun=ir.SymRef(id="f"), args=[ir.SymRef(id="y")])],
                    ),
                ),
                output=ir.SymRef(id="y"),
                inputs=[ir.SymRef(id="x")],
            ),
        ],
    )
    expected = ti.Fencil(
        "foo",
        ti.Tuple(
            (
                ti.FunDef("f", ti.Fun(ti.Tuple((ti.Var(0),)), ti.Var(0))),
                ti.FunDef(
                    "g",
                    ti.Fun(
                        ti.Tuple((ti.Val(ti.Iterator(), ti.Var(1), ti.Var(2)),)),
                        ti.Val(ti.Value(), ti.Var(1), ti.Var(2)),
                    ),
                ),
            )
        ),
        ti.Tuple(
            (
                ti.Val(ti.Value(), ti.Primitive("int"), ti.Scalar()),
                ti.Val(ti.Value(), ti.Primitive("int"), ti.Scalar()),
                ti.Val(ti.Value(), ti.Primitive("int"), ti.Scalar()),
                ti.Val(ti.Iterator(), ti.Var(3), ti.Column()),
                ti.Val(ti.Iterator(), ti.Var(3), ti.Column()),
                ti.Val(ti.Iterator(), ti.Var(4), ti.Column()),
                ti.Val(ti.Iterator(), ti.Var(4), ti.Column()),
                # TODO: proper let-polymorphism
                ti.Val(ti.Iterator(), ti.Var(3), ti.Column()),
                ti.Val(ti.Iterator(), ti.Var(3), ti.Column()),
            )
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert (
        ti.pretty_str(inferred)
        == "{f :: (T₀) → T₀, g :: (It[T₁²]) → T₁², foo(intˢ, intˢ, intˢ, It[T₃ᶜ], It[T₃ᶜ], It[T₄ᶜ], It[T₄ᶜ], It[T₃ᶜ], It[T₃ᶜ])}"
    )
