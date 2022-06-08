from functional.iterator import ir, type_inference as ti


def test_renamer():
    r = ti._Renamer()
    actual = [
        (
            ti._Box(value=ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1))),
            ti._Box(value=ti.TypeVar(idx=0)),
        )
    ]
    src = ti.TypeVar(idx=0)
    dst = ti.TypeVar(idx=1)
    for s, t in actual:
        r.register(s)
        r.register(t)
    r.register(src)
    r.register(dst)
    r.rename(src, dst)
    expected = [
        (
            ti._Box(value=ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=1), size=ti.TypeVar(idx=1))),
            ti._Box(value=ti.TypeVar(idx=1)),
        )
    ]
    assert actual == expected


def test_sym_ref():
    testee = ir.SymRef(id="x")
    expected = ti.TypeVar(idx=0)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "T₀"


def test_bool_literal():
    testee = ir.Literal(value="False", type="bool")
    expected = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="bool"), size=ti.TypeVar(idx=0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "bool⁰"


def test_int_literal():
    testee = ir.Literal(value="3", type="int")
    expected = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.TypeVar(idx=0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "int⁰"


def test_float_literal():
    testee = ir.Literal(value="3.0", type="float")
    expected = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="float"), size=ti.TypeVar(idx=0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "float⁰"


def test_deref():
    testee = ir.SymRef(id="deref")
    expected = ti.FunctionType(
        args=ti.Tuple(
            elems=(ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1)),)
        ),
        ret=ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[T₀¹]) → T₀¹"


def test_deref_call():
    testee = ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x")])
    expected = ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "T₀¹"


def test_lambda():
    testee = ir.Lambda(params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
    expected = ti.FunctionType(args=ti.Tuple(elems=(ti.TypeVar(idx=0),)), ret=ti.TypeVar(idx=0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(T₀) → T₀"


def test_plus():
    testee = ir.SymRef(id="plus")
    t = ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1))
    expected = ti.FunctionType(args=ti.Tuple(elems=(t, t)), ret=t)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(T₀¹, T₀¹) → T₀¹"


def test_eq():
    testee = ir.SymRef(id="eq")
    t = ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1))
    expected = ti.FunctionType(
        args=ti.Tuple(elems=(t, t)),
        ret=ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="bool"), size=ti.TypeVar(idx=1)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(T₀¹, T₀¹) → bool¹"


def test_if():
    testee = ir.SymRef(id="if_")
    c = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="bool"), size=ti.TypeVar(idx=0))
    t = ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=1), size=ti.TypeVar(idx=0))
    expected = ti.FunctionType(args=ti.Tuple(elems=(c, t, t)), ret=t)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(bool⁰, T₁⁰, T₁⁰) → T₁⁰"


def test_not():
    testee = ir.SymRef(id="not_")
    t = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="bool"), size=ti.TypeVar(idx=0))
    expected = ti.FunctionType(args=ti.Tuple(elems=(t,)), ret=t)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(bool⁰) → bool⁰"


def test_and():
    testee = ir.SymRef(id="and_")
    t = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="bool"), size=ti.TypeVar(idx=0))
    expected = ti.FunctionType(args=ti.Tuple(elems=(t, t)), ret=t)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(bool⁰, bool⁰) → bool⁰"


def test_lift():
    testee = ir.SymRef(id="lift")
    expected = ti.FunctionType(
        args=ti.Tuple(
            elems=(
                ti.FunctionType(
                    args=ti.ValTuple(
                        kind=ti.Iterator(), dtypes=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1)
                    ),
                    ret=ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=2), size=ti.TypeVar(idx=1)),
                ),
            )
        ),
        ret=ti.FunctionType(
            args=ti.ValTuple(kind=ti.Iterator(), dtypes=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1)),
            ret=ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=2), size=ti.TypeVar(idx=1)),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "((It[T¹], …)₀ → T₂¹) → (It[T¹], …)₀ → It[T₂¹]"


def test_lift_application():
    testee = ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="deref")])
    expected = ti.FunctionType(
        args=ti.Tuple(
            elems=(ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1)),)
        ),
        ret=ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[T₀¹]) → It[T₀¹]"


def test_lifted_call():
    testee = ir.FunCall(
        fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="deref")]),
        args=[ir.SymRef(id="x")],
    )
    expected = ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "It[T₀¹]"


def test_make_tuple():
    testee = ir.FunCall(
        fun=ir.SymRef(id="make_tuple"),
        args=[
            ir.Literal(value="True", type="bool"),
            ir.Literal(value="42.0", type="float"),
            ir.SymRef(id="x"),
        ],
    )
    expected = ti.Val(
        kind=ti.Value(),
        dtype=ti.Tuple(
            elems=(ti.Primitive(name="bool"), ti.Primitive(name="float"), ti.TypeVar(idx=0))
        ),
        size=ti.TypeVar(idx=1),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(bool, float, T₀)¹"


def test_tuple_get():
    testee = ir.FunCall(
        fun=ir.SymRef(id="tuple_get"),
        args=[
            ir.Literal(value="1", type="int"),
            ir.FunCall(
                fun=ir.SymRef(id="make_tuple"),
                args=[
                    ir.Literal(value="True", type="bool"),
                    ir.Literal(value="42.0", type="float"),
                ],
            ),
        ],
    )
    expected = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="float"), size=ti.TypeVar(idx=0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "float⁰"


def test_tuple_get_in_lambda():
    testee = ir.Lambda(
        params=[ir.Sym(id="x")],
        expr=ir.FunCall(
            fun=ir.SymRef(id="tuple_get"),
            args=[ir.Literal(value="1", type="int"), ir.SymRef(id="x")],
        ),
    )
    expected = ti.FunctionType(
        args=ti.Tuple(
            elems=(
                ti.Val(
                    kind=ti.TypeVar(idx=0),
                    dtype=ti.PartialTupleVar(
                        idx=2, elem_indices=(1,), elem_values=(ti.TypeVar(idx=1),)
                    ),
                    size=ti.TypeVar(idx=3),
                ),
            )
        ),
        ret=ti.Val(kind=ti.TypeVar(idx=0), dtype=ti.TypeVar(idx=1), size=ti.TypeVar(idx=3)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(ItOrVal₀[(_, T₁, …)₂³]) → ItOrVal₀[T₁³]"


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
    testee = ir.FunCall(
        fun=ir.SymRef(id="reduce"), args=[reduction_f, ir.Literal(value="0", type="int")]
    )
    expected = ti.FunctionType(
        args=ti.Tuple(
            elems=(
                ti.Val(kind=ti.Iterator(), dtype=ti.Primitive(name="int"), size=ti.TypeVar(idx=0)),
                ti.Val(kind=ti.Iterator(), dtype=ti.Primitive(name="int"), size=ti.TypeVar(idx=0)),
            )
        ),
        ret=ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.TypeVar(idx=0)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[int⁰], It[int⁰]) → int⁰"


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
        args=[scan_f, ir.Literal(value="True", type="bool"), ir.Literal(value="0", type="int")],
    )
    expected = ti.FunctionType(
        args=ti.Tuple(
            elems=(
                ti.Val(kind=ti.Iterator(), dtype=ti.Primitive(name="int"), size=ti.Column()),
                ti.Val(kind=ti.Iterator(), dtype=ti.Primitive(name="int"), size=ti.Column()),
            )
        ),
        ret=ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.Column()),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[intᶜ], It[intᶜ]) → intᶜ"


def test_shift():
    testee = ir.FunCall(
        fun=ir.SymRef(id="shift"), args=[ir.SymRef(id="i"), ir.Literal(value="1", type="int")]
    )
    expected = ti.FunctionType(
        args=ti.Tuple(
            elems=(ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1)),)
        ),
        ret=ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[T₀¹]) → It[T₀¹]"


def test_function_definition():
    testee = ir.FunctionDefinition(id="f", params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
    expected = ti.FunctionDefinitionType(
        name="f",
        fun=ti.FunctionType(args=ti.Tuple(elems=(ti.TypeVar(idx=0),)), ret=ti.TypeVar(idx=0)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "f :: (T₀) → T₀"


CARTESIAN_DOMAIN = ir.FunCall(
    fun=ir.SymRef(id="domain"),
    args=[
        ir.FunCall(
            fun=ir.SymRef(id="named_range"),
            args=[
                ir.AxisLiteral(value="IDim"),
                ir.Literal(value="0", type="int"),
                ir.SymRef(id="i"),
            ],
        ),
        ir.FunCall(
            fun=ir.SymRef(id="named_range"),
            args=[
                ir.AxisLiteral(value="JDim"),
                ir.Literal(value="0", type="int"),
                ir.SymRef(id="j"),
            ],
        ),
        ir.FunCall(
            fun=ir.SymRef(id="named_range"),
            args=[
                ir.AxisLiteral(value="KDim"),
                ir.Literal(value="0", type="int"),
                ir.SymRef(id="k"),
            ],
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
        output=ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=0), size=ti.Column()),
        inputs=ti.Tuple(
            elems=(ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=0), size=ti.Column()),)
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[T₀ᶜ]) ⇒ It[T₀ᶜ]"


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
    expected = ti.FencilDefinitionType(
        name="f",
        fundefs=(),
        params=(
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.Scalar()),
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.Scalar()),
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.Scalar()),
            ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=0), size=ti.Column()),
            ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=0), size=ti.Column()),
            ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=1), size=ti.Column()),
            ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=1), size=ti.Column()),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "{f(intˢ, intˢ, intˢ, It[T₀ᶜ], It[T₀ᶜ], It[T₁ᶜ], It[T₁ᶜ])}"


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
    expected = ti.FencilDefinitionType(
        name="foo",
        fundefs=(
            ti.FunctionDefinitionType(
                name="f",
                fun=ti.FunctionType(
                    args=ti.Tuple(elems=(ti.TypeVar(idx=0),)), ret=ti.TypeVar(idx=0)
                ),
            ),
            ti.FunctionDefinitionType(
                name="g",
                fun=ti.FunctionType(
                    args=ti.Tuple(
                        elems=(
                            ti.Val(
                                kind=ti.Iterator(), dtype=ti.TypeVar(idx=1), size=ti.TypeVar(idx=2)
                            ),
                        )
                    ),
                    ret=ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=1), size=ti.TypeVar(idx=2)),
                ),
            ),
        ),
        params=(
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.Scalar()),
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.Scalar()),
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.Scalar()),
            ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=3), size=ti.Column()),
            ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=3), size=ti.Column()),
            ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=4), size=ti.Column()),
            ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=4), size=ti.Column()),
            ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=5), size=ti.Column()),
            ti.Val(kind=ti.Iterator(), dtype=ti.TypeVar(idx=5), size=ti.Column()),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert (
        ti.pformat(inferred)
        == "{f :: (T₀) → T₀, g :: (It[T₁²]) → T₁², foo(intˢ, intˢ, intˢ, It[T₃ᶜ], It[T₃ᶜ], It[T₄ᶜ], It[T₄ᶜ], It[T₅ᶜ], It[T₅ᶜ])}"
    )


def test_pformat():
    vs = [ti.TypeVar(idx=i) for i in range(4)]
    assert ti.pformat(vs[0]) == "T₀"
    assert ti.pformat(ti.Tuple(elems=tuple(vs[:2]))) == "(T₀, T₁)"
    assert (
        ti.pformat(ti.PartialTupleVar(idx=0, elem_indices=(1, 3), elem_values=(vs[0], vs[1])))
        == "(_, T₀, _, T₁, …)₀"
    )
    assert ti.pformat(ti.PrefixTuple(prefix=vs[0], others=vs[1])) == "T₀:T₁"
    assert ti.pformat(ti.FunctionType(args=vs[0], ret=vs[1])) == "T₀ → T₁"
    assert ti.pformat(ti.Val(kind=vs[0], dtype=vs[1], size=vs[2])) == "ItOrVal₀[T₁²]"
    assert ti.pformat(ti.Val(kind=ti.Value(), dtype=vs[0], size=vs[1])) == "T₀¹"
    assert ti.pformat(ti.Val(kind=ti.Iterator(), dtype=vs[0], size=vs[1])) == "It[T₀¹]"
    assert ti.pformat(ti.Val(kind=ti.Value(), dtype=vs[0], size=ti.Scalar())) == "T₀ˢ"
    assert ti.pformat(ti.Val(kind=ti.Value(), dtype=vs[0], size=ti.Column())) == "T₀ᶜ"
    assert ti.pformat(ti.ValTuple(kind=vs[0], dtypes=vs[1], size=vs[2])) == "(ItOrVal₀[T²], …)₁"
    assert (
        ti.pformat(ti.UniformValTupleVar(idx=0, kind=vs[1], dtype=vs[2], size=vs[3]))
        == "(ItOrVal₁[T₂³] × n₀)"
    )
    assert ti.pformat(ti.Primitive(name="foo")) == "foo"
    assert ti.pformat(ti.Closure(output=vs[0], inputs=vs[1])) == "T₁ ⇒ T₀"
    assert (
        ti.pformat(ti.FunctionDefinitionType(name="f", fun=ti.FunctionType(args=vs[0], ret=vs[1])))
        == "f :: T₀ → T₁"
    )
    assert ti.pformat(ti.FencilDefinitionType(name="f", fundefs=(), params=())) == "{f()}"
