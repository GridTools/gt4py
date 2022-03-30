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
    assert ti.pretty_str(inferred) == "(It[T₀]¹) → T₀¹"


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
    testee = ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="deref")])
    expected = ti.Fun(
        ti.Tuple((ti.Val(ti.Iterator(), ti.Var(0), ti.Var(1)),)),
        ti.Val(ti.Iterator(), ti.Var(0), ti.Var(1)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(It[T₀]¹) → It[T₀]¹"


def test_lifted_call():
    testee = ir.FunCall(
        fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="deref")]),
        args=[ir.SymRef(id="x")],
    )
    expected = ti.Val(ti.Iterator(), ti.Var(0), ti.Var(1))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "It[T₀]¹"


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
    assert ti.pretty_str(inferred) == "(It[int]⁰, It[int]⁰) → int⁰"


def test_scan():
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
        fun=ir.SymRef(id="scan"),
        args=[reduction_f, ir.BoolLiteral(value=True), ir.IntLiteral(value=0)],
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
    assert ti.pretty_str(inferred) == "(It[int]ᶜ, It[int]ᶜ) → intᶜ"


def test_shift():
    testee = ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.SymRef(id="i"), ir.IntLiteral(value=1)])
    expected = ti.Fun(
        ti.Tuple((ti.Val(ti.Iterator(), ti.Var(0), ti.Var(1)),)),
        ti.Val(ti.Iterator(), ti.Var(0), ti.Var(1)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pretty_str(inferred) == "(It[T₀]¹) → It[T₀]¹"
