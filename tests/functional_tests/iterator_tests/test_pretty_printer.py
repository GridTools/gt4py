from functional.iterator import ir
from functional.iterator.pretty_printer import PrettyPrinter, pformat


def test_hmerge():
    a = ["This is", "block ‘a’. "]
    b = ["This is", "block ‘b’. "]
    c = ["This is", "block ‘c’. "]
    expected = [
        "This is",
        "block ‘a’. This is",
        "           block ‘b’. This is",
        "                      block ‘c’. ",
    ]
    actual = PrettyPrinter()._hmerge(a, b, c)
    assert actual == expected


def test_vmerge():
    a = ["This is", "block ‘a’."]
    b = ["This is", "block ‘b’."]
    c = ["This is", "block ‘c’."]
    expected = [
        "This is",
        "block ‘a’.",
        "This is",
        "block ‘b’.",
        "This is",
        "block ‘c’.",
    ]
    actual = PrettyPrinter()._vmerge(a, b, c)
    assert actual == expected


def test_indent():
    a = ["This is", "block ‘a’."]
    expected = ["  This is", "  block ‘a’."]
    actual = PrettyPrinter()._indent(a)
    assert actual == expected


def test_cost():
    assert PrettyPrinter()._cost(["This is a single line."]) < PrettyPrinter()._cost(
        ["These are", "multiple", "short", "lines."]
    )
    assert PrettyPrinter()._cost(["This is a short line."]) < PrettyPrinter()._cost(
        [
            "This is a very long line; longer than the maximum allowed line length. "
            "So it should get a penalty for its length."
        ]
    )
    assert PrettyPrinter()._cost(
        ["Equal length!", "Equal length!", "Equal length!"]
    ) < PrettyPrinter()._cost(["Unequal length.", "Short…", "Looooooooooooooooooong…"])


def test_optimum():
    assert PrettyPrinter()._optimum(
        ["This is a single line."], ["These are", "multiple", "short", "lines."]
    ) == ["This is a single line."]


def test_prec_parens():
    a = ["This is", "block ‘a’."]
    assert PrettyPrinter()._prec_parens(a, 42, 42) == a
    assert PrettyPrinter()._prec_parens(a, 42, 0) == ["(This is", " block ‘a’.)"]


def test_hinterleave():
    blocks = [["a", "a"], ["b"], ["c"]]
    expected = [["a", "a,"], ["b,"], ["c"]]
    actual = list(PrettyPrinter()._hinterleave(blocks, ","))
    assert actual == expected


def test_hinterleave_indented():
    blocks = [["a", "a"], ["b"], ["c"]]
    expected = [["  a", "  a,"], ["  b,"], ["  c"]]
    actual = list(PrettyPrinter()._hinterleave(blocks, ",", indent=True))
    assert actual == expected


def test_lambda():
    testee = ir.Lambda(params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
    expected = "λ(x) → x"
    actual = pformat(testee)
    assert actual == expected


def test_arithmetic():
    testee = ir.FunCall(
        fun=ir.SymRef(id="divides"),
        args=[
            ir.FunCall(
                fun=ir.SymRef(id="multiplies"),
                args=[
                    ir.FunCall(
                        fun=ir.SymRef(id="plus"),
                        args=[ir.IntLiteral(value=1), ir.IntLiteral(value=2)],
                    ),
                    ir.IntLiteral(value=3),
                ],
            ),
            ir.IntLiteral(value=4),
        ],
    )
    expected = "(1 + 2) × 3 / 4"
    actual = pformat(testee)
    assert actual == expected


def test_deref():
    testee = ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x")])
    expected = "·x"
    actual = pformat(testee)
    assert actual == expected


def test_lift():
    testee = ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="x")])
    expected = "↑x"
    actual = pformat(testee)
    assert actual == expected


def test_bool_arithmetic():
    testee = ir.FunCall(
        fun=ir.SymRef(id="not_"),
        args=[
            ir.FunCall(
                fun=ir.SymRef(id="or_"),
                args=[
                    ir.FunCall(fun=ir.SymRef(id="not_"), args=[ir.SymRef(id="a")]),
                    ir.FunCall(
                        fun=ir.SymRef(id="and_"),
                        args=[
                            ir.SymRef(id="b"),
                            ir.FunCall(
                                fun=ir.SymRef(id="or_"), args=[ir.SymRef(id="c"), ir.SymRef(id="d")]
                            ),
                        ],
                    ),
                ],
            )
        ],
    )
    expected = "¬(¬a ∨ b ∧ (c ∨ d))"
    actual = pformat(testee)
    assert actual == expected


def test_shift():
    testee = ir.FunCall(
        fun=ir.SymRef(id="shift"),
        args=[ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1)],
    )
    expected = "⟪I, 1⟫"
    actual = pformat(testee)
    assert actual == expected


def test_tuple_get():
    testee = ir.FunCall(
        fun=ir.SymRef(id="tuple_get"), args=[ir.IntLiteral(value=42), ir.SymRef(id="x")]
    )
    expected = "x[42]"
    actual = pformat(testee)
    assert actual == expected


def test_make_tuple():
    testee = ir.FunCall(fun=ir.SymRef(id="make_tuple"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")])
    expected = "{x, y}"
    actual = pformat(testee)
    assert actual == expected


def test_named_range():
    testee = ir.FunCall(
        fun=ir.SymRef(id="named_range"),
        args=[ir.AxisLiteral(value="IDim"), ir.SymRef(id="x"), ir.SymRef(id="y")],
    )
    expected = "IDim: [x, y)"
    actual = pformat(testee)
    assert actual == expected


def test_domain():
    testee = ir.FunCall(
        fun=ir.SymRef(id="domain"),
        args=[ir.SymRef(id="x"), ir.SymRef(id="y")],
    )
    expected = "{ x × y }"
    actual = pformat(testee)
    assert actual == expected


def test_if_short():
    testee = ir.FunCall(
        fun=ir.SymRef(id="if_"),
        args=[ir.SymRef(id="x"), ir.SymRef(id="y"), ir.SymRef(id="z")],
    )
    expected = "if x then y else z"
    actual = pformat(testee)
    assert actual == expected


def test_if_long():
    testee = ir.FunCall(
        fun=ir.SymRef(id="if_"),
        args=[
            ir.SymRef(
                id="very_loooooooooooooooooooong_condition_to_force_a_line_break_and_test_alignment_of_branches"
            ),
            ir.SymRef(id="y"),
            ir.SymRef(id="z"),
        ],
    )
    expected = "if   very_loooooooooooooooooooong_condition_to_force_a_line_break_and_test_alignment_of_branches\nthen y\nelse z"
    actual = pformat(testee)
    assert actual == expected


def test_fun_call():
    testee = ir.FunCall(
        fun=ir.SymRef(id="f"),
        args=[ir.SymRef(id="x")],
    )
    expected = "f(x)"
    actual = pformat(testee)
    assert actual == expected


def test_lambda_call():
    testee = ir.FunCall(
        fun=ir.Lambda(params=[ir.Sym(id="x")], expr=ir.SymRef(id="x")),
        args=[ir.SymRef(id="x")],
    )
    expected = "(λ(x) → x)(x)"
    actual = pformat(testee)
    assert actual == expected


def test_function_definition():
    testee = ir.FunctionDefinition(id="f", params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
    expected = "f = λ(x) → x"
    actual = pformat(testee)
    assert actual == expected


def test_stencil_closure():
    testee = ir.StencilClosure(
        domain=ir.SymRef(id="d"),
        stencil=ir.SymRef(id="deref"),
        output=ir.SymRef(id="y"),
        inputs=[ir.SymRef(id="x")],
    )
    expected = "y ← (deref)(x) @ d"
    actual = pformat(testee)
    assert actual == expected


def test_fencil_definition():
    testee = ir.FencilDefinition(
        id="f",
        function_definitions=[
            ir.FunctionDefinition(id="g", params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
        ],
        params=[ir.Sym(id="d"), ir.Sym(id="x"), ir.Sym(id="y")],
        closures=[
            ir.StencilClosure(
                domain=ir.SymRef(id="d"),
                stencil=ir.SymRef(id="deref"),
                output=ir.SymRef(id="y"),
                inputs=[ir.SymRef(id="x")],
            )
        ],
    )
    actual = pformat(testee)
    expected = "f(d, x, y) {\n  g = λ(x) → x\n  y ← (deref)(x) @ d\n}"
    assert actual == expected
