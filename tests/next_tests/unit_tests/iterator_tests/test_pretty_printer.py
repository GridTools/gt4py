# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.pretty_printer import PrettyPrinter, pformat
from gt4py.next.type_system import type_specifications as ts


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
    expected = ["This is", "block ‘a’.", "This is", "block ‘b’.", "This is", "block ‘c’."]
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


def test_offset_literal():
    testee = ir.OffsetLiteral(value="I")
    expected = "Iₒ"
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
                        args=[
                            im.literal("1", "int64"),
                            im.literal("2", "int64"),
                        ],
                    ),
                    im.literal("3", "int64"),
                ],
            ),
            im.literal("4", "int64"),
        ],
    )
    expected = "(1 + 2) × 3 / 4"
    actual = pformat(testee)
    assert actual == expected


def test_associativity():
    testee = ir.FunCall(
        fun=ir.SymRef(id="plus"),
        args=[
            ir.FunCall(
                fun=ir.SymRef(id="plus"),
                args=[im.literal("1", "int64"), im.literal("2", "int64")],
            ),
            ir.FunCall(
                fun=ir.SymRef(id="plus"),
                args=[im.literal("3", "int64"), im.literal("4", "int64")],
            ),
        ],
    )
    expected = "1 + 2 + (3 + 4)"
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


def test_as_fieldop():
    testee = ir.FunCall(fun=ir.SymRef(id="as_fieldop"), args=[ir.SymRef(id="x")])
    expected = "⇑x"
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
        fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1)]
    )
    expected = "⟪Iₒ, 1ₒ⟫"
    actual = pformat(testee)
    assert actual == expected


def test_tuple_get():
    testee = ir.FunCall(
        fun=ir.SymRef(id="tuple_get"),
        args=[im.literal("42", ir.INTEGER_INDEX_BUILTIN), ir.SymRef(id="x")],
    )
    expected = "x[42]"
    actual = pformat(testee)
    assert actual == expected


def test_make_tuple():
    testee = ir.FunCall(fun=ir.SymRef(id="make_tuple"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")])
    expected = "{x, y}"
    actual = pformat(testee)
    assert actual == expected


def test_axis_literal_horizontal():
    testee = ir.AxisLiteral(value="I", kind=ir.DimensionKind.HORIZONTAL)
    expected = "Iₕ"
    actual = pformat(testee)
    assert actual == expected


def test_axis_literal_vertical():
    testee = ir.AxisLiteral(value="I", kind=ir.DimensionKind.VERTICAL)
    expected = "Iᵥ"
    actual = pformat(testee)
    assert actual == expected


def test_named_range_horizontal():
    testee = ir.FunCall(
        fun=ir.SymRef(id="named_range"),
        args=[ir.AxisLiteral(value="IDim"), ir.SymRef(id="x"), ir.SymRef(id="y")],
    )
    expected = "IDimₕ: [x, y["
    actual = pformat(testee)
    assert actual == expected


def test_cartesian_domain():
    testee = ir.FunCall(
        fun=ir.SymRef(id="cartesian_domain"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")]
    )
    expected = "c⟨ x, y ⟩"
    actual = pformat(testee)
    assert actual == expected


def test_unstructured_domain():
    testee = ir.FunCall(
        fun=ir.SymRef(id="unstructured_domain"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")]
    )
    expected = "u⟨ x, y ⟩"
    actual = pformat(testee)
    assert actual == expected


def test_if_short():
    testee = ir.FunCall(
        fun=ir.SymRef(id="if_"), args=[ir.SymRef(id="x"), ir.SymRef(id="y"), ir.SymRef(id="z")]
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
    testee = ir.FunCall(fun=ir.SymRef(id="f"), args=[ir.SymRef(id="x")])
    expected = "f(x)"
    actual = pformat(testee)
    assert actual == expected


def test_lambda_call():
    testee = ir.FunCall(
        fun=ir.Lambda(params=[ir.Sym(id="x")], expr=ir.SymRef(id="x")), args=[ir.SymRef(id="x")]
    )
    expected = "(λ(x) → x)(x)"
    actual = pformat(testee)
    assert actual == expected


def test_function_definition():
    testee = ir.FunctionDefinition(id="f", params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
    expected = "f = λ(x) → x;"
    actual = pformat(testee)
    assert actual == expected


def test_temporary():
    testee = ir.Temporary(
        id="t", domain=ir.SymRef(id="domain"), dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    )
    expected = "t = temporary(domain=domain, dtype=float64);"
    actual = pformat(testee)
    assert actual == expected


def test_set_at():
    testee = ir.SetAt(
        expr=ir.SymRef(id="x"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
        target=ir.SymRef(id="y"),
    )
    expected = "y @ cartesian_domain() ← x;"
    actual = pformat(testee)
    assert actual == expected


def test_program():
    testee = ir.Program(
        id="f",
        function_definitions=[
            ir.FunctionDefinition(id="g", params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
        ],
        params=[ir.Sym(id="d"), ir.Sym(id="x"), ir.Sym(id="y")],
        declarations=[
            ir.Temporary(
                id="tmp",
                domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
                dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
            ),
        ],
        body=[
            ir.SetAt(
                expr=ir.SymRef(id="x"),
                domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
                target=ir.SymRef(id="y"),
            )
        ],
    )
    actual = pformat(testee)
    expected = "f(d, x, y) {\n  g = λ(x) → x;\n  tmp = temporary(domain=cartesian_domain(), dtype=float64);\n  y @ cartesian_domain() ← x;\n}"
    assert actual == expected
