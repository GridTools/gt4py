# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator import ir, builtins
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.pretty_parser import pparse
from gt4py.next.type_system import type_specifications as ts


def test_symref():
    testee = "x"
    expected = ir.SymRef(id="x")
    actual = pparse(testee)
    assert actual == expected


def test_lambda():
    testee = "λ(x) → x"
    expected = ir.Lambda(params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
    actual = pparse(testee)
    assert actual == expected


def test_arithmetic():
    testee = "(1 + 2) × 3 / 4"
    expected = ir.FunCall(
        fun=ir.SymRef(id="divides"),
        args=[
            ir.FunCall(
                fun=ir.SymRef(id="multiplies"),
                args=[
                    ir.FunCall(
                        fun=ir.SymRef(id="plus"),
                        args=[
                            im.literal("1", "int32"),
                            im.literal("2", "int32"),
                        ],
                    ),
                    im.literal("3", "int32"),
                ],
            ),
            im.literal("4", "int32"),
        ],
    )
    actual = pparse(testee)
    assert actual == expected


def test_deref():
    testee = "·x"
    expected = ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x")])
    actual = pparse(testee)
    assert actual == expected


def test_lift():
    testee = "↑x"
    expected = ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="x")])
    actual = pparse(testee)
    assert actual == expected


def test_as_fieldop():
    testee = "⇑x"
    expected = ir.FunCall(fun=ir.SymRef(id="as_fieldop"), args=[ir.SymRef(id="x")])
    actual = pparse(testee)
    assert actual == expected


def test_bool_arithmetic():
    testee = "¬(¬a ∨ b ∧ (c ∨ d))"
    expected = ir.FunCall(
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
    actual = pparse(testee)
    assert actual == expected


def test_shift():
    testee = "⟪Iₒ, 1ₒ⟫"
    expected = ir.FunCall(
        fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1)]
    )
    actual = pparse(testee)
    assert actual == expected


def test_tuple_get():
    testee = "x[42]"
    expected = ir.FunCall(
        fun=ir.SymRef(id="tuple_get"),
        args=[im.literal("42", builtins.INTEGER_INDEX_BUILTIN), ir.SymRef(id="x")],
    )
    actual = pparse(testee)
    assert actual == expected


def test_make_tuple():
    testee = "{x, y}"
    expected = ir.FunCall(
        fun=ir.SymRef(id="make_tuple"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")]
    )
    actual = pparse(testee)
    assert actual == expected


def test_named_range_horizontal():
    testee = "IDimₕ: [x, y["
    expected = ir.FunCall(
        fun=ir.SymRef(id="named_range"),
        args=[ir.AxisLiteral(value="IDim"), ir.SymRef(id="x"), ir.SymRef(id="y")],
    )
    actual = pparse(testee)
    assert actual == expected


def test_named_range_vertical():
    testee = "IDimᵥ: [x, y["
    expected = ir.FunCall(
        fun=ir.SymRef(id="named_range"),
        args=[
            ir.AxisLiteral(value="IDim", kind=ir.DimensionKind.VERTICAL),
            ir.SymRef(id="x"),
            ir.SymRef(id="y"),
        ],
    )
    actual = pparse(testee)
    assert actual == expected


def test_cartesian_domain():
    testee = "c⟨ x, y ⟩"
    expected = ir.FunCall(
        fun=ir.SymRef(id="cartesian_domain"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")]
    )
    actual = pparse(testee)
    assert actual == expected


def test_unstructured_domain():
    testee = "u⟨ x, y ⟩"
    expected = ir.FunCall(
        fun=ir.SymRef(id="unstructured_domain"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")]
    )
    actual = pparse(testee)
    assert actual == expected


def test_if():
    testee = "if x then y else z"
    expected = ir.FunCall(
        fun=ir.SymRef(id="if_"), args=[ir.SymRef(id="x"), ir.SymRef(id="y"), ir.SymRef(id="z")]
    )
    actual = pparse(testee)
    assert actual == expected


def test_fun_call():
    testee = "f(x)"
    expected = ir.FunCall(fun=ir.SymRef(id="f"), args=[ir.SymRef(id="x")])
    actual = pparse(testee)
    assert actual == expected


def test_lambda_call():
    testee = "(λ(x) → x)(x)"
    expected = ir.FunCall(
        fun=ir.Lambda(params=[ir.Sym(id="x")], expr=ir.SymRef(id="x")), args=[ir.SymRef(id="x")]
    )
    actual = pparse(testee)
    assert actual == expected


def test_function_definition():
    testee = "f = λ(x) → x;"
    expected = ir.FunctionDefinition(id="f", params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
    actual = pparse(testee)
    assert actual == expected


def test_temporary():
    testee = "t = temporary(domain=domain, dtype=float64);"
    float64_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    expected = ir.Temporary(id="t", domain=ir.SymRef(id="domain"), dtype=float64_type)
    actual = pparse(testee)
    assert actual == expected


def test_set_at():
    testee = "y @ cartesian_domain() ← x;"
    expected = ir.SetAt(
        expr=ir.SymRef(id="x"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
        target=ir.SymRef(id="y"),
    )
    actual = pparse(testee)
    assert actual == expected


def test_if_stmt():
    testee = """if (cond) { 
      y @ cartesian_domain() ← x;
      if (cond) {
        y @ cartesian_domain() ← x;
      } else {
      }
    } else {
      y @ cartesian_domain() ← x;
    }"""
    stmt = ir.SetAt(
        expr=im.ref("x"),
        domain=im.domain("cartesian_domain", {}),
        target=im.ref("y"),
    )
    expected = ir.IfStmt(
        cond=im.ref("cond"),
        true_branch=[
            stmt,
            ir.IfStmt(
                cond=im.ref("cond"),
                true_branch=[stmt],
                false_branch=[],
            ),
        ],
        false_branch=[stmt],
    )
    actual = pparse(testee)
    assert actual == expected


def test_program():
    testee = "f(d, x, y) {\n  g = λ(x) → x;\n  tmp = temporary(domain=cartesian_domain(), dtype=float64);\n  y @ cartesian_domain() ← x;\n}"
    expected = ir.Program(
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
    actual = pparse(testee)
    assert actual == expected
