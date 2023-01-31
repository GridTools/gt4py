# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.next.iterator import ir
from gt4py.next.iterator.pretty_parser import pparse


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
                        args=[ir.Literal(value="1", type="int"), ir.Literal(value="2", type="int")],
                    ),
                    ir.Literal(value="3", type="int"),
                ],
            ),
            ir.Literal(value="4", type="int"),
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
        fun=ir.SymRef(id="shift"),
        args=[ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1)],
    )
    actual = pparse(testee)
    assert actual == expected


def test_tuple_get():
    testee = "x[42]"
    expected = ir.FunCall(
        fun=ir.SymRef(id="tuple_get"), args=[ir.Literal(value="42", type="int"), ir.SymRef(id="x")]
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


def test_named_range():
    testee = "IDim: [x, y)"
    expected = ir.FunCall(
        fun=ir.SymRef(id="named_range"),
        args=[ir.AxisLiteral(value="IDim"), ir.SymRef(id="x"), ir.SymRef(id="y")],
    )
    actual = pparse(testee)
    assert actual == expected


def test_cartesian_domain():
    testee = "c⟨ x, y ⟩"
    expected = ir.FunCall(
        fun=ir.SymRef(id="cartesian_domain"),
        args=[ir.SymRef(id="x"), ir.SymRef(id="y")],
    )
    actual = pparse(testee)
    assert actual == expected


def test_unstructured_domain():
    testee = "u⟨ x, y ⟩"
    expected = ir.FunCall(
        fun=ir.SymRef(id="unstructured_domain"),
        args=[ir.SymRef(id="x"), ir.SymRef(id="y")],
    )
    actual = pparse(testee)
    assert actual == expected


def test_if():
    testee = "if x then y else z"
    expected = ir.FunCall(
        fun=ir.SymRef(id="if_"),
        args=[ir.SymRef(id="x"), ir.SymRef(id="y"), ir.SymRef(id="z")],
    )
    actual = pparse(testee)
    assert actual == expected


def test_fun_call():
    testee = "f(x)"
    expected = ir.FunCall(
        fun=ir.SymRef(id="f"),
        args=[ir.SymRef(id="x")],
    )
    actual = pparse(testee)
    assert actual == expected


def test_lambda_call():
    testee = "(λ(x) → x)(x)"
    expected = ir.FunCall(
        fun=ir.Lambda(params=[ir.Sym(id="x")], expr=ir.SymRef(id="x")),
        args=[ir.SymRef(id="x")],
    )
    actual = pparse(testee)
    assert actual == expected


def test_function_definition():
    testee = "f = λ(x) → x;"
    expected = ir.FunctionDefinition(id="f", params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
    actual = pparse(testee)
    assert actual == expected


def test_stencil_closure():
    testee = "y ← (deref)(x) @ cartesian_domain();"
    expected = ir.StencilClosure(
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
        stencil=ir.SymRef(id="deref"),
        output=ir.SymRef(id="y"),
        inputs=[ir.SymRef(id="x")],
    )
    actual = pparse(testee)
    assert actual == expected


def test_fencil_definition():
    testee = "f(d, x, y) {\n  g = λ(x) → x;\n  y ← (deref)(x) @ cartesian_domain();\n}"
    expected = ir.FencilDefinition(
        id="f",
        function_definitions=[
            ir.FunctionDefinition(id="g", params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
        ],
        params=[ir.Sym(id="d"), ir.Sym(id="x"), ir.Sym(id="y")],
        closures=[
            ir.StencilClosure(
                domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
                stencil=ir.SymRef(id="deref"),
                output=ir.SymRef(id="y"),
                inputs=[ir.SymRef(id="x")],
            )
        ],
    )
    actual = pparse(testee)
    assert actual == expected
