# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from gt4py.next.iterator import ir, ir_makers as im
from gt4py.next.iterator.transforms.cse import CommonSubexpressionElimination as CSE


def test_trivial():
    common = ir.FunCall(fun=ir.SymRef(id="plus"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")])
    testee = ir.FunCall(fun=ir.SymRef(id="plus"), args=[common, common])
    expected = ir.FunCall(
        fun=ir.Lambda(
            params=[ir.Sym(id="_cs_1")],
            expr=ir.FunCall(
                fun=ir.SymRef(id="plus"), args=[ir.SymRef(id="_cs_1"), ir.SymRef(id="_cs_1")]
            ),
        ),
        args=[common],
    )
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_capture():
    common = ir.FunCall(fun=ir.SymRef(id="plus"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")])
    testee = ir.FunCall(fun=ir.Lambda(params=[ir.Sym(id="x")], expr=common), args=[common])
    expected = testee
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_no_capture():
    common = im.plus("x", "y")
    testee = im.call(im.lambda_("z")(im.plus("x", "y")))(im.plus("x", "y"))
    expected = im.let("_cs_1", common)("_cs_1")
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_nested_capture():
    def common_expr():
        return im.plus("x", "y")

    # (λ(x, y) → x + y)(x + y, x + y)
    testee = im.call(im.lambda_("x", "y")(common_expr()))(common_expr(), common_expr())
    # (λ(_cs_1) → _cs_1 + _cs_1)(x + y)
    expected = im.let("_cs_1", common_expr())(im.plus("_cs_1", "_cs_1"))
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_nested_capture_scoped():
    def common_expr():
        return im.plus("x", "x")

    # λ(x) → (λ(y) → y + (x + x + (x + x)))(z)
    testee = im.lambda_("x")(
        im.call(im.lambda_("y")(im.plus("y", im.plus(common_expr(), common_expr()))))("z")
    )
    # λ(x) → (λ(_cs_1) → (λ(y) → y + (_cs_1 + _cs_1))(z))(x + x)
    expected = im.lambda_("x")(
        im.call(
            im.lambda_("_cs_1")(
                im.call(im.lambda_("y")(im.plus("y", im.plus("_cs_1", "_cs_1"))))("z")
            )
        )(common_expr())
    )
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_redef():
    def common_expr():
        return im.lambda_("a")(im.plus("a", 1))

    # (λ(f1) → (λ(f2) → f1(2) + f2(3))(λ(a) → a + 1))(λ(a) → a + 1)
    testee = im.let("f1", common_expr())(
        im.let("f2", common_expr())(im.plus(im.call("f1")(2), im.call("f2")(3)))
    )
    # (λ(_cs_1) → _cs_1(2) + _cs_1(3))(λ(a) → a + 1)
    expected = im.let("_cs_1", common_expr())(im.plus(im.call("_cs_1")(2), im.call("_cs_1")(3)))
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_redef_same_arg():
    def common_expr():
        return im.lambda_("a")(im.plus("a", 1))

    # (λ(f1) → (λ(f2) → f1(2) + f2(2))(λ(a) → a + 1))(λ(a) → a + 1)
    testee = im.let("f1", common_expr())(
        im.let("f2", common_expr())(im.plus(im.call("f1")(2), im.call("f2")(2)))
    )
    # (λ(_cs_1) → (λ(_cs_2) → _cs_2 + _cs_2)(_cs_1(2)))(λ(a) → a + 1)
    expected = im.let("_cs_1", common_expr())(
        im.let("_cs_2", im.call("_cs_1")(2))(im.plus("_cs_2", "_cs_2"))
    )
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_redef_same_arg_scope():
    def common_expr():
        return im.lambda_("a")(im.plus("a", im.plus(1, 1)))

    # (λ(f1) → (λ(f2) → f1(2) + f2(2))(λ(a) → a + (1 + 1)))(λ(a) → a + (1 + 1)) + (1 + 1 + (1 + 1))
    testee = im.plus(
        im.let("f1", common_expr())(
            im.let("f2", common_expr())(im.plus(im.call("f1")(2), im.call("f2")(2)))
        ),
        im.plus(im.plus(1, 1), im.plus(1, 1)),
    )
    # (λ(_cs_3) → (λ(_cs_1) → (λ(_cs_4) → _cs_4 + _cs_4 + (_cs_3 + _cs_3))(_cs_1(2)))(
    #   λ(a) → a + _cs_3))(1 + 1)
    expected = im.let("_cs_3", im.plus(1, 1))(
        im.let("_cs_1", im.lambda_("a")(im.plus("a", "_cs_3")))(
            im.let("_cs_4", im.call("_cs_1")(2))(
                im.plus(im.plus("_cs_4", "_cs_4"), im.plus("_cs_3", "_cs_3"))
            )
        )
    )
    actual = CSE().visit(testee)
    assert actual == expected


def test_if_can_deref_no_extraction():
    # Test that a subexpression only occurring in one branch of an `if_` is not moved outside the
    # if statement. A case using `can_deref` is used here as it is common.

    # if can_deref(⟪Iₒ, 1ₒ⟫(it)) then ·⟪Iₒ, 1ₒ⟫(it) + ·⟪Iₒ, 1ₒ⟫(it) else 1
    testee = im.if_(
        im.call("can_deref")(im.shift("I", 1)("it")),
        im.plus(im.deref(im.shift("I", 1)("it")), im.deref(im.shift("I", 1)("it"))),
        # use something more involved where a subexpression can still be eliminated
        im.literal("1", "int32"),
    )
    # (λ(_cs_1) → if can_deref(_cs_1) then (λ(_cs_2) → _cs_2 + _cs_2)(·_cs_1) else 1)(⟪Iₒ, 1ₒ⟫(it))
    expected = im.let("_cs_1", im.shift("I", 1)("it"))(
        im.if_(
            im.call("can_deref")("_cs_1"),
            im.let("_cs_2", im.deref("_cs_1"))(im.plus("_cs_2", "_cs_2")),
            im.literal("1", "int32"),
        )
    )

    actual = CSE().visit(testee)
    assert actual == expected


def test_if_can_deref_eligible_extraction():
    # Test that a subexpression only occurring in both branches of an `if_` is moved outside the
    # if statement. A case using `can_deref` is used here as it is common.

    # if can_deref(⟪Iₒ, 1ₒ⟫(it)) then ·⟪Iₒ, 1ₒ⟫(it) else ·⟪Iₒ, 1ₒ⟫(it) + ·⟪Iₒ, 1ₒ⟫(it)
    testee = im.if_(
        im.call("can_deref")(im.shift("I", 1)("it")),
        im.deref(im.shift("I", 1)("it")),
        im.plus(im.deref(im.shift("I", 1)("it")), im.deref(im.shift("I", 1)("it"))),
    )
    # (λ(_cs_3) → (λ(_cs_1) → if can_deref(_cs_3) then _cs_1 else _cs_1 + _cs_1)(·_cs_3))(⟪Iₒ, 1ₒ⟫(it))
    expected = im.let("_cs_3", im.shift("I", 1)("it"))(
        im.let("_cs_1", im.deref("_cs_3"))(
            im.if_(im.call("can_deref")("_cs_3"), "_cs_1", im.plus("_cs_1", "_cs_1"))
        )
    )

    actual = CSE().visit(testee)
    assert actual == expected


def test_if_eligible_extraction():
    # Test that a subexpression only occurring in the condition of an `if_` is moved outside the
    # if statement.

    # if ((a ∧ b) ∧ (a ∧ b)) then c else d
    testee = im.if_(
        im.and_(im.and_("a", "b"), im.and_("a", "b")),
        "c",
        "d",
    )
    # (λ(_cs_1) → if _cs_1 ∧ _cs_1 then c else d)(a ∧ b)
    expected = im.let("_cs_1", im.and_("a", "b"))(im.if_(im.and_("_cs_1", "_cs_1"), "c", "d"))

    actual = CSE().visit(testee)
    assert actual == expected
