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

from gt4py.next.iterator import ir, makers as im
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
    common = im.plus_("x", "y")
    testee = im.call_(im.lambda__("z")(im.plus_("x", "y")))(im.plus_("x", "y"))
    expected = im.let("_cs_1", common)("_cs_1")
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_nested_capture():
    def common_expr():
        return im.plus_("x", "y")

    # (λ(x, y) → x + y)(x + y, x + y)
    testee = im.call_(im.lambda__("x", "y")(common_expr()))(common_expr(), common_expr())
    # (λ(_cs_1) → _cs_1 + _cs_1)(x + y)
    expected = im.let("_cs_1", common_expr())(im.plus_("_cs_1", "_cs_1"))
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_nested_capture_scoped():
    def common_expr():
        return im.plus_("x", "x")

    # λ(x) → (λ(y) → y + (x + x + (x + x)))(z)
    testee = im.lambda__("x")(
        im.call_(im.lambda__("y")(im.plus_("y", im.plus_(common_expr(), common_expr()))))("z")
    )
    # λ(x) → (λ(_cs_1) → (λ(y) → y + (_cs_1 + _cs_1))(z))(x + x)
    expected = im.lambda__("x")(
        im.call_(
            im.lambda__("_cs_1")(
                im.call_(im.lambda__("y")(im.plus_("y", im.plus_("_cs_1", "_cs_1"))))("z")
            )
        )(common_expr())
    )
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_redef():
    def common_expr():
        return im.lambda__("a")(im.plus_("a", 1))

    # (λ(f1) → (λ(f2) → f1(2) + f2(3))(λ(a) → a + 1))(λ(a) → a + 1)
    testee = im.let("f1", common_expr())(
        im.let("f2", common_expr())(im.plus_(im.call_("f1")(2), im.call_("f2")(3)))
    )
    # (λ(_cs_1) → _cs_1(2) + _cs_1(3))(λ(a) → a + 1)
    expected = im.let("_cs_1", common_expr())(im.plus_(im.call_("_cs_1")(2), im.call_("_cs_1")(3)))
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_redef_same_arg():
    def common_expr():
        return im.lambda__("a")(im.plus_("a", 1))

    # (λ(f1) → (λ(f2) → f1(2) + f2(2))(λ(a) → a + 1))(λ(a) → a + 1)
    testee = im.let("f1", common_expr())(
        im.let("f2", common_expr())(im.plus_(im.call_("f1")(2), im.call_("f2")(2)))
    )
    # (λ(_cs_1) → (λ(_cs_2) → _cs_2 + _cs_2)(_cs_1(2)))(λ(a) → a + 1)
    expected = im.let("_cs_1", common_expr())(
        im.let("_cs_2", im.call_("_cs_1")(2))(im.plus_("_cs_2", "_cs_2"))
    )
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_redef_same_arg_scope():
    def common_expr():
        return im.lambda__("a")(im.plus_("a", im.plus_(1, 1)))

    # (λ(f1) → (λ(f2) → f1(2) + f2(2))(λ(a) → a + 1))(λ(a) → a + 1)
    testee = im.plus_(
        im.let("f1", common_expr())(
            im.let("f2", common_expr())(im.plus_(im.call_("f1")(2), im.call_("f2")(2)))
        ),
        im.plus_(im.plus_(1, 1), im.plus_(1, 1)),
    )
    # (λ(_cs_1) → (λ(_cs_2) → _cs_2 + _cs_2)(_cs_1(2)))(λ(a) → a + 1)
    expected = im.let("_cs_3", im.plus_(1, 1))(
        im.let("_cs_1", im.lambda__("a")(im.plus_("a", "_cs_3")))(
            im.let("_cs_4", im.call_("_cs_1")(2))(
                im.plus_(im.plus_("_cs_4", "_cs_4"), im.plus_("_cs_3", "_cs_3"))
            )
        )
    )
    actual = CSE().visit(testee)
    assert actual == expected
