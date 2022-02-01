# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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
import pytest

from functional.common import Field
from functional.ffront import mockitir as mi
from functional.ffront.fbuiltins import float64, int64
from functional.ffront.foast_to_itir import FieldOperatorLowering
from functional.ffront.func_to_foast import FieldOperatorParser
from functional.iterator.runtime import CartesianAxis, offset


IDim = CartesianAxis("IDim")


def debug_itir(tree):
    import black
    from devtools import debug

    from functional.iterator.backends.roundtrip import EmbeddedDSL

    debug(black.format_str(EmbeddedDSL.apply(tree), mode=black.Mode()))


def test_copy():
    def copy_field(inp: Field[..., "float64"]):
        return inp

    # ast_passes
    parsed = FieldOperatorParser.apply_to_function(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.id == "copy_field"
    assert lowered.expr == mi.deref("inp")


@pytest.mark.skip(reason="Constant-to-field promotion not enabled yet.")
@pytest.mark.skip(reason="Iterator IR does not allow arg-less lambdas yet.")
def test_constant():
    def constant():
        return 5

    # ast_passes
    parsed = FieldOperatorParser.apply_to_function(constant)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == mi.deref(mi.lift(mi.lambda_()(5)))


def test_multicopy():
    def multicopy(inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]):
        return inp1, inp2

    parsed = FieldOperatorParser.apply_to_function(multicopy)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(
        mi.lift(mi.lambda_("inp1", "inp2")(mi.make_tuple(mi.deref("inp1"), mi.deref("inp2"))))(
            mi.ref("inp1"), mi.ref("inp2")
        )
    )

    assert lowered.expr == reference


def test_arithmetic():
    def arithmetic(inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]):
        return inp1 + inp2

    # ast_passes
    parsed = FieldOperatorParser.apply_to_function(arithmetic)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(
        mi.lift(mi.lambda_("inp1", "inp2")(mi.plus(mi.deref("inp1"), mi.deref("inp2"))))(
            mi.ref("inp1"), mi.ref("inp2")
        )
    )

    assert lowered.expr == reference


def test_shift():
    Ioff = offset("Ioff")

    def shift_by_one(inp: Field[[IDim], float64]):
        return inp(Ioff[1])

    # ast_passes
    parsed = FieldOperatorParser.apply_to_function(shift_by_one)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(mi.shift("Ioff", 1)("inp"))

    assert lowered.expr == reference


def test_temp_assignment():
    def copy_field(inp: Field[..., "float64"]):
        tmp = inp
        inp = tmp
        tmp2 = inp
        return tmp2

    parsed = FieldOperatorParser.apply_to_function(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(
        mi.liftlet("tmp__0", mi.ref("inp"))(
            mi.deref(
                mi.liftlet("inp__0", mi.ref("tmp__0"))(
                    mi.deref(mi.liftlet("tmp2__0", mi.ref("inp__0"))(mi.deref("tmp2__0")))
                )
            )
        )
    )

    assert lowered.expr == reference


def test_unary_ops():
    def unary(inp: Field[..., "float64"]):
        tmp = +inp
        tmp = -tmp
        return tmp

    parsed = FieldOperatorParser.apply_to_function(unary)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(
        mi.liftlet(
            "tmp__0",
            mi.lift(mi.lambda_("inp")(mi.plus(0, mi.deref("inp"))))(mi.ref("inp")),
        )(
            mi.deref(
                mi.liftlet(
                    "tmp__1",
                    mi.lift(mi.lambda_("tmp__0")(mi.minus(0, mi.deref("tmp__0"))))(
                        mi.ref("tmp__0")
                    ),
                )(mi.deref("tmp__1"))
            )
        )
    )

    assert lowered.expr == reference


def test_unpacking():
    """Unpacking assigns should get separated."""

    def unpacking(inp1: Field[..., "float64"], inp2: Field[..., "float64"]):
        tmp1, tmp2 = inp1, inp2  # noqa
        return tmp1

    parsed = FieldOperatorParser.apply_to_function(unpacking)
    lowered = FieldOperatorLowering.apply(parsed)

    tuple_expr = mi.lift(
        mi.lambda_("inp1", "inp2")(mi.make_tuple(mi.deref("inp1"), mi.deref("inp2")))
    )(mi.ref("inp1"), mi.ref("inp2"))

    tmp1_expr = mi.lift(mi.lambda_("inp1", "inp2")(mi.tuple_get(mi.deref(tuple_expr), 0)))(
        mi.ref("inp1"), mi.ref("inp2")
    )

    tmp2_expr = mi.lift(mi.lambda_("inp1", "inp2")(mi.tuple_get(mi.deref(tuple_expr), 1)))(
        mi.ref("inp1"), mi.ref("inp2")
    )

    reference = mi.deref(
        mi.liftlet("tmp1__0", tmp1_expr)(
            mi.deref(mi.liftlet("tmp2__0", tmp2_expr)(mi.deref("tmp1__0")))
        )
    )

    assert lowered.expr == reference


def test_annotated_assignment():
    def copy_field(inp: Field[..., "float64"]):
        tmp: Field[..., "float64"] = inp
        return tmp

    parsed = FieldOperatorParser.apply_to_function(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(mi.liftlet("tmp__0", "inp")(mi.deref("tmp__0")))

    assert lowered.expr == reference


def test_call():
    def identity(x: Field[..., "float64"]) -> Field[..., "float64"]:
        return x

    def call(inp: Field[..., "float64"]) -> Field[..., "float64"]:
        return identity(inp)

    parsed = FieldOperatorParser.apply_to_function(call)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(mi.lift(mi.lambda_("inp")(mi.call("identity")("inp")))(mi.ref("inp")))

    assert lowered.expr == reference


def test_temp_tuple():
    """Returning a temp tuple should work."""

    def temp_tuple(a: Field[..., float64], b: Field[..., int64]):
        tmp = a, b
        return tmp

    parsed = FieldOperatorParser.apply_to_function(temp_tuple)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(
        mi.liftlet(
            "tmp__0",
            mi.lift(mi.lambda_("a", "b")(mi.make_tuple(mi.deref("a"), mi.deref("b"))))(
                mi.ref("a"), mi.ref("b")
            ),
        )(mi.deref("tmp__0"))
    )

    assert lowered.expr == reference


def test_unary_not():
    def unary_not(cond: Field[..., "bool"]):
        return not cond

    parsed = FieldOperatorParser.apply_to_function(unary_not)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(mi.lift(mi.lambda_("cond")(mi.not_(mi.deref("cond"))))(mi.ref("cond")))

    assert lowered.expr == reference


def test_binary_plus():
    def plus(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a + b

    parsed = FieldOperatorParser.apply_to_function(plus)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(
        mi.lift(mi.lambda_("a", "b")(mi.plus(mi.deref("a"), mi.deref("b"))))(
            mi.ref("a"), mi.ref("b")
        )
    )

    assert lowered.expr == reference


def test_binary_mult():
    def mult(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a * b

    parsed = FieldOperatorParser.apply_to_function(mult)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(
        mi.lift(mi.lambda_("a", "b")(mi.multiplies(mi.deref("a"), mi.deref("b"))))(
            mi.ref("a"), mi.ref("b")
        )
    )

    assert lowered.expr == reference


def test_binary_minus():
    def minus(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a - b

    parsed = FieldOperatorParser.apply_to_function(minus)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(
        mi.lift(mi.lambda_("a", "b")(mi.minus(mi.deref("a"), mi.deref("b"))))(
            mi.ref("a"), mi.ref("b")
        )
    )

    assert lowered.expr == reference


def test_binary_div():
    def division(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a / b

    parsed = FieldOperatorParser.apply_to_function(division)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(
        mi.lift(mi.lambda_("a", "b")(mi.divides(mi.deref("a"), mi.deref("b"))))(
            mi.ref("a"), mi.ref("b")
        )
    )

    assert lowered.expr == reference


def test_binary_and():
    def bit_and(a: Field[..., "bool"], b: Field[..., "bool"]):
        return a & b

    parsed = FieldOperatorParser.apply_to_function(bit_and)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(
        mi.lift(mi.lambda_("a", "b")(mi.and_(mi.deref("a"), mi.deref("b"))))(
            mi.ref("a"), mi.ref("b")
        )
    )

    assert lowered.expr == reference


def test_binary_or():
    def bit_or(a: Field[..., "bool"], b: Field[..., "bool"]):
        return a | b

    parsed = FieldOperatorParser.apply_to_function(bit_or)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(
        mi.lift(mi.lambda_("a", "b")(mi.or_(mi.deref("a"), mi.deref("b"))))(
            mi.ref("a"), mi.ref("b")
        )
    )

    assert lowered.expr == reference


def test_compare_gt():
    def comp_gt(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a > b

    parsed = FieldOperatorParser.apply_to_function(comp_gt)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(
        mi.lift(mi.lambda_("a", "b")(mi.greater(mi.deref("a"), mi.deref("b"))))(
            mi.ref("a"), mi.ref("b")
        )
    )

    assert lowered.expr == reference


def test_compare_lt():
    def comp_lt(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a < b

    parsed = FieldOperatorParser.apply_to_function(comp_lt)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(
        mi.lift(mi.lambda_("a", "b")(mi.less(mi.deref("a"), mi.deref("b"))))(
            mi.ref("a"), mi.ref("b")
        )
    )

    assert lowered.expr == reference


def test_compare_eq():
    def comp_eq(a: Field[..., "int64"], b: Field[..., "int64"]):
        return a == b

    parsed = FieldOperatorParser.apply_to_function(comp_eq)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(
        mi.lift(mi.lambda_("a", "b")(mi.eq(mi.deref("a"), mi.deref("b"))))(mi.ref("a"), mi.ref("b"))
    )

    assert lowered.expr == reference


def test_compare_chain():
    def compare_chain(a: Field[..., "float64"], b: Field[..., "float64"], c: Field[..., "float64"]):
        return a > b > c

    parsed = FieldOperatorParser.apply_to_function(compare_chain)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref(
        mi.lift(
            mi.lambda_("a", "b", "c")(
                mi.greater(
                    mi.deref("a"),
                    mi.deref(
                        mi.lift(mi.lambda_("b", "c")(mi.greater(mi.deref("b"), mi.deref("c"))))(
                            mi.ref("b"), mi.ref("c")
                        )
                    ),
                )
            )
        )(mi.ref("a"), mi.ref("b"), mi.ref("c"))
    )

    assert lowered.expr == reference
