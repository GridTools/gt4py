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
from functional.ffront import itir_makers as mi
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
    assert lowered.expr == mi.deref_("inp")


@pytest.mark.skip(reason="Constant-to-field promotion not enabled yet.")
@pytest.mark.skip(reason="Iterator IR does not allow arg-less lambdas yet.")
def test_constant():
    def constant():
        return 5

    # ast_passes
    parsed = FieldOperatorParser.apply_to_function(constant)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == mi.deref_(mi.lift_(mi.lambda__()(5)))


def test_multicopy():
    def multicopy(inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]):
        return inp1, inp2

    parsed = FieldOperatorParser.apply_to_function(multicopy)
    lowered = FieldOperatorLowering.apply(parsed)

    # TODO(ricoh): reevaluate after tuple of fields is allowed in iterator model
    reference = mi.deref_(mi.make_tuple_("inp1", "inp2"))

    assert lowered.expr == reference


def test_arithmetic():
    def arithmetic(inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]):
        return inp1 + inp2

    # ast_passes
    parsed = FieldOperatorParser.apply_to_function(arithmetic)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(
        mi.lift_(mi.lambda__("inp1", "inp2")(mi.plus_(mi.deref_("inp1"), mi.deref_("inp2"))))(
            "inp1", "inp2"
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

    reference = mi.deref_(mi.shift_("Ioff", 1)("inp"))

    assert lowered.expr == reference


def test_temp_assignment():
    def copy_field(inp: Field[..., "float64"]):
        tmp = inp
        inp = tmp
        tmp2 = inp
        return tmp2

    parsed = FieldOperatorParser.apply_to_function(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(
        mi.let("tmp__0", "inp")(mi.let("inp__0", "tmp__0")(mi.let("tmp2__0", "inp__0")("tmp2__0")))
    )

    assert lowered.expr == reference


def test_unary_ops():
    def unary(inp: Field[..., "float64"]):
        tmp = +inp
        tmp = -tmp
        return tmp

    parsed = FieldOperatorParser.apply_to_function(unary)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(
        mi.let("tmp__0", mi.lift_(mi.lambda__("inp")(mi.plus_(0, mi.deref_("inp"))))("inp"),)(
            mi.let(
                "tmp__1",
                mi.lift_(mi.lambda__("tmp__0")(mi.minus_(0, mi.deref_("tmp__0"))))("tmp__0"),
            )("tmp__1")
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

    reference = mi.deref_(
        mi.let("tmp1__0", mi.tuple_get_(0, mi.make_tuple_("inp1", "inp2")))(
            mi.let("tmp2__0", mi.tuple_get_(1, mi.make_tuple_("inp1", "inp2")))("tmp1__0")
        )
    )

    assert lowered.expr == reference


def test_annotated_assignment():
    def copy_field(inp: Field[..., "float64"]):
        tmp: Field[..., "float64"] = inp
        return tmp

    parsed = FieldOperatorParser.apply_to_function(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(mi.let("tmp__0", "inp")("tmp__0"))

    assert lowered.expr == reference


def test_call():
    def identity(x: Field[..., "float64"]) -> Field[..., "float64"]:
        return x

    def call(inp: Field[..., "float64"]) -> Field[..., "float64"]:
        return identity(inp)

    parsed = FieldOperatorParser.apply_to_function(call)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(mi.lift_(mi.lambda__("inp")(mi.call_("identity")("inp")))("inp"))

    assert lowered.expr == reference


def test_temp_tuple():
    """Returning a temp tuple should work."""

    def temp_tuple(a: Field[..., float64], b: Field[..., int64]):
        tmp = a, b
        return tmp

    parsed = FieldOperatorParser.apply_to_function(temp_tuple)
    lowered = FieldOperatorLowering.apply(parsed)

    # TODO(ricoh): reevaluate after tuple of fields is allowed in iterator model
    reference = mi.deref_(mi.let("tmp__0", mi.make_tuple_("a", "b"))("tmp__0"))

    assert lowered.expr == reference


def test_unary_not():
    def unary_not(cond: Field[..., "bool"]):
        return not cond

    parsed = FieldOperatorParser.apply_to_function(unary_not)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(mi.lift_(mi.lambda__("cond")(mi.not__(mi.deref_("cond"))))("cond"))

    assert lowered.expr == reference


def test_binary_plus():
    def plus(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a + b

    parsed = FieldOperatorParser.apply_to_function(plus)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(
        mi.lift_(mi.lambda__("a", "b")(mi.plus_(mi.deref_("a"), mi.deref_("b"))))("a", "b")
    )

    assert lowered.expr == reference


def test_binary_mult():
    def mult(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a * b

    parsed = FieldOperatorParser.apply_to_function(mult)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(
        mi.lift_(mi.lambda__("a", "b")(mi.multiplies_(mi.deref_("a"), mi.deref_("b"))))("a", "b")
    )

    assert lowered.expr == reference


def test_binary_minus():
    def minus(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a - b

    parsed = FieldOperatorParser.apply_to_function(minus)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(
        mi.lift_(mi.lambda__("a", "b")(mi.minus_(mi.deref_("a"), mi.deref_("b"))))("a", "b")
    )

    assert lowered.expr == reference


def test_binary_div():
    def division(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a / b

    parsed = FieldOperatorParser.apply_to_function(division)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(
        mi.lift_(mi.lambda__("a", "b")(mi.divides_(mi.deref_("a"), mi.deref_("b"))))("a", "b")
    )

    assert lowered.expr == reference


def test_binary_and():
    def bit_and(a: Field[..., "bool"], b: Field[..., "bool"]):
        return a & b

    parsed = FieldOperatorParser.apply_to_function(bit_and)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(
        mi.lift_(mi.lambda__("a", "b")(mi.and__(mi.deref_("a"), mi.deref_("b"))))("a", "b")
    )

    assert lowered.expr == reference


def test_binary_or():
    def bit_or(a: Field[..., "bool"], b: Field[..., "bool"]):
        return a | b

    parsed = FieldOperatorParser.apply_to_function(bit_or)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(
        mi.lift_(mi.lambda__("a", "b")(mi.or__(mi.deref_("a"), mi.deref_("b"))))("a", "b")
    )

    assert lowered.expr == reference


def test_compare_gt():
    def comp_gt(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a > b

    parsed = FieldOperatorParser.apply_to_function(comp_gt)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(
        mi.lift_(mi.lambda__("a", "b")(mi.greater_(mi.deref_("a"), mi.deref_("b"))))("a", "b")
    )

    assert lowered.expr == reference


def test_compare_lt():
    def comp_lt(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a < b

    parsed = FieldOperatorParser.apply_to_function(comp_lt)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(
        mi.lift_(mi.lambda__("a", "b")(mi.less_(mi.deref_("a"), mi.deref_("b"))))("a", "b")
    )

    assert lowered.expr == reference


def test_compare_eq():
    def comp_eq(a: Field[..., "int64"], b: Field[..., "int64"]):
        return a == b

    parsed = FieldOperatorParser.apply_to_function(comp_eq)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(
        mi.lift_(mi.lambda__("a", "b")(mi.eq_(mi.deref_("a"), mi.deref_("b"))))("a", "b")
    )

    assert lowered.expr == reference


def test_compare_chain():
    def compare_chain(a: Field[..., "float64"], b: Field[..., "float64"], c: Field[..., "float64"]):
        return a > b > c

    parsed = FieldOperatorParser.apply_to_function(compare_chain)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = mi.deref_(
        mi.lift_(
            mi.lambda__("a", "b", "c")(
                mi.greater_(
                    mi.deref_("a"),
                    mi.deref_(
                        mi.lift_(
                            mi.lambda__("b", "c")(mi.greater_(mi.deref_("b"), mi.deref_("c")))
                        )("b", "c")
                    ),
                )
            )
        )("a", "b", "c")
    )

    assert lowered.expr == reference
