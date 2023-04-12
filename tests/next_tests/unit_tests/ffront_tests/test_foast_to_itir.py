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

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gt4py.next.common import DimensionKind, Field
from gt4py.next.ffront import itir_makers as im, type_specifications as ts_ffront
from gt4py.next.ffront.fbuiltins import (
    Dimension,
    FieldOffset,
    float32,
    float64,
    int32,
    int64,
    neighbor_sum,
)
from gt4py.next.ffront.foast_to_itir import FieldOperatorLowering
from gt4py.next.ffront.func_to_foast import FieldOperatorParser
from gt4py.next.type_system import type_specifications as ts, type_translation


IDim = Dimension("IDim")
Edge = Dimension("Edge")
Vertex = Dimension("Vertex")
Cell = Dimension("Cell")
V2EDim = Dimension("V2E", DimensionKind.LOCAL)
V2E = FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
TDim = Dimension("TDim")  # Meaningless dimension, used for tests.


def debug_itir(tree):
    """Compare tree snippets while debugging."""
    from devtools import debug

    from gt4py.eve.codegen import format_python_source
    from gt4py.next.program_processors import EmbeddedDSL

    debug(format_python_source(EmbeddedDSL.apply(tree)))


def test_copy():
    def copy_field(inp: Field[[TDim], float64]):
        return inp

    parsed = FieldOperatorParser.apply_to_function(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.id == "copy_field"
    assert lowered.expr == im.ref("inp")


def test_scalar_arg():
    def scalar_arg(bar: Field[[IDim], int64], alpha: int64) -> Field[[IDim], int64]:
        return alpha * bar

    parsed = FieldOperatorParser.apply_to_function(scalar_arg)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("multiplies")(
        "alpha", "bar"
    )  # no difference to non-scalar arg

    assert lowered.expr == reference


def test_multicopy():
    def multicopy(inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]):
        return inp1, inp2

    parsed = FieldOperatorParser.apply_to_function(multicopy)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("make_tuple")("inp1", "inp2")

    assert lowered.expr == reference


def test_arithmetic():
    def arithmetic(inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]):
        return inp1 + inp2

    parsed = FieldOperatorParser.apply_to_function(arithmetic)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("plus")("inp1", "inp2")

    assert lowered.expr == reference


def test_shift():
    Ioff = FieldOffset("Ioff", source=IDim, target=(IDim,))

    def shift_by_one(inp: Field[[IDim], float64]):
        return inp(Ioff[1])

    parsed = FieldOperatorParser.apply_to_function(shift_by_one)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.lift_(im.lambda__("it")(im.deref_(im.shift_("Ioff", 1)("it"))))("inp")

    assert lowered.expr == reference


def test_negative_shift():
    Ioff = FieldOffset("Ioff", source=IDim, target=(IDim,))

    def shift_by_one(inp: Field[[IDim], float64]):
        return inp(Ioff[-1])

    parsed = FieldOperatorParser.apply_to_function(shift_by_one)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.lift_(im.lambda__("it")(im.deref_(im.shift_("Ioff", -1)("it"))))("inp")

    assert lowered.expr == reference


def test_temp_assignment():
    def copy_field(inp: Field[[TDim], float64]):
        tmp = inp
        inp = tmp
        tmp2 = inp
        return tmp2

    parsed = FieldOperatorParser.apply_to_function(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.let("tmp__0", "inp")(
        im.let("inp__0", "tmp__0")(im.let("tmp2__0", "inp__0")("tmp2__0"))
    )

    assert lowered.expr == reference


def test_unary_ops():
    def unary(inp: Field[[TDim], float64]):
        tmp = +inp
        tmp = -tmp
        return tmp

    parsed = FieldOperatorParser.apply_to_function(unary)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.let(
        "tmp__0",
        im.promote_to_lifted_stencil("plus")(
            im.promote_to_const_iterator(im.literal_("0", "float64")), "inp"
        ),
    )(
        im.let(
            "tmp__1",
            im.promote_to_lifted_stencil("minus")(
                im.promote_to_const_iterator(im.literal_("0", "float64")), "tmp__0"
            ),
        )("tmp__1")
    )

    assert lowered.expr == reference


def test_unpacking():
    """Unpacking assigns should get separated."""

    def unpacking(
        inp1: Field[[TDim], float64], inp2: Field[[TDim], float64]
    ) -> Field[[TDim], float64]:
        tmp1, tmp2 = inp1, inp2  # noqa
        return tmp1

    parsed = FieldOperatorParser.apply_to_function(unpacking)
    lowered = FieldOperatorLowering.apply(parsed)

    tuple_expr = im.promote_to_lifted_stencil("make_tuple")("inp1", "inp2")
    tuple_access_0 = im.promote_to_lifted_stencil(lambda x: im.tuple_get_(0, x))("__tuple_tmp_0")
    tuple_access_1 = im.promote_to_lifted_stencil(lambda x: im.tuple_get_(1, x))("__tuple_tmp_0")

    reference = im.let("__tuple_tmp_0", tuple_expr)(
        im.let("tmp1__0", tuple_access_0)(im.let("tmp2__0", tuple_access_1)("tmp1__0"))
    )

    assert lowered.expr == reference


def test_annotated_assignment():
    pytest.skip("Annotated assignments are not properly supported at the moment.")

    def copy_field(inp: Field[[TDim], float64]):
        tmp: Field[[TDim], float64] = inp
        return tmp

    parsed = FieldOperatorParser.apply_to_function(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.let("tmp__0", "inp")("tmp__0")

    assert lowered.expr == reference


def test_call():
    # create something that appears to the lowering like a field operator.
    #  we could also create an actual field operator, but we want to avoid
    #  using such heavy constructs for testing the lowering.
    field_type = type_translation.from_type_hint(Field[[TDim], float64])
    identity = SimpleNamespace(
        __gt_type__=lambda: ts_ffront.FieldOperatorType(
            definition=ts.FunctionType(args=[field_type], kwargs={}, returns=field_type)
        )
    )

    def call(inp: Field[[TDim], float64]) -> Field[[TDim], float64]:
        return identity(inp)

    parsed = FieldOperatorParser.apply_to_function(call)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.lift_(im.lambda__("__arg0")(im.call_("identity")("__arg0")))("inp")

    assert lowered.expr == reference


def test_temp_tuple():
    """Returning a temp tuple should work."""

    def temp_tuple(a: Field[[TDim], float64], b: Field[[TDim], int64]):
        tmp = a, b
        return tmp

    parsed = FieldOperatorParser.apply_to_function(temp_tuple)
    lowered = FieldOperatorLowering.apply(parsed)

    tuple_expr = im.promote_to_lifted_stencil("make_tuple")("a", "b")
    reference = im.let("tmp__0", tuple_expr)("tmp__0")

    assert lowered.expr == reference


def test_unary_not():
    def unary_not(cond: Field[[TDim], "bool"]):
        return not cond

    parsed = FieldOperatorParser.apply_to_function(unary_not)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("not_")("cond")

    assert lowered.expr == reference


def test_binary_plus():
    def plus(a: Field[[TDim], float64], b: Field[[TDim], float64]):
        return a + b

    parsed = FieldOperatorParser.apply_to_function(plus)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("plus")("a", "b")

    assert lowered.expr == reference


def test_add_scalar_literal_to_field():
    def scalar_plus_field(a: Field[[IDim], float64]) -> Field[[IDim], float64]:
        return 2.0 + a

    parsed = FieldOperatorParser.apply_to_function(scalar_plus_field)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("plus")(
        im.promote_to_const_iterator(im.literal_("2.0", "float64")), "a"
    )

    assert lowered.expr == reference


def test_add_scalar_literals():
    def scalar_plus_scalar(a: Field[[IDim], "int32"]) -> Field[[IDim], "int32"]:
        tmp = int32(1) + int32("1")
        return a + tmp

    parsed = FieldOperatorParser.apply_to_function(scalar_plus_scalar)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.let(
        "tmp__0",
        im.promote_to_lifted_stencil("plus")(
            im.promote_to_const_iterator(im.literal_("1", "int32")),
            im.promote_to_const_iterator(im.literal_("1", "int32")),
        ),
    )(im.promote_to_lifted_stencil("plus")("a", "tmp__0"))

    assert lowered.expr == reference


def test_binary_mult():
    def mult(a: Field[[TDim], float64], b: Field[[TDim], float64]):
        return a * b

    parsed = FieldOperatorParser.apply_to_function(mult)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("multiplies")("a", "b")

    assert lowered.expr == reference


def test_binary_minus():
    def minus(a: Field[[TDim], float64], b: Field[[TDim], float64]):
        return a - b

    parsed = FieldOperatorParser.apply_to_function(minus)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("minus")("a", "b")

    assert lowered.expr == reference


def test_binary_div():
    def division(a: Field[[TDim], float64], b: Field[[TDim], float64]):
        return a / b

    parsed = FieldOperatorParser.apply_to_function(division)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("divides")("a", "b")

    assert lowered.expr == reference


def test_binary_and():
    def bit_and(a: Field[[TDim], "bool"], b: Field[[TDim], "bool"]):
        return a & b

    parsed = FieldOperatorParser.apply_to_function(bit_and)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("and_")("a", "b")

    assert lowered.expr == reference


def test_scalar_and():
    def scalar_and(a: Field[[IDim], "bool"]) -> Field[[IDim], "bool"]:
        return a & False

    parsed = FieldOperatorParser.apply_to_function(scalar_and)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("and_")(
        "a", im.promote_to_const_iterator(im.literal_("False", "bool"))
    )

    assert lowered.expr == reference


def test_binary_or():
    def bit_or(a: Field[[TDim], "bool"], b: Field[[TDim], "bool"]):
        return a | b

    parsed = FieldOperatorParser.apply_to_function(bit_or)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("or_")("a", "b")

    assert lowered.expr == reference


def test_compare_scalars():
    def comp_scalars() -> bool:
        return 3 > 4

    parsed = FieldOperatorParser.apply_to_function(comp_scalars)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("greater")(
        im.promote_to_const_iterator(im.literal_("3", "int64")),
        im.promote_to_const_iterator(im.literal_("4", "int64")),
    )

    assert lowered.expr == reference


def test_compare_gt():
    def comp_gt(a: Field[[TDim], float64], b: Field[[TDim], float64]):
        return a > b

    parsed = FieldOperatorParser.apply_to_function(comp_gt)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("greater")("a", "b")

    assert lowered.expr == reference


def test_compare_lt():
    def comp_lt(a: Field[[TDim], float64], b: Field[[TDim], float64]):
        return a < b

    parsed = FieldOperatorParser.apply_to_function(comp_lt)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("less")("a", "b")

    assert lowered.expr == reference


def test_compare_eq():
    def comp_eq(a: Field[[TDim], "int64"], b: Field[[TDim], "int64"]):
        return a == b

    parsed = FieldOperatorParser.apply_to_function(comp_eq)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("eq")("a", "b")

    assert lowered.expr == reference


def test_compare_chain():
    def compare_chain(
        a: Field[[IDim], float64], b: Field[[IDim], float64], c: Field[[IDim], float64]
    ) -> Field[[IDim], bool]:
        return a > b > c

    parsed = FieldOperatorParser.apply_to_function(compare_chain)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("and_")(
        im.promote_to_lifted_stencil("greater")("a", "b"),
        im.promote_to_lifted_stencil("greater")("b", "c"),
    )

    assert lowered.expr == reference


def test_reduction_lowering_simple():
    def reduction(edge_f: Field[[Edge], float64]):
        return neighbor_sum(edge_f(V2E), axis=V2EDim)

    parsed = FieldOperatorParser.apply_to_function(reduction)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil(
        im.call_(
            im.call_("reduce")(
                "plus",
                im.deref_(im.promote_to_const_iterator(im.literal_(value="0", typename="float64"))),
            ),
        )
    )(
        im.lifted_neighbors("V2E", "edge_f"),
    )

    assert lowered.expr == reference


def test_reduction_lowering_expr():
    def reduction(e1: Field[[Edge], float64], e2: Field[[Vertex, V2EDim], float64]):
        e1_nbh = e1(V2E)
        return neighbor_sum(1.1 * (e1_nbh + e2), axis=V2EDim)

    parsed = FieldOperatorParser.apply_to_function(reduction)
    lowered = FieldOperatorLowering.apply(parsed)

    mapped = im.promote_to_lifted_stencil(im.map__("multiplies"))(
        im.promote_to_lifted_stencil("make_const_list")(
            im.promote_to_const_iterator(im.literal_("1.1", "float64"))
        ),
        im.promote_to_lifted_stencil(im.map__("plus"))("e1_nbh__0", "e2"),
    )

    reference = im.let("e1_nbh__0", im.lifted_neighbors("V2E", "e1"))(
        im.promote_to_lifted_stencil(
            im.call_(
                im.call_("reduce")(
                    "plus",
                    im.deref_(
                        im.promote_to_const_iterator(im.literal_(value="0", typename="float64"))
                    ),
                ),
            )
        )(
            mapped,
        )
    )

    assert lowered.expr == reference


def test_builtin_int_constructors():
    def int_constrs() -> (
        tuple[
            int,
            int,
            int32,
            int64,
            int,
            int32,
            int64,
        ]
    ):
        return 1, int(1), int32(1), int64(1), int("1"), int32("1"), int64("1")

    parsed = FieldOperatorParser.apply_to_function(int_constrs)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("make_tuple")(
        im.promote_to_const_iterator(im.literal_("1", "int64")),
        im.promote_to_const_iterator(im.literal_("1", "int64")),
        im.promote_to_const_iterator(im.literal_("1", "int32")),
        im.promote_to_const_iterator(im.literal_("1", "int64")),
        im.promote_to_const_iterator(im.literal_("1", "int64")),
        im.promote_to_const_iterator(im.literal_("1", "int32")),
        im.promote_to_const_iterator(im.literal_("1", "int64")),
    )

    assert lowered.expr == reference


def test_builtin_float_constructors():
    def float_constrs() -> (
        tuple[
            float,
            float,
            float32,
            float64,
            float,
            float32,
            float64,
        ]
    ):
        return (
            0.1,
            float(0.1),
            float32(0.1),
            float64(0.1),
            float(".1"),
            float32(".1"),
            float64(".1"),
        )

    parsed = FieldOperatorParser.apply_to_function(float_constrs)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("make_tuple")(
        im.promote_to_const_iterator(im.literal_("0.1", "float64")),
        im.promote_to_const_iterator(im.literal_("0.1", "float64")),
        im.promote_to_const_iterator(im.literal_("0.1", "float32")),
        im.promote_to_const_iterator(im.literal_("0.1", "float64")),
        im.promote_to_const_iterator(im.literal_(".1", "float64")),
        im.promote_to_const_iterator(im.literal_(".1", "float32")),
        im.promote_to_const_iterator(im.literal_(".1", "float64")),
    )

    assert lowered.expr == reference


def test_builtin_bool_constructors():
    def bool_constrs() -> tuple[bool, bool, bool, bool, bool, bool, bool, bool]:
        return True, False, bool(True), bool(False), bool(0), bool(5), bool("True"), bool("False")

    parsed = FieldOperatorParser.apply_to_function(bool_constrs)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("make_tuple")(
        im.promote_to_const_iterator(im.literal_(str(True), "bool")),
        im.promote_to_const_iterator(im.literal_(str(False), "bool")),
        im.promote_to_const_iterator(im.literal_(str(True), "bool")),
        im.promote_to_const_iterator(im.literal_(str(False), "bool")),
        im.promote_to_const_iterator(im.literal_(str(bool(0)), "bool")),
        im.promote_to_const_iterator(im.literal_(str(bool(5)), "bool")),
        im.promote_to_const_iterator(im.literal_(str(bool("True")), "bool")),
        im.promote_to_const_iterator(im.literal_(str(bool("False")), "bool")),
    )

    assert lowered.expr == reference
