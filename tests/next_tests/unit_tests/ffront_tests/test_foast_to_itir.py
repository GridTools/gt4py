# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# TODO(tehrengruber): The style of the tests in this file is not optimal as a single change in the
#  lowering can (and often does) make all of them fail. Once we have embedded field view we want to
#  switch to executing the different cases here; once with a regular backend (i.e. including
#  parsing) and then with embedded field view (i.e. no parsing). If the results match the lowering
#  should be correct.

from __future__ import annotations

from types import SimpleNamespace

import pytest

import gt4py.next as gtx
from gt4py.next import float32, float64, int32, int64, neighbor_sum
from gt4py.next.ffront import type_specifications as ts_ffront
from gt4py.next.ffront.ast_passes import single_static_assign as ssa
from gt4py.next.ffront.foast_to_itir import FieldOperatorLowering
from gt4py.next.ffront.func_to_foast import FieldOperatorParser
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.type_system import type_specifications as ts, type_translation
from gt4py.next.iterator.type_system import type_specifications as it_ts


IDim = gtx.Dimension("IDim")
Edge = gtx.Dimension("Edge")
Vertex = gtx.Dimension("Vertex")
Cell = gtx.Dimension("Cell")
V2EDim = gtx.Dimension("V2E", gtx.DimensionKind.LOCAL)
V2E = gtx.FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
TDim = gtx.Dimension("TDim")  # Meaningless dimension, used for tests.


def debug_itir(tree):
    """Compare tree snippets while debugging."""
    from devtools import debug

    from gt4py.eve.codegen import format_python_source
    from gt4py.next.program_processors import EmbeddedDSL

    debug(format_python_source(EmbeddedDSL.apply(tree)))


def test_copy():
    def copy_field(inp: gtx.Field[[TDim], float64]):
        return inp

    parsed = FieldOperatorParser.apply_to_function(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.id == "copy_field"
    assert lowered.expr == im.ref("inp")


def test_scalar_arg():
    def scalar_arg(bar: gtx.Field[[IDim], int64], alpha: int64) -> gtx.Field[[IDim], int64]:
        return alpha * bar

    parsed = FieldOperatorParser.apply_to_function(scalar_arg)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("multiplies")(
        "alpha", "bar"
    )  # no difference to non-scalar arg

    assert lowered.expr == reference


def test_multicopy():
    def multicopy(inp1: gtx.Field[[IDim], float64], inp2: gtx.Field[[IDim], float64]):
        return inp1, inp2

    parsed = FieldOperatorParser.apply_to_function(multicopy)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.make_tuple("inp1", "inp2")

    assert lowered.expr == reference


def test_arithmetic():
    def arithmetic(inp1: gtx.Field[[IDim], float64], inp2: gtx.Field[[IDim], float64]):
        return inp1 + inp2

    parsed = FieldOperatorParser.apply_to_function(arithmetic)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("plus")("inp1", "inp2")

    assert lowered.expr == reference


def test_shift():
    Ioff = gtx.FieldOffset("Ioff", source=IDim, target=(IDim,))

    def shift_by_one(inp: gtx.Field[[IDim], float64]):
        return inp(Ioff[1])

    parsed = FieldOperatorParser.apply_to_function(shift_by_one)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.lift(im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))))("inp")

    assert lowered.expr == reference


def test_negative_shift():
    Ioff = gtx.FieldOffset("Ioff", source=IDim, target=(IDim,))

    def shift_by_one(inp: gtx.Field[[IDim], float64]):
        return inp(Ioff[-1])

    parsed = FieldOperatorParser.apply_to_function(shift_by_one)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.lift(im.lambda_("it")(im.deref(im.shift("Ioff", -1)("it"))))("inp")

    assert lowered.expr == reference


def test_temp_assignment():
    def copy_field(inp: gtx.Field[[TDim], float64]):
        tmp = inp
        inp = tmp
        tmp2 = inp
        return tmp2

    parsed = FieldOperatorParser.apply_to_function(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.let(ssa.unique_name("tmp", 0), "inp")(
        im.let(
            ssa.unique_name("inp", 0),
            ssa.unique_name("tmp", 0),
        )(
            im.let(
                ssa.unique_name("tmp2", 0),
                ssa.unique_name("inp", 0),
            )(ssa.unique_name("tmp2", 0))
        )
    )

    assert lowered.expr == reference


def test_unary_ops():
    def unary(inp: gtx.Field[[TDim], float64]):
        tmp = +inp
        tmp = -tmp
        return tmp

    parsed = FieldOperatorParser.apply_to_function(unary)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.let(
        ssa.unique_name("tmp", 0),
        im.promote_to_lifted_stencil("plus")(
            im.promote_to_const_iterator(im.literal("0", "float64")), "inp"
        ),
    )(
        im.let(
            ssa.unique_name("tmp", 1),
            im.promote_to_lifted_stencil("minus")(
                im.promote_to_const_iterator(im.literal("0", "float64")), ssa.unique_name("tmp", 0)
            ),
        )(ssa.unique_name("tmp", 1))
    )

    assert lowered.expr == reference


def test_unpacking():
    """Unpacking assigns should get separated."""

    def unpacking(
        inp1: gtx.Field[[TDim], float64], inp2: gtx.Field[[TDim], float64]
    ) -> gtx.Field[[TDim], float64]:
        tmp1, tmp2 = inp1, inp2  # noqa
        return tmp1

    parsed = FieldOperatorParser.apply_to_function(unpacking)
    lowered = FieldOperatorLowering.apply(parsed)

    tuple_expr = im.make_tuple("inp1", "inp2")
    tuple_access_0 = im.tuple_get(0, "__tuple_tmp_0")
    tuple_access_1 = im.tuple_get(1, "__tuple_tmp_0")

    reference = im.let("__tuple_tmp_0", tuple_expr)(
        im.let(
            ssa.unique_name("tmp1", 0),
            tuple_access_0,
        )(
            im.let(
                ssa.unique_name("tmp2", 0),
                tuple_access_1,
            )(ssa.unique_name("tmp1", 0))
        )
    )

    assert lowered.expr == reference


def test_annotated_assignment():
    pytest.xfail("Annotated assignments are not properly supported at the moment.")

    def copy_field(inp: gtx.Field[[TDim], float64]):
        tmp: gtx.Field[[TDim], float64] = inp
        return tmp

    parsed = FieldOperatorParser.apply_to_function(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.let(ssa.unique_name("tmp", 0), "inp")(ssa.unique_name("tmp", 0))

    assert lowered.expr == reference


def test_call():
    # create something that appears to the lowering like a field operator.
    #  we could also create an actual field operator, but we want to avoid
    #  using such heavy constructs for testing the lowering.
    field_type = type_translation.from_type_hint(gtx.Field[[TDim], float64])
    identity = SimpleNamespace(
        __gt_type__=lambda: ts_ffront.FieldOperatorType(
            definition=ts.FunctionType(
                pos_only_args=[field_type], pos_or_kw_args={}, kw_only_args={}, returns=field_type
            )
        )
    )

    def call(inp: gtx.Field[[TDim], float64]) -> gtx.Field[[TDim], float64]:
        return identity(inp)

    parsed = FieldOperatorParser.apply_to_function(call)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.call("identity")("inp")

    assert lowered.expr == reference


def test_temp_tuple():
    """Returning a temp tuple should work."""

    def temp_tuple(a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], int64]):
        tmp = a, b
        return tmp

    parsed = FieldOperatorParser.apply_to_function(temp_tuple)
    lowered = FieldOperatorLowering.apply(parsed)

    tuple_expr = im.make_tuple("a", "b")
    reference = im.let(ssa.unique_name("tmp", 0), tuple_expr)(ssa.unique_name("tmp", 0))

    assert lowered.expr == reference


def test_unary_not():
    def unary_not(cond: gtx.Field[[TDim], "bool"]):
        return not cond

    parsed = FieldOperatorParser.apply_to_function(unary_not)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("not_")("cond")

    assert lowered.expr == reference


def test_binary_plus():
    def plus(a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], float64]):
        return a + b

    parsed = FieldOperatorParser.apply_to_function(plus)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("plus")("a", "b")

    assert lowered.expr == reference


def test_add_scalar_literal_to_field():
    def scalar_plus_field(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        return 2.0 + a

    parsed = FieldOperatorParser.apply_to_function(scalar_plus_field)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("plus")(
        im.promote_to_const_iterator(im.literal("2.0", "float64")), "a"
    )

    assert lowered.expr == reference


def test_add_scalar_literals():
    def scalar_plus_scalar(a: gtx.Field[[IDim], "int32"]) -> gtx.Field[[IDim], "int32"]:
        tmp = int32(1) + int32("1")
        return a + tmp

    parsed = FieldOperatorParser.apply_to_function(scalar_plus_scalar)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.let(
        ssa.unique_name("tmp", 0),
        im.promote_to_lifted_stencil("plus")(
            im.promote_to_const_iterator(im.literal("1", "int32")),
            im.promote_to_const_iterator(im.literal("1", "int32")),
        ),
    )(im.promote_to_lifted_stencil("plus")("a", ssa.unique_name("tmp", 0)))

    assert lowered.expr == reference


def test_binary_mult():
    def mult(a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], float64]):
        return a * b

    parsed = FieldOperatorParser.apply_to_function(mult)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("multiplies")("a", "b")

    assert lowered.expr == reference


def test_binary_minus():
    def minus(a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], float64]):
        return a - b

    parsed = FieldOperatorParser.apply_to_function(minus)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("minus")("a", "b")

    assert lowered.expr == reference


def test_binary_div():
    def division(a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], float64]):
        return a / b

    parsed = FieldOperatorParser.apply_to_function(division)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("divides")("a", "b")

    assert lowered.expr == reference


def test_binary_and():
    def bit_and(a: gtx.Field[[TDim], "bool"], b: gtx.Field[[TDim], "bool"]):
        return a & b

    parsed = FieldOperatorParser.apply_to_function(bit_and)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("and_")("a", "b")

    assert lowered.expr == reference


def test_scalar_and():
    def scalar_and(a: gtx.Field[[IDim], "bool"]) -> gtx.Field[[IDim], "bool"]:
        return a & False

    parsed = FieldOperatorParser.apply_to_function(scalar_and)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("and_")(
        "a", im.promote_to_const_iterator(im.literal("False", "bool"))
    )

    assert lowered.expr == reference


def test_binary_or():
    def bit_or(a: gtx.Field[[TDim], "bool"], b: gtx.Field[[TDim], "bool"]):
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
        im.promote_to_const_iterator(im.literal("3", "int32")),
        im.promote_to_const_iterator(im.literal("4", "int32")),
    )

    assert lowered.expr == reference


def test_compare_gt():
    def comp_gt(a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], float64]):
        return a > b

    parsed = FieldOperatorParser.apply_to_function(comp_gt)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("greater")("a", "b")

    assert lowered.expr == reference


def test_compare_lt():
    def comp_lt(a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], float64]):
        return a < b

    parsed = FieldOperatorParser.apply_to_function(comp_lt)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("less")("a", "b")

    assert lowered.expr == reference


def test_compare_eq():
    def comp_eq(a: gtx.Field[[TDim], "int64"], b: gtx.Field[[TDim], "int64"]):
        return a == b

    parsed = FieldOperatorParser.apply_to_function(comp_eq)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("eq")("a", "b")

    assert lowered.expr == reference


def test_compare_chain():
    def compare_chain(
        a: gtx.Field[[IDim], float64], b: gtx.Field[[IDim], float64], c: gtx.Field[[IDim], float64]
    ) -> gtx.Field[[IDim], bool]:
        return a > b > c

    parsed = FieldOperatorParser.apply_to_function(compare_chain)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil("and_")(
        im.promote_to_lifted_stencil("greater")("a", "b"),
        im.promote_to_lifted_stencil("greater")("b", "c"),
    )

    assert lowered.expr == reference


def test_reduction_lowering_simple():
    def reduction(edge_f: gtx.Field[[Edge], float64]):
        return neighbor_sum(edge_f(V2E), axis=V2EDim)

    parsed = FieldOperatorParser.apply_to_function(reduction)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.promote_to_lifted_stencil(
        im.call(
            im.call("reduce")(
                "plus",
                im.deref(im.promote_to_const_iterator(im.literal(value="0", typename="float64"))),
            )
        )
    )(im.lifted_neighbors("V2E", "edge_f"))

    assert lowered.expr == reference


def test_reduction_lowering_expr():
    def reduction(e1: gtx.Field[[Edge], float64], e2: gtx.Field[[Vertex, V2EDim], float64]):
        e1_nbh = e1(V2E)
        return neighbor_sum(1.1 * (e1_nbh + e2), axis=V2EDim)

    parsed = FieldOperatorParser.apply_to_function(reduction)
    lowered = FieldOperatorLowering.apply(parsed)

    mapped = im.promote_to_lifted_stencil(im.map_("multiplies"))(
        im.promote_to_lifted_stencil("make_const_list")(
            im.promote_to_const_iterator(im.literal("1.1", "float64"))
        ),
        im.promote_to_lifted_stencil(im.map_("plus"))(ssa.unique_name("e1_nbh", 0), "e2"),
    )

    reference = im.let(
        ssa.unique_name("e1_nbh", 0),
        im.lifted_neighbors("V2E", "e1"),
    )(
        im.promote_to_lifted_stencil(
            im.call(
                im.call("reduce")(
                    "plus",
                    im.deref(
                        im.promote_to_const_iterator(im.literal(value="0", typename="float64"))
                    ),
                )
            )
        )(mapped)
    )

    assert lowered.expr == reference


def test_builtin_int_constructors():
    def int_constrs() -> tuple[int32, int32, int64, int32, int64]:
        return 1, int32(1), int64(1), int32("1"), int64("1")

    parsed = FieldOperatorParser.apply_to_function(int_constrs)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.make_tuple(
        im.promote_to_const_iterator(im.literal("1", "int32")),
        im.promote_to_const_iterator(im.literal("1", "int32")),
        im.promote_to_const_iterator(im.literal("1", "int64")),
        im.promote_to_const_iterator(im.literal("1", "int32")),
        im.promote_to_const_iterator(im.literal("1", "int64")),
    )

    assert lowered.expr == reference


def test_builtin_float_constructors():
    def float_constrs() -> tuple[float, float, float32, float64, float, float32, float64]:
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

    reference = im.make_tuple(
        im.promote_to_const_iterator(im.literal("0.1", "float64")),
        im.promote_to_const_iterator(im.literal("0.1", "float64")),
        im.promote_to_const_iterator(im.literal("0.1", "float32")),
        im.promote_to_const_iterator(im.literal("0.1", "float64")),
        im.promote_to_const_iterator(im.literal(".1", "float64")),
        im.promote_to_const_iterator(im.literal(".1", "float32")),
        im.promote_to_const_iterator(im.literal(".1", "float64")),
    )

    assert lowered.expr == reference


def test_builtin_bool_constructors():
    def bool_constrs() -> tuple[bool, bool, bool, bool, bool, bool, bool, bool]:
        return True, False, bool(True), bool(False), bool(0), bool(5), bool("True"), bool("False")

    parsed = FieldOperatorParser.apply_to_function(bool_constrs)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.make_tuple(
        im.promote_to_const_iterator(im.literal(str(True), "bool")),
        im.promote_to_const_iterator(im.literal(str(False), "bool")),
        im.promote_to_const_iterator(im.literal(str(True), "bool")),
        im.promote_to_const_iterator(im.literal(str(False), "bool")),
        im.promote_to_const_iterator(im.literal(str(bool(0)), "bool")),
        im.promote_to_const_iterator(im.literal(str(bool(5)), "bool")),
        im.promote_to_const_iterator(im.literal(str(bool("True")), "bool")),
        im.promote_to_const_iterator(im.literal(str(bool("False")), "bool")),
    )

    assert lowered.expr == reference
