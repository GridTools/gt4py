# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.eve import utils as eve_utils
from gt4py.next import (
    astype,
    broadcast,
    common,
    float32,
    float64,
    int32,
    int64,
    max_over,
    min_over,
    neighbor_sum,
    where,
)
from gt4py.next.ffront import type_specifications as ts_ffront
from gt4py.next.ffront.ast_passes import single_static_assign as ssa
from gt4py.next.ffront.experimental import as_offset
from gt4py.next.ffront.fbuiltins import exp, minimum
from gt4py.next.ffront.foast_to_gtir import FieldOperatorLowering
from gt4py.next.ffront.func_to_foast import FieldOperatorParser
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import inline_lambdas
from gt4py.next.type_system import type_specifications as ts, type_translation


Edge = gtx.Dimension("Edge")
Vertex = gtx.Dimension("Vertex")
V2EDim = gtx.Dimension("V2E", gtx.DimensionKind.LOCAL)
V2E = gtx.FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))

TDim = gtx.Dimension("TDim")
TOff = gtx.FieldOffset("TDim", source=TDim, target=(TDim,))
UDim = gtx.Dimension("UDim")


def test_return():
    def foo(inp: gtx.Field[[TDim], float64]):
        return inp

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.id == "foo"
    assert lowered.expr == im.ref("inp")


def test_return_literal_tuple():
    def foo():
        return (1.0, True)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.id == "foo"
    assert lowered.expr == im.make_tuple(im.literal_from_value(1.0), im.literal_from_value(True))


def test_field_and_scalar_arg():
    def foo(bar: gtx.Field[[TDim], int64], alpha: int64) -> gtx.Field[[TDim], int64]:
        return alpha * bar

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("multiplies")("alpha", "bar")  # no difference to non-scalar arg

    assert lowered.expr == reference


def test_scalar_arg_only():
    def foo(bar: int64, alpha: int64) -> int64:
        return alpha * bar

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.call("multiplies")("alpha", "bar")

    assert lowered.expr == reference


def test_multivalue_identity():
    def foo(inp1: gtx.Field[[TDim], float64], inp2: gtx.Field[[TDim], float64]):
        return inp1, inp2

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.make_tuple("inp1", "inp2")

    assert lowered.expr == reference


def test_premap():
    def foo(inp: gtx.Field[[TDim], float64]):
        return inp(TOff[1])

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.as_fieldop(im.lambda_("__it")(im.deref(im.shift("TOff", 1)("__it"))))("inp")

    assert lowered.expr == reference


def test_premap_cartesian_syntax():
    def foo(inp: gtx.Field[[TDim], float64]):
        return inp(TDim + 1)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.as_fieldop(
        im.lambda_("__it")(
            im.deref(im.shift(common.dimension_to_implicit_offset(TDim.value), 1)("__it"))
        )
    )("inp")

    assert lowered.expr == reference


def test_as_offset():
    def foo(inp: gtx.Field[[TDim], float64], offset: gtx.Field[[TDim], int]):
        return inp(as_offset(TOff, offset))

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.as_fieldop(
        im.lambda_("__it", "__offset")(im.deref(im.shift("TOff", im.deref("__offset"))("__it")))
    )("inp", "offset")

    assert lowered.expr == reference


def test_temp_assignment():
    def foo(inp: gtx.Field[[TDim], float64]):
        tmp = inp
        inp = tmp
        tmp2 = inp
        return tmp2

    parsed = FieldOperatorParser.apply_to_function(foo)
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


def test_where():
    def foo(
        a: gtx.Field[[TDim], bool], b: gtx.Field[[TDim], float64], c: gtx.Field[[TDim], float64]
    ):
        return where(a, b, c)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("if_")("a", "b", "c")

    assert lowered.expr == reference


def test_where_tuple():
    def foo(
        a: gtx.Field[[TDim], bool],
        b: tuple[gtx.Field[[TDim], float64], gtx.Field[[TDim], float64]],
        c: tuple[gtx.Field[[TDim], float64], gtx.Field[[TDim], float64]],
    ):
        return where(a, b, c)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    lowered_inlined = inline_lambdas.InlineLambdas.apply(
        lowered
    )  # we generate a let for the condition which is removed by inlining for easier testing

    reference = im.make_tuple(
        im.op_as_fieldop("if_")("a", im.tuple_get(0, "b"), im.tuple_get(0, "c")),
        im.op_as_fieldop("if_")("a", im.tuple_get(1, "b"), im.tuple_get(1, "c")),
    )

    assert lowered_inlined.expr == reference


def test_ternary():
    def foo(a: bool, b: gtx.Field[[TDim], float64], c: gtx.Field[[TDim], float64]):
        return b if a else c

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.if_("a", "b", "c")

    assert lowered.expr == reference


def test_if_unconditional_return():
    def foo(a: bool, b: gtx.Field[[TDim], float64], c: gtx.Field[[TDim], float64]):
        if a:
            return b
        else:
            return c

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.if_("a", "b", "c")

    assert lowered.expr == reference


def test_if_no_return():
    def foo(a: bool, b: gtx.Field[[TDim], float64], c: gtx.Field[[TDim], float64]):
        if a:
            res = b
        else:
            res = c
        return res

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)
    lowered_inlined = inline_lambdas.InlineLambdas.apply(lowered)
    lowered_inlined = inline_lambdas.InlineLambdas.apply(lowered_inlined)

    reference = im.tuple_get(0, im.if_("a", im.make_tuple("b"), im.make_tuple("c")))

    assert lowered_inlined.expr == reference


def test_if_conditional_return():
    def foo(a: bool, b: gtx.Field[[TDim], float64], c: gtx.Field[[TDim], float64]):
        if a:
            res = b
        else:
            if a:
                return c
            res = b
        return res

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)
    lowered_inlined = inline_lambdas.InlineLambdas.apply(lowered)
    lowered_inlined = inline_lambdas.InlineLambdas.apply(lowered_inlined)

    reference = im.if_("a", "b", im.if_("a", "c", "b"))

    assert lowered_inlined.expr == reference


def test_astype():
    def foo(a: gtx.Field[[TDim], float64]):
        return astype(a, int32)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)
    lowered_inlined = inline_lambdas.InlineLambdas.apply(lowered)

    reference = im.cast_as_fieldop("int32")("a")

    assert lowered_inlined.expr == reference


def test_astype_local_field():
    def foo(a: gtx.Field[gtx.Dims[Vertex, V2EDim], float64]):
        return astype(a, int32)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop(im.map_(im.lambda_("val")(im.call("cast_")("val", "int32"))))("a")

    assert lowered.expr == reference


def test_astype_scalar():
    def foo(a: float64):
        return astype(a, int32)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)
    lowered_inlined = inline_lambdas.InlineLambdas.apply(lowered)

    reference = im.call("cast_")("a", "int32")

    assert lowered_inlined.expr == reference


def test_astype_tuple():
    def foo(a: tuple[gtx.Field[[TDim], float64], gtx.Field[[TDim], float64]]):
        return astype(a, int32)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)
    lowered_inlined = inline_lambdas.InlineLambdas.apply(lowered)

    reference = im.make_tuple(
        im.cast_as_fieldop("int32")(im.tuple_get(0, "a")),
        im.cast_as_fieldop("int32")(im.tuple_get(1, "a")),
    )

    assert lowered_inlined.expr == reference


def test_astype_tuple_scalar_and_field():
    def foo(a: tuple[gtx.Field[[TDim], float64], float64]):
        return astype(a, int32)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)
    lowered_inlined = inline_lambdas.InlineLambdas.apply(lowered)

    reference = im.make_tuple(
        im.cast_as_fieldop("int32")(im.tuple_get(0, "a")),
        im.call("cast_")(im.tuple_get(1, "a"), "int32"),
    )

    assert lowered_inlined.expr == reference


def test_astype_nested_tuple():
    def foo(
        a: tuple[
            tuple[gtx.Field[[TDim], float64], gtx.Field[[TDim], float64]],
            gtx.Field[[TDim], float64],
        ],
    ):
        return astype(a, int32)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)
    lowered_inlined = inline_lambdas.InlineLambdas.apply(lowered)

    reference = im.make_tuple(
        im.make_tuple(
            im.cast_as_fieldop("int32")(im.tuple_get(0, im.tuple_get(0, "a"))),
            im.cast_as_fieldop("int32")(im.tuple_get(1, im.tuple_get(0, "a"))),
        ),
        im.cast_as_fieldop("int32")(im.tuple_get(1, "a")),
    )

    assert lowered_inlined.expr == reference


def test_unary_minus():
    def foo(inp: gtx.Field[[TDim], float64]):
        return -inp

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("neg")("inp")

    assert lowered.expr == reference


def test_unary_plus():
    def foo(inp: gtx.Field[[TDim], float64]):
        return +inp

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.ref("inp")

    assert lowered.expr == reference


@pytest.mark.parametrize("var, var_type", [("-1.0", "float64"), ("True", "bool")])
def test_unary_op_type_conversion(var, var_type):
    def unary_float():
        return float(-1)

    def unary_bool():
        return bool(-1)

    fun = unary_bool if var_type == "bool" else unary_float
    parsed = FieldOperatorParser.apply_to_function(fun)
    lowered = FieldOperatorLowering.apply(parsed)
    reference = im.literal(var, var_type)

    assert lowered.expr == reference


def test_unpacking():
    """Unpacking assigns should get separated."""

    def foo(
        inp1: gtx.Field[[TDim], float64], inp2: gtx.Field[[TDim], float64]
    ) -> gtx.Field[[TDim], float64]:
        tmp1, tmp2 = inp1, inp2  # noqa
        return tmp1

    parsed = FieldOperatorParser.apply_to_function(foo)
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

    def foo(inp: gtx.Field[[TDim], float64]):
        tmp: gtx.Field[[TDim], float64] = inp
        return tmp

    parsed = FieldOperatorParser.apply_to_function(foo)
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

    def foo(inp: gtx.Field[[TDim], float64]) -> gtx.Field[[TDim], float64]:
        return identity(inp)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.call("identity")("inp")

    assert lowered.expr == reference


def test_return_constructed_tuple():
    def foo(a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], int64]):
        return a, b

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.make_tuple("a", "b")

    assert lowered.expr == reference


def test_fieldop_with_tuple_arg():
    def foo(a: tuple[gtx.Field[[TDim], float64], gtx.Field[[TDim], float64]]):
        return a[0]

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.tuple_get(0, "a")

    assert lowered.expr == reference


def test_unary_not():
    def foo(cond: gtx.Field[[TDim], "bool"]):
        return not cond

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("not_")("cond")

    assert lowered.expr == reference


def test_binary_plus():
    def foo(a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], float64]):
        return a + b

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("plus")("a", "b")

    assert lowered.expr == reference


def test_add_scalar_literal_to_field():
    def foo(a: gtx.Field[[TDim], float64]) -> gtx.Field[[TDim], float64]:
        return 2.0 + a

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("plus")(im.literal("2.0", "float64"), "a")

    assert lowered.expr == reference


def test_add_scalar_literals():
    def foo(a: gtx.Field[[TDim], "int32"]) -> gtx.Field[[TDim], "int32"]:
        tmp = int32(1) + int32("1")
        return a + tmp

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.let(
        ssa.unique_name("tmp", 0),
        im.call("plus")(
            im.literal("1", "int32"),
            im.literal("1", "int32"),
        ),
    )(im.op_as_fieldop("plus")("a", ssa.unique_name("tmp", 0)))

    assert lowered.expr == reference


def test_literal_tuple():
    tup = eve_utils.FrozenNamespace(a=(4, 2))

    def foo():
        return tup.a

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.make_tuple(
        im.literal("4", "int32"),
        im.literal("2", "int32"),
    )

    assert lowered.expr == reference


def test_binary_mult():
    def foo(a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], float64]):
        return a * b

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("multiplies")("a", "b")

    assert lowered.expr == reference


def test_binary_minus():
    def foo(a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], float64]):
        return a - b

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("minus")("a", "b")

    assert lowered.expr == reference


def test_binary_div():
    def foo(a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], float64]):
        return a / b

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("divides")("a", "b")

    assert lowered.expr == reference


def test_binary_and():
    def foo(a: gtx.Field[[TDim], "bool"], b: gtx.Field[[TDim], "bool"]):
        return a & b

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("and_")("a", "b")

    assert lowered.expr == reference


def test_scalar_and():
    def foo(a: gtx.Field[[TDim], "bool"]) -> gtx.Field[[TDim], "bool"]:
        return a & False

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("and_")("a", im.literal("False", "bool"))

    assert lowered.expr == reference


def test_binary_or():
    def foo(a: gtx.Field[[TDim], "bool"], b: gtx.Field[[TDim], "bool"]):
        return a | b

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("or_")("a", "b")

    assert lowered.expr == reference


def test_compare_scalars():
    def foo() -> bool:
        return 3 > 4

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.call("greater")(
        im.literal("3", "int32"),
        im.literal("4", "int32"),
    )

    assert lowered.expr == reference


def test_compare_gt():
    def foo(a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], float64]):
        return a > b

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("greater")("a", "b")

    assert lowered.expr == reference


def test_compare_lt():
    def foo(a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], float64]):
        return a < b

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("less")("a", "b")

    assert lowered.expr == reference


def test_compare_eq():
    def foo(a: gtx.Field[[TDim], "int64"], b: gtx.Field[[TDim], "int64"]):
        return a == b

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("eq")("a", "b")

    assert lowered.expr == reference


def test_compare_chain():
    def foo(
        a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], float64], c: gtx.Field[[TDim], float64]
    ) -> gtx.Field[[TDim], bool]:
        return a > b > c

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("and_")(
        im.op_as_fieldop("greater")("a", "b"),
        im.op_as_fieldop("greater")("b", "c"),
    )

    assert lowered.expr == reference


def test_unary_math_builtin():
    def foo(a: gtx.Field[[TDim], float64]):
        return exp(a)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("exp")("a")

    assert lowered.expr == reference


def test_binary_math_builtin():
    def foo(a: gtx.Field[[TDim], float64], b: gtx.Field[[TDim], float64]):
        return minimum(a, b)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop("minimum")("a", "b")

    assert lowered.expr == reference


def test_premap_to_local_field():
    def foo(edge_f: gtx.Field[gtx.Dims[Edge], float64]):
        return edge_f(V2E)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.as_fieldop_neighbors("V2E", "edge_f")

    assert lowered.expr == reference


def test_reduction_lowering_neighbor_sum():
    def foo(edge_f: gtx.Field[[Edge], float64]):
        return neighbor_sum(edge_f(V2E), axis=V2EDim)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop(
        im.call(
            im.call("reduce")(
                "plus",
                im.literal(value="0", typename="float64"),
            )
        )
    )(im.as_fieldop_neighbors("V2E", "edge_f"))

    assert lowered.expr == reference


def test_reduction_lowering_max_over():
    def foo(edge_f: gtx.Field[[Edge], float64]):
        return max_over(edge_f(V2E), axis=V2EDim)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop(
        im.call(
            im.call("reduce")(
                "maximum",
                im.literal(value=str(np.finfo(np.float64).min), typename="float64"),
            )
        )
    )(im.as_fieldop_neighbors("V2E", "edge_f"))

    assert lowered.expr == reference


def test_reduction_lowering_min_over():
    def foo(edge_f: gtx.Field[[Edge], float64]):
        return min_over(edge_f(V2E), axis=V2EDim)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.op_as_fieldop(
        im.call(
            im.call("reduce")(
                "minimum",
                im.literal(value=str(np.finfo(np.float64).max), typename="float64"),
            )
        )
    )(im.as_fieldop_neighbors("V2E", "edge_f"))

    assert lowered.expr == reference


def test_reduction_lowering_expr():
    def foo(e1: gtx.Field[[Edge], float64], e2: gtx.Field[[Vertex, V2EDim], float64]):
        e1_nbh = e1(V2E)
        return neighbor_sum(1.1 * (e1_nbh + e2), axis=V2EDim)

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    mapped = im.op_as_fieldop(im.map_("multiplies"))(
        im.op_as_fieldop("make_const_list")(im.literal("1.1", "float64")),
        im.op_as_fieldop(im.map_("plus"))(ssa.unique_name("e1_nbh", 0), "e2"),
    )

    reference = im.let(
        ssa.unique_name("e1_nbh", 0),
        im.as_fieldop_neighbors("V2E", "e1"),
    )(
        im.op_as_fieldop(
            im.call(
                im.call("reduce")(
                    "plus",
                    im.literal(value="0", typename="float64"),
                )
            )
        )(mapped)
    )

    assert lowered.expr == reference


def test_builtin_int_constructors():
    def foo() -> tuple[int32, int32, int64, int32, int64]:
        return 1, int32(1), int64(1), int32("1"), int64("1")

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.make_tuple(
        im.literal("1", "int32"),
        im.literal("1", "int32"),
        im.literal("1", "int64"),
        im.literal("1", "int32"),
        im.literal("1", "int64"),
    )

    assert lowered.expr == reference


def test_builtin_float_constructors():
    def foo() -> tuple[float, float, float32, float64, float, float32, float64]:
        return (
            0.1,
            float(0.1),
            float32(0.1),
            float64(0.1),
            float(".1"),
            float32(".1"),
            float64(".1"),
        )

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.make_tuple(
        im.literal("0.1", "float64"),
        im.literal("0.1", "float64"),
        im.literal("0.1", "float32"),
        im.literal("0.1", "float64"),
        im.literal("0.1", "float64"),
        im.literal("0.1", "float32"),
        im.literal("0.1", "float64"),
    )

    assert lowered.expr == reference


def test_builtin_bool_constructors():
    def foo() -> tuple[bool, bool, bool, bool, bool, bool, bool, bool]:
        return True, False, bool(True), bool(False), bool(0), bool(5), bool("True"), bool("False")

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    reference = im.make_tuple(
        im.literal(str(True), "bool"),
        im.literal(str(False), "bool"),
        im.literal(str(True), "bool"),
        im.literal(str(False), "bool"),
        im.literal(str(bool(0)), "bool"),
        im.literal(str(bool(5)), "bool"),
        im.literal(str(bool("True")), "bool"),
        im.literal(str(bool("False")), "bool"),
    )

    assert lowered.expr == reference


def test_broadcast():
    def foo(inp: gtx.Field[[TDim], float64]):
        return broadcast(inp, (UDim, TDim))

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.id == "foo"
    assert lowered.expr == im.as_fieldop("deref")(im.ref("inp"))


def test_scalar_broadcast():
    def foo():
        return broadcast(1, (UDim, TDim))

    parsed = FieldOperatorParser.apply_to_function(foo)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.id == "foo"
    assert lowered.expr == im.as_fieldop("deref")(1)
