# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.check_inout_field import (
    InOutFieldReadWithOffset,
    find_inout_fields_read_with_offset,
)
from gt4py.next.type_system import type_specifications as ts


IDim = common.Dimension("IDim")
KDim = common.Dimension("KDim", kind=common.DimensionKind.VERTICAL)
E2VDim = common.Dimension("E2V", kind=common.DimensionKind.LOCAL)
Vertex = common.Dimension("Vertex")
Edge = common.Dimension("Edge")

offset_types = {
    "Ioff": ts.OffsetType(source=IDim, target=(IDim,)),
    "Koff": ts.OffsetType(source=KDim, target=(KDim,)),
    "E2V": ts.OffsetType(source=Vertex, target=(Edge, E2VDim)),
}


def shifted(offset: str, index: int):
    return im.as_fieldop(im.lambda_("it")(im.deref(im.shift(offset, index)("it"))))


def plus_one(field: itir.Expr) -> itir.Expr:
    return im.as_fieldop(im.lambda_("it")(im.plus(im.deref("it"), 1.0)))(field)


def check(expr: itir.Expr, target: itir.Expr | str, **kwargs) -> list[InOutFieldReadWithOffset]:
    return find_inout_fields_read_with_offset(
        expr, im.ensure_expr(target), offset_types=offset_types, **kwargs
    )


@pytest.mark.parametrize(
    "expr, offsets",
    [
        # f ← (⇑(λ(it) → ·⟪Koffₒ, 1ₒ⟫(it)))(f), i.e. `A[0:10] = A[1:11]`
        (shifted("Koff", 1)("f"), ("shifted by +1 along 'KDim'",)),
        (shifted("Koff", -1)("f"), ("shifted by -1 along 'KDim'",)),
        # arithmetic on the shifted value
        (plus_one(shifted("Koff", 1)("f")), ("shifted by +1 along 'KDim'",)),
        # shifts compose across stencils
        (
            shifted("Ioff", 1)(shifted("Koff", 2)("f")),
            ("shifted by +1 along 'IDim' and shifted by +2 along 'KDim'",),
        ),
        # the first neighbor is not the point itself
        (shifted("E2V", 0)("f"), ("through 'E2V'",)),
        (
            im.as_fieldop(im.lambda_("it")(im.reduce("plus", 0.0)(im.neighbors("E2V", "it"))))("f"),
            ("through 'E2V'",),
        ),
        # dynamic offset, as lowered from `f(as_offset(Koff, offsets))`
        (
            im.as_fieldop(
                im.lambda_("it", "offset")(
                    im.deref(im.shift(im.cartesian_offset(KDim), im.deref("offset"))("it"))
                )
            )("f", "offsets"),
            ("at a data-dependent offset",),
        ),
    ],
)
def test_shifted_read_of_target(expr, offsets):
    assert check(expr, "f") == [InOutFieldReadWithOffset("f", (), offsets)]


@pytest.mark.parametrize(
    "expr",
    [
        im.ref("f"),
        plus_one("f"),
        shifted("Koff", 0)("f"),
        # shifts that cancel out
        shifted("Koff", -1)(shifted("Koff", 1)("f")),
        # a shifted read of another field
        im.as_fieldop(
            im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref(im.shift("Koff", 1)("b"))))
        )("f", "g"),
        # a let-bound field that is not shifted
        im.let("tmp", plus_one("f"))(plus_one("tmp")),
    ],
)
def test_pointwise_read_of_target(expr):
    assert check(expr, "f") == []


def test_shifted_read_of_other_field():
    assert check(shifted("Koff", 1)("f"), "g") == []


def test_shifted_read_through_let():
    expr = im.let("tmp", shifted("Koff", 1)("f"))(plus_one("tmp"))
    assert check(expr, "f") == [InOutFieldReadWithOffset("f", (), ("shifted by +1 along 'KDim'",))]


def test_shifted_read_in_let_bound_lambda():
    # as lowered from an `if` statement that returns in one branch only: the code after the
    # statement becomes a lambda, called in the branch that does not return
    continuation = im.lambda_("x")(plus_one(shifted("Koff", 1)("x")))
    expr = im.let("cont", continuation)(im.if_("cond", "f", im.call("cont")("f")))
    assert check(expr, "f") == [InOutFieldReadWithOffset("f", (), ("shifted by +1 along 'KDim'",))]
    assert check(expr, "g") == []
    # the lambda can also read the field without getting it as an argument
    expr = im.let("cont", im.lambda_()(shifted("Koff", 1)("f")))(
        im.if_("cond", "f", im.call("cont")())
    )
    assert check(expr, "f") == [InOutFieldReadWithOffset("f", (), ("shifted by +1 along 'KDim'",))]
    # a lambda that is not called counts with what it reads by itself
    expr = im.let("cont", im.lambda_("x")(shifted("Koff", 1)("f")))(im.if_("cond", "f", "cont"))
    assert check(expr, "f") == [InOutFieldReadWithOffset("f", (), ("shifted by +1 along 'KDim'",))]


def test_shifted_read_through_function_definition():
    fundef = itir.FunctionDefinition(
        id="copy_from_above", params=[im.sym("x")], expr=shifted("Koff", 1)("x")
    )
    expr = im.call("copy_from_above")("f")
    assert check(expr, "f", function_definitions=[fundef]) == [
        InOutFieldReadWithOffset("f", (), ("shifted by +1 along 'KDim'",))
    ]


def test_function_definition_called_from_stencil():
    # e.g. a scan pass calling a field operator on scalars
    fundef = itir.FunctionDefinition(
        id="add", params=[im.sym("a"), im.sym("b")], expr=im.plus("a", "b")
    )
    stencil = im.lambda_("x", "y")(
        im.call("add")(im.deref("x"), im.deref(im.shift("Koff", 1)("y")))
    )
    expr = im.as_fieldop(stencil)("f", "g")
    assert check(expr, "f", function_definitions=[fundef]) == []
    assert check(expr, "g", function_definitions=[fundef]) == [
        InOutFieldReadWithOffset("g", (), ("shifted by +1 along 'KDim'",))
    ]


def test_multi_output_shifted_slot():
    # {f, g} ← {⟪Koff, 1⟫(f), g + 1}
    expr = im.make_tuple(shifted("Koff", 1)("f"), plus_one("g"))
    assert check(expr, im.make_tuple("f", "g")) == [
        InOutFieldReadWithOffset("f", (), ("shifted by +1 along 'KDim'",))
    ]


def test_multi_output_passthrough():
    # {g, f} ← {⟪Koff, 1⟫(f), f}: writing `f` with its own value does not change memory
    expr = im.make_tuple(shifted("Koff", 1)("f"), "f")
    assert check(expr, im.make_tuple("g", "f")) == []
    # also through an identity stencil and a let
    expr = im.let("x", "f")(im.make_tuple(shifted("Koff", 1)("x"), im.as_fieldop("deref")("x")))
    assert check(expr, im.make_tuple("g", "f")) == []


def test_multi_output_shifted_read_in_other_slot():
    # {g, f} ← {⟪Koff, 1⟫(f), f + 1}: `g` may read `f` after it was written
    expr = im.make_tuple(shifted("Koff", 1)("f"), plus_one("f"))
    assert check(expr, im.make_tuple("g", "f")) == [
        InOutFieldReadWithOffset("f", (), ("shifted by +1 along 'KDim'",))
    ]


def test_selection_read_at_offset():
    # f_new = f + 1 if cond else f;  {f, g} ← {f_new, ⟪Koff, 1⟫(f_new)}
    # either `f` is written with its own value, or the shifted read does not read `f`
    computed = im.as_fieldop(im.lambda_("it")(im.plus(im.deref("it"), 1.0)))("h")
    expr = im.let("f_new", im.if_("cond", computed, "f"))(
        im.make_tuple("f_new", shifted("Koff", 1)("f_new"))
    )
    assert check(expr, im.make_tuple("f", "g")) == []


def test_selection_reading_target_is_not_harmless():
    # f_new = f + 1 if cond else f, the computed branch reads `f` too
    expr = im.let("f_new", im.if_("cond", plus_one("f"), "f"))(
        im.make_tuple("f_new", shifted("Koff", 1)("f_new"))
    )
    assert check(expr, im.make_tuple("f", "g")) == [
        InOutFieldReadWithOffset("f", (), ("shifted by +1 along 'KDim'",))
    ]


def test_shifted_read_of_other_selection():
    # the shifted read goes through a different selection than the one written to `f`
    computed = im.as_fieldop(im.lambda_("it")(im.plus(im.deref("it"), 1.0)))("h")
    expr = im.make_tuple(
        im.if_("cond", computed, "f"),
        shifted("Koff", 1)(im.if_("other_cond", computed, "f")),
    )
    assert check(expr, im.make_tuple("f", "g")) == [
        InOutFieldReadWithOffset("f", (), ("shifted by +1 along 'KDim'",))
    ]


def test_tuple_parameter():
    # t[0] ← ⟪Koff, 1⟫(t[0])
    expr = shifted("Koff", 1)(im.tuple_get(0, "t"))
    assert check(expr, im.tuple_get(0, "t")) == [
        InOutFieldReadWithOffset("t", (0,), ("shifted by +1 along 'KDim'",))
    ]
    assert check(expr, im.tuple_get(1, "t")) == []
    # writing the whole tuple
    expr = im.make_tuple(shifted("Koff", 1)(im.tuple_get(0, "t")), im.tuple_get(1, "t"))
    assert check(expr, "t") == [
        InOutFieldReadWithOffset("t", (0,), ("shifted by +1 along 'KDim'",))
    ]
    # `t[1]` is read at an offset, but written with its own value
    expr = im.make_tuple(shifted("Koff", 1)(im.tuple_get(1, "t")), im.tuple_get(1, "t"))
    assert check(expr, "t") == []


def test_cartesian_offset():
    # f ← f(KDim + 1)
    expr = im.as_fieldop(im.lambda_("it")(im.deref(im.shift(im.cartesian_offset(KDim), 1)("it"))))(
        "f"
    )
    assert check(expr, "f") == [InOutFieldReadWithOffset("f", (), ("shifted by +1 along 'KDim'",))]


def test_staggered_cartesian_offsets_cancel():
    # f ← f(KHalfDim + 0.5)(KDim - 0.5), which is `f` itself
    k_half = common.flip_staggered(KDim)
    to_half = common.connectivity_for_cartesian_shift(k_half, 0.5)
    from_half = common.connectivity_for_cartesian_shift(KDim, -0.5)

    def shift(conn: common.CartesianConnectivity):
        return im.as_fieldop(
            im.lambda_("it")(
                im.deref(
                    im.shift(im.cartesian_offset(conn.domain_dim, conn.codomain), conn.offset)("it")
                )
            )
        )

    assert check(shift(from_half)(shift(to_half)("f")), "f") == []
    assert check(shift(to_half)("f"), "f") == [
        InOutFieldReadWithOffset("f", (), ("shifted by +0.5 along 'KDim'",))
    ]
