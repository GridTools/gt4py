# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.merge_let import MergeLet
from gt4py.next.type_system import type_specifications as ts


def test_merge_nested_let():
    testee = im.let("a", "arg2")(im.let("b", "arg1")(im.plus("a", "b")))
    expected = im.call(im.lambda_("a", "b")(im.plus("a", "b")))("arg2", "arg1")

    assert MergeLet().visit(testee) == expected


def test_merge_renames_colliding_inner_param():
    # (λ(a) → (λ(a) → a)(1))(i); the inner binder shadows the outer one and is renamed apart
    testee = im.let("a", "i")(im.let("a", 1)("a"))
    expected = im.call(im.lambda_("a", "a_")("a_"))("i", 1)

    assert MergeLet().visit(testee) == expected


def test_merge_renames_inner_param_spelled_by_outer_arg():
    # (λ(x) → (λ(b) → b)(1))(b); `b` is free in the outer argument, so the inner binder is
    # renamed rather than left to shadow it in the merged parameter list
    testee = im.let("x", "b")(im.let("b", 1)("b"))
    expected = im.call(im.lambda_("x", "b_")("b_"))("b", 1)

    assert MergeLet().visit(testee) == expected


def test_merge_is_alpha_invariant():
    """Renaming a bound symbol must not change whether, or how, the pass merges.

    The two testees differ only in the name of the inner binder, which is not visible
    outside the inner lambda. Both merge, and the results agree up to that same renaming,
    i.e. the rewrite is a function of the program's alpha-equivalence class rather than of
    the chosen spelling.
    """
    colliding = MergeLet().visit(im.let("a", "i")(im.let("a", 1)("a")))
    apart = MergeLet().visit(im.let("a", "i")(im.let("b", 1)("b")))

    assert apart == im.call(im.lambda_("a", "b")("b"))("i", 1)
    assert colliding.args == apart.args
    assert colliding.fun.expr == im.ref(colliding.fun.params[1].id)
    assert apart.fun.expr == im.ref(apart.fun.params[1].id)


def test_no_merge_if_inner_arg_uses_outer_param():
    # the one genuine obstacle: hoisting `a` would move it out of the binder it refers to
    testee = im.let("a", "i")(im.let("b", "a")("b"))

    assert MergeLet().visit(testee) == testee


def test_no_merge_if_inner_arg_uses_outer_param_under_collision():
    # same obstacle, but with the inner binder spelled like the outer one
    testee = im.let("a", "i")(im.let("a", "a")("a"))

    assert MergeLet().visit(testee) == testee


def test_rename_avoids_capture_by_nested_binder():
    # the fresh name must dodge a binder occurring inside the inner lambda's body, even one
    # that is never referenced, otherwise the renamed parameter gets captured by it
    inner = itir.Lambda(
        params=[itir.Sym(id="a")],
        expr=im.as_fieldop(itir.Lambda(params=[itir.Sym(id="a_")], expr=im.deref("a")))(),
    )
    testee = itir.FunCall(
        fun=itir.Lambda(
            params=[itir.Sym(id="a")], expr=itir.FunCall(fun=inner, args=[im.ref("q")])
        ),
        args=[im.ref("i")],
    )

    actual = MergeLet().visit(testee)

    renamed = actual.fun.params[1].id
    assert renamed != "a_"
    assert (
        actual.fun.expr
        == im.as_fieldop(itir.Lambda(params=[itir.Sym(id="a_")], expr=im.deref(renamed)))()
    )


def test_merge_preserves_type_of_renamed_param():
    dtype = ts.ScalarType(kind=ts.ScalarKind.FLOAT32)
    inner = itir.Lambda(params=[itir.Sym(id="a", type=dtype)], expr=im.ref("a"))
    testee = itir.FunCall(
        fun=itir.Lambda(
            params=[itir.Sym(id="a", type=dtype)],
            expr=itir.FunCall(fun=inner, args=[im.literal_from_value(1)]),
        ),
        args=[im.ref("i")],
    )

    actual = MergeLet().visit(testee)

    assert [param.type for param in actual.fun.params] == [dtype, dtype]
