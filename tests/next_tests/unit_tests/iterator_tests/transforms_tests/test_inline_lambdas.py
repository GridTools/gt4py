# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next.type_system import type_specifications as ts
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas


test_data = [
    ("identity1", im.call(im.lambda_("x")("x"))("y"), im.ref("y")),
    ("identity2", im.call(im.lambda_("x")("x"))(im.plus("y", "y")), im.plus("y", "y")),
    ("unused_param", im.call(im.lambda_("x", "y")("x"))("x", "y"), im.ref("x")),
    (
        "composed_addition",
        im.call(im.lambda_("x")(im.plus("x", "x")))(im.plus("y", "y")),
        {
            True: im.call(im.lambda_("x")(im.plus("x", "x")))(im.plus("y", "y")),  # stays as is
            False: im.plus(im.plus("y", "y"), im.plus("y", "y")),
        },
    ),
    (
        "name_collision",
        im.call(im.lambda_("x")(im.plus("x", "x")))(im.plus("x", "y")),
        {
            True: im.call(im.lambda_("x")(im.plus("x", "x")))(im.plus("x", "y")),  # stays as is
            False: im.plus(im.plus("x", "y"), im.plus("x", "y")),
        },
    ),
    (
        "name_shadowing",
        im.call(im.lambda_("x")(im.multiplies_(im.call(im.lambda_("x")(im.plus("x", 1)))(2), "x")))(
            im.plus("x", "x")
        ),
        im.multiplies_(im.plus(2, 1), im.plus("x", "x")),
    ),
    (
        # ensure opcount preserving option works whether `itir.SymRef` has a type or not
        "typed_ref",
        im.let("a", im.call("opaque")())(
            im.plus(im.ref("a", ts.ScalarType(kind=ts.ScalarKind.FLOAT32)), im.ref("a", None))
        ),
        {
            True: im.let("a", im.call("opaque")())(
                im.plus(  # stays as is
                    im.ref("a", ts.ScalarType(kind=ts.ScalarKind.FLOAT32)), im.ref("a", None)
                )
            ),
            False: im.plus(im.call("opaque")(), im.call("opaque")()),
        },
    ),
    (
        # symbol clash when partially inlining (opcount preserving)
        "symbol_clash",
        im.call(im.lambda_("x", "y")(im.call("f")("x", im.plus("x", "y"))))(im.plus("y", "y"), "x"),
        {
            True: im.call(im.lambda_("x_")(im.call("f")("x_", im.plus("x_", "x"))))(
                im.plus("y", "y")
            ),
            False: im.call("f")(im.plus("y", "y"), im.plus(im.plus("y", "y"), "x")),
        },
    ),
]


@pytest.mark.parametrize("opcount_preserving", [True, False])
@pytest.mark.parametrize("name,testee,expected", test_data)
def test(name, opcount_preserving, testee, expected):
    if isinstance(expected, dict):
        expected = expected[opcount_preserving]

    inlined = InlineLambdas.apply(testee, opcount_preserving=opcount_preserving)
    assert inlined == expected


def test_inline_lambda_args():
    testee = im.let("reduce_step", im.lambda_("x", "y")(im.plus("x", "y")))(
        im.lambda_("a")(
            im.call("reduce_step")(im.call("reduce_step")(im.call("reduce_step")("a", 1), 2), 3)
        )
    )
    expected = im.lambda_("a")(
        im.call(im.lambda_("x", "y")(im.plus("x", "y")))(
            im.call(im.lambda_("x", "y")(im.plus("x", "y")))(
                im.call(im.lambda_("x", "y")(im.plus("x", "y")))("a", 1), 2
            ),
            3,
        )
    )
    inlined = InlineLambdas.apply(testee, opcount_preserving=True, force_inline_lambda_args=True)
    assert inlined == expected


def test_type_preservation():
    testee = im.let("a", "b")("a")
    testee.type = testee.annex.type = ts.ScalarType(kind=ts.ScalarKind.FLOAT32)
    inlined = InlineLambdas.apply(testee)
    assert inlined.type == inlined.annex.type == ts.ScalarType(kind=ts.ScalarKind.FLOAT32)


def test_dbg1():
    #      (λ(foo, bar) →
    #    (λ(foo, bar) →
    #       if_(c0,
    #                    foo,
    #                    if_(c1, foo, bar)))(
    #      if_(c2, foo, bar), foo
    #    ))(a, b)

    testee = im.call(
        im.lambda_("foo", "bar")(
            im.call(
                im.lambda_("foo", "bar")(
                    im.if_(im.ref("c0"), "foo", im.if_(im.ref("c1"), "foo", "bar"))
                )
            )(
                im.if_(im.ref("c2"), "foo", "bar"),
                "foo",
            )
        )
    )("a", "b")
    print(testee)
    inlined = InlineLambdas.apply(testee, opcount_preserving=True)
    print(inlined)
    inlined = InlineLambdas.apply(inlined, opcount_preserving=False)
    print(inlined)

    # if c0 then b else if c1 then b else b

    # expected:
    #     if_(c0,
    #   if_(c2, a, b),
    #   if_(c1,
    #     if_(c2, a, b),
    #     a)
    #   )
    expected = im.if_(
        im.ref("c0"),
        im.if_(im.ref("c2"), "a", "b"),
        im.if_(im.ref("c1"), im.if_(im.ref("c2"), "a", "b"), "a"),
    )
    print(expected)
    assert inlined == expected


# def test_dbg2():
#     testee = im.call(
#         im.lambda_("x", "y")(im.multiplies_(im.call(im.lambda_("x")(im.plus("x", 1)))("y"), "x"))
#     )(im.plus("x", "x"), "x")

#     print(testee)
#     inlined = InlineLambdas.apply(testee, opcount_preserving=True)
#     print(inlined)


def test_dbg2():
    testee = im.call(
        im.lambda_("x", "y")(
            im.call(im.lambda_("x", "y")(im.call("f")("x", im.plus("x", "y"))))(
                im.plus("y", "y"), "x"
            )
        )
    )("a", "b")

    print(testee)
    inlined = InlineLambdas.apply(testee, opcount_preserving=True)
    print(inlined)
    inlined = InlineLambdas.apply(inlined, opcount_preserving=False)
    print(inlined)

    direct = InlineLambdas.apply(testee, opcount_preserving=False)
    print(direct)


def test_dbg3():
    testee = im.call(im.lambda_("x", "y")(im.call("f")("x", im.plus("x", "y"))))(
        im.plus("y", "y"), "x"
    )

    print(testee)
    # inlined = InlineLambdas.apply(testee, opcount_preserving=True)
    inlined = InlineLambdas.apply(testee, opcount_preserving=True)
    print(inlined)
    # inlined = inline_lambda(testee, opcount_preserving=False)
    # print(inlined)

    # expected = (λ(x) → f(x_, x_ + x))(y + y)
    expected = im.call(im.lambda_("x_")(im.call("f")("x_", im.plus("x_", "x"))))(im.plus("y", "y"))
    print(expected)
    assert inlined == expected

    # inlined = InlineLambdas.apply(inlined, opcount_preserving=False)
    # print(inlined)

    # direct = InlineLambdas.apply(testee, opcount_preserving=False)
    # print(direct)
