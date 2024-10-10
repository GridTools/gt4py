# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.inline_lifts import InlineLifts


def inline_lift_test_data():
    return [
        # (testee, expected)
        (
            # deref(lift(f)(args...)) -> f(args...)
            im.deref(im.lift("f")("arg")),
            im.call("f")("arg"),
        ),
        (
            # deref(shift(...)(lift(f)(args...))) -> f(shift(...)(args)...)
            im.deref(im.shift("I", 1)(im.lift("f")("arg"))),
            im.call("f")(im.shift("I", 1)("arg")),
        ),
        (
            # can_deref(lift(f)(args...)) -> and(can_deref(arg[0]), and(can_deref(arg[1]), ...))
            im.call("can_deref")(im.lift("f")("arg1", "arg2")),
            im.and_(im.call("can_deref")("arg1"), im.call("can_deref")("arg2")),
        ),
        (
            # can_deref(shift(...)(lift(f)(args...)) -> and(can_deref(shift(...)(arg[0])), and(can_deref(shift(...)(arg[1])), ...))
            im.call("can_deref")(im.shift("I", 1)(im.lift("f")("arg1", "arg2"))),
            im.and_(
                im.call("can_deref")(im.shift("I", 1)("arg1")),
                im.call("can_deref")(im.shift("I", 1)("arg2")),
            ),
        ),
        (
            # (↑(λ(arg1, arg2) → ·arg1 + ·arg2))((↑(λ(arg1, arg2) → ·arg1 × ·arg2))(a, b), c)
            im.lift(
                im.lambda_(im.sym("arg1"), im.sym("arg2"))(
                    im.plus(im.deref("arg1"), im.deref("arg2"))
                )
            )(
                im.lift(
                    im.lambda_(im.sym("arg1"), im.sym("arg2"))(
                        im.multiplies_(im.deref("arg1"), im.deref("arg2"))
                    )
                )(im.ref("a"), im.ref("b")),
                im.ref("c"),
            ),
            # (↑(λ(a, b, c) → ·a × ·b + ·c))(a, b, c)
            im.lift(
                im.lambda_("a", "b", "c")(
                    im.plus(im.multiplies_(im.deref("a"), im.deref("b")), im.deref("c"))
                )
            )(im.ref("a"), im.ref("b"), im.ref("c")),
        ),
        (
            # (↑(λ(arg1, arg2) → ·arg1 + ·arg2))(arg1, (↑(λ(arg1) → ·arg1))(f()))
            im.lift(
                im.lambda_(im.sym("arg1"), im.sym("arg2"))(
                    im.plus(im.deref("arg1"), im.deref("arg2"))
                )
            )(
                im.ref("arg1"),
                # The `f()` argument must not be inlined and becomes an argument to the outer
                # lift. The inliner assigns a new symbol for this argument that by default is
                # named like the respective argument of the inner lift, i.e. here the `arg1`
                # below. Here we test that if this symbol collides with a symbol of the outer
                # lifted stencil, i.e. the lambda function above, the collision is properly
                # resolved.
                im.lift(im.lambda_(im.sym("arg1"))(im.deref("arg1")))(im.call("f")()),
            ),
            # (↑(λ(arg1, arg1_) → ·arg1 + ·arg1_))(arg1, f())
            im.lift(im.lambda_("arg1", "arg1_")(im.plus(im.deref("arg1"), im.deref("arg1_"))))(
                im.ref("arg1"), im.call("f")()
            ),
        ),
        (
            # similar to the test case above, but the collision is with a symbol from
            # the outer scope
            # λ(arg1) → (↑(λ(arg2) → ·arg2 + arg1))((↑(λ(arg1) → ·arg1))(f()))
            im.lambda_("arg1")(
                im.lift(im.lambda_(im.sym("arg2"))(im.plus(im.deref("arg2"), im.ref("arg1"))))(
                    im.lift(im.lambda_(im.sym("arg1"))(im.deref("arg1")))(im.call("f")())
                )
            ),
            # λ(arg1) → (↑(λ(arg1_) → ·arg1_ + arg1))(f())
            im.lambda_("arg1")(
                im.lift(im.lambda_(im.sym("arg1_"))(im.plus(im.deref("arg1_"), im.ref("arg1"))))(
                    im.call("f")()
                )
            ),
        ),
    ]


@pytest.mark.parametrize("testee, expected", inline_lift_test_data())
def test_deref_lift(testee, expected):
    result = InlineLifts().visit(testee)
    assert result == expected
