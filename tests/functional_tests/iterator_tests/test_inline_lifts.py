import pytest

from functional.ffront import itir_makers as im
from functional.iterator import ir
from functional.iterator.transforms.inline_lifts import InlineLifts


def inline_lift_test_data():
    return [
        # (testee, expected)
        (
            # deref(lift(f)(args...)) -> f(args...)
            im.deref_(im.lift_("f")("arg")),
            im.call_("f")("arg"),
        ),
        (
            # deref(shift(...)(lift(f)(args...))) -> f(shift(...)(args)...)
            im.deref_(im.shift_("I", 1)(im.lift_("f")("arg"))),
            im.call_("f")(im.shift_("I", 1)("arg")),
        ),
        (
            # can_deref(lift(f)(args...)) -> and(can_deref(arg[0]), and(can_deref(arg[1]), ...))
            im.call_("can_deref")(im.lift_("f")("arg1", "arg2")),
            im.and__(im.call_("can_deref")("arg1"), im.call_("can_deref")("arg2")),
        ),
        (
            # can_deref(shift(...)(lift(f)(args...)) -> and(can_deref(shift(...)(arg[0])), and(can_deref(shift(...)(arg[1])), ...))
            im.call_("can_deref")(im.shift_("I", 1)(im.lift_("f")("arg1", "arg2"))),
            im.and__(
                im.call_("can_deref")(im.shift_("I", 1)("arg1")),
                im.call_("can_deref")(im.shift_("I", 1)("arg2")),
            ),
        ),
        (
            # (↑(λ(arg1, arg2) → ·arg1 + ·arg2))((↑(λ(arg1, arg2) → ·arg1 × ·arg2))(a, b), c)
            im.lift_(
                im.lambda__(im.sym("arg1"), im.sym("arg2"))(
                    im.plus_(im.deref_("arg1"), im.deref_("arg2"))
                )
            )(
                im.lift_(
                    im.lambda__(im.sym("arg1"), im.sym("arg2"))(
                        im.multiplies_(im.deref_("arg1"), im.deref_("arg2"))
                    )
                )(im.ref("a"), im.ref("b")),
                im.ref("c"),
            ),
            # (↑(λ(a, b, c) → ·a × ·b + ·c))(a, b, c)
            im.lift_(
                im.lambda__("a", "b", "c")(
                    im.plus_(im.multiplies_(im.deref_("a"), im.deref_("b")), im.deref_("c"))
                )
            )(im.ref("a"), im.ref("b"), im.ref("c")),
        ),
        (
            # (↑(λ(arg1, arg2) → ·arg1 + ·arg2))(arg1, (↑(λ(arg1) → ·arg1))(f()))
            im.lift_(
                im.lambda__(im.sym("arg1"), im.sym("arg2"))(
                    im.plus_(im.deref_("arg1"), im.deref_("arg2"))
                )
            )(
                im.ref("arg1"),
                # The `f()` argument must not be inlined and becomes an argument to the outer
                # lift. The inliner assigns a new symbol for this argument that by default is
                # named like the respective argument of the inner lift, i.e. here the `arg1`
                # below. Here we test that if this symbol collides with a symbol of the outer
                # lifted stencil, i.e. the lambda function above, the collision is properly
                # resolved.
                im.lift_(im.lambda__(im.sym("arg1"))(im.deref_("arg1")))(im.call_("f")()),
            ),
            # (↑(λ(arg1, arg1_) → ·arg1 + ·arg1_))(arg1, f())
            im.lift_(im.lambda__("arg1", "arg1_")(im.plus_(im.deref_("arg1"), im.deref_("arg1_"))))(
                im.ref("arg1"), im.call_("f")()
            ),
        ),
        (
            # similar to the test case above, but the collision is with a symbol from
            # the outer scope
            # λ(arg1) → (↑(λ(arg2) → ·arg2 + arg1))((↑(λ(arg1) → ·arg1))(f()))
            im.lambda__("arg1")(
                im.lift_(im.lambda__(im.sym("arg2"))(im.plus_(im.deref_("arg2"), im.ref("arg1"))))(
                    im.lift_(im.lambda__(im.sym("arg1"))(im.deref_("arg1")))(im.call_("f")())
                )
            ),
            # λ(arg1) → (↑(λ(arg1_) → ·arg1_ + arg1))(f())
            im.lambda__("arg1")(
                im.lift_(
                    im.lambda__(im.sym("arg1_"))(im.plus_(im.deref_("arg1_"), im.ref("arg1")))
                )(im.call_("f")())
            ),
        ),
    ]


@pytest.mark.parametrize("testee, expected", inline_lift_test_data())
def test_deref_lift(testee, expected):
    result = InlineLifts().visit(testee)
    assert result == expected
