import pytest

from functional.iterator import ir
from functional.iterator.transforms.inline_lifts import InlineLifts


def inline_lift_test_data():
    foo_shift = ir.FunCall(
        fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1)]
    )
    return [
        # (testee, expected)
        (
            # deref(lift(f)(args...)) -> f(args...)
            ir.FunCall(
                fun=ir.SymRef(id="deref"),
                args=[
                    ir.FunCall(
                        fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="f")]),
                        args=[ir.SymRef(id="arg")],
                    )
                ],
            ),
            ir.FunCall(fun=ir.SymRef(id="f"), args=[ir.SymRef(id="arg")]),
        ),
        (
            # deref(shift(...)(lift(f)(args...))) -> f(shift(...)(args)...)
            ir.FunCall(
                fun=ir.SymRef(id="deref"),
                args=[
                    ir.FunCall(
                        fun=foo_shift,
                        args=[
                            ir.FunCall(
                                fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="f")]),
                                args=[ir.SymRef(id="arg")],
                            )
                        ],
                    )
                ],
            ),
            ir.FunCall(
                fun=ir.SymRef(id="f"), args=[ir.FunCall(fun=foo_shift, args=[ir.SymRef(id="arg")])]
            ),
        ),
        (
            # can_deref(lift(f)(args...)) -> and(can_deref(arg[0]), and(can_deref(arg[1]), ...))
            ir.FunCall(
                fun=ir.SymRef(id="can_deref"),
                args=[
                    ir.FunCall(
                        fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="f")]),
                        args=[ir.SymRef(id="arg1"), ir.SymRef(id="arg2")],
                    )
                ],
            ),
            ir.FunCall(
                fun=ir.SymRef(id="and_"),
                args=[
                    ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[ir.SymRef(id="arg1")]),
                    ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[ir.SymRef(id="arg2")]),
                ],
            ),
        ),
        (
            # can_deref(shift(...)(lift(f)(args...)) -> and(can_deref(shift(...)(arg[0])), and(can_deref(shift(...)(arg[1])), ...))
            ir.FunCall(
                fun=ir.SymRef(id="can_deref"),
                args=[
                    ir.FunCall(
                        fun=foo_shift,
                        args=[
                            ir.FunCall(
                                fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="f")]),
                                args=[ir.SymRef(id="arg1"), ir.SymRef(id="arg2")],
                            )
                        ],
                    )
                ],
            ),
            ir.FunCall(
                fun=ir.SymRef(id="and_"),
                args=[
                    ir.FunCall(
                        fun=ir.SymRef(id="can_deref"),
                        args=[ir.FunCall(fun=foo_shift, args=[ir.SymRef(id="arg1")])],
                    ),
                    ir.FunCall(
                        fun=ir.SymRef(id="can_deref"),
                        args=[ir.FunCall(fun=foo_shift, args=[ir.SymRef(id="arg2")])],
                    ),
                ],
            ),
        ),
    ]


@pytest.mark.parametrize("testee, expected", inline_lift_test_data())
def test_deref_lift(testee, expected):
    result = InlineLifts().visit(testee)
    assert result == expected
