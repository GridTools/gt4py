from types import SimpleNamespace

import pytest

from eve.utils import UIDs
from functional.iterator import ir
from functional.iterator.transforms.unroll_reduce import UnrollReduce


@pytest.fixture
def basic_reduction():
    UIDs.reset_sequence()
    return ir.FunCall(
        fun=ir.FunCall(
            fun=ir.SymRef(id="reduce"),
            args=[ir.SymRef(id="plus"), ir.Literal(value="0.0", type="float")],
        ),
        args=[
            ir.FunCall(
                fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="dim")]),
                args=[ir.SymRef(id="x")],
            )
        ],
    )


def _expected(red, dim, max_neighbors, has_skip_values):
    acc = ir.SymRef(id="_acc_1")
    offset = ir.SymRef(id="_i_2")
    step = ir.SymRef(id="_step_3")

    red_fun, red_init = red.fun.args

    shifted_args = [
        ir.FunCall(
            fun=ir.SymRef(id="deref"),
            args=[
                ir.FunCall(
                    fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=[offset]),
                    args=[arg],
                )
            ],
        )
        for arg in red.args
    ]

    step_expr = ir.FunCall(fun=red_fun, args=[acc] + shifted_args)
    if has_skip_values:
        can_deref = ir.FunCall(
            fun=ir.SymRef(id="can_deref"),
            args=[
                ir.FunCall(
                    fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=[offset]),
                    args=[red.args[0]],
                )
            ],
        )
        step_expr = ir.FunCall(fun=ir.SymRef(id="if_"), args=[can_deref, step_expr, acc])
    step_fun = ir.Lambda(params=[ir.Sym(id=acc.id), ir.Sym(id=offset.id)], expr=step_expr)

    step_app = red_init
    for i in range(max_neighbors):
        step_app = ir.FunCall(fun=step, args=[step_app, ir.OffsetLiteral(value=i)])

    return ir.FunCall(fun=ir.Lambda(params=[ir.Sym(id=step.id)], expr=step_app), args=[step_fun])


def test_no_skip_values(basic_reduction):
    expected = _expected(basic_reduction, "dim", 3, False)

    offset_provider = {"dim": SimpleNamespace(max_neighbors=3, has_skip_values=False)}
    actual = UnrollReduce().visit(basic_reduction, offset_provider=offset_provider)
    assert actual == expected


def test_skip_values(basic_reduction):
    expected = _expected(basic_reduction, "dim", 3, True)

    offset_provider = {"dim": SimpleNamespace(max_neighbors=3, has_skip_values=True)}
    actual = UnrollReduce().visit(basic_reduction, offset_provider=offset_provider)
    assert actual == expected
