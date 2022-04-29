from types import SimpleNamespace

import pytest

from functional.iterator import ir
from functional.iterator.transforms.unroll_reduce import UnrollReduce


@pytest.fixture
def basic_reduction():
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


def _shift_apply(args, offset, fun="deref"):
    return [
        ir.FunCall(
            fun=ir.SymRef(id=fun),
            args=[
                ir.FunCall(
                    fun=ir.FunCall(
                        fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value=offset)]
                    ),
                    args=[arg],
                )
            ],
        )
        for arg in args
    ]


def test_no_skip_values(basic_reduction):
    offset_provider = {"dim": SimpleNamespace(max_neighbors=3, has_skip_values=False)}
    acc = basic_reduction.fun.args[1]
    for offset in range(offset_provider["dim"].max_neighbors):
        acc = ir.FunCall(
            fun=basic_reduction.fun.args[0], args=[acc] + _shift_apply(basic_reduction.args, offset)
        )
    expected = acc
    actual = UnrollReduce().visit(basic_reduction, offset_provider=offset_provider)
    assert actual == expected


def test_skip_values(basic_reduction):
    offset_provider = {"dim": SimpleNamespace(max_neighbors=3, has_skip_values=True)}
    acc = basic_reduction.fun.args[1]
    for offset in range(offset_provider["dim"].max_neighbors):
        cond = _shift_apply(basic_reduction.args[:1], offset, fun="can_deref")[0]
        val = ir.FunCall(
            fun=basic_reduction.fun.args[0], args=[acc] + _shift_apply(basic_reduction.args, offset)
        )
        acc = ir.FunCall(fun=ir.SymRef(id="if_"), args=[cond, val, acc])
    expected = acc
    actual = UnrollReduce().visit(basic_reduction, offset_provider=offset_provider)
    assert actual == expected
