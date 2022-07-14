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
            args=[ir.SymRef(id="foo"), ir.Literal(value="0.0", type="float")],
        ),
        args=[
            ir.FunCall(
                fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="dim")]),
                args=[ir.SymRef(id="x")],
            )
        ],
    )


@pytest.fixture
def reduction_with_shift_on_second_arg():
    UIDs.reset_sequence()
    return ir.FunCall(
        fun=ir.FunCall(
            fun=ir.SymRef(id="reduce"),
            args=[ir.SymRef(id="foo"), ir.Literal(value="0.0", type="float")],
        ),
        args=[
            ir.SymRef(id="x"),
            ir.FunCall(
                fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="dim")]),
                args=[ir.SymRef(id="y")],
            ),
        ],
    )


@pytest.fixture
def reduction_with_incompatible_shifts():
    UIDs.reset_sequence()
    return ir.FunCall(
        fun=ir.FunCall(
            fun=ir.SymRef(id="reduce"),
            args=[ir.SymRef(id="foo"), ir.Literal(value="0.0", type="float")],
        ),
        args=[
            ir.FunCall(
                fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="dim")]),
                args=[ir.SymRef(id="x")],
            ),
            ir.FunCall(
                fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="dim2")]),
                args=[ir.SymRef(id="y")],
            ),
        ],
    )


@pytest.fixture
def reduction_with_irrelevant_full_shift():
    UIDs.reset_sequence()
    return ir.FunCall(
        fun=ir.FunCall(
            fun=ir.SymRef(id="reduce"),
            args=[ir.SymRef(id="foo"), ir.Literal(value="0.0", type="float")],
        ),
        args=[
            ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="shift"),
                    args=[
                        ir.OffsetLiteral(value="irrelevant_dim"),
                        ir.OffsetLiteral(value="0"),
                        ir.OffsetLiteral(value="dim"),
                    ],
                ),
                args=[ir.SymRef(id="x")],
            ),
            ir.FunCall(
                fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="dim")]),
                args=[ir.SymRef(id="y")],
            ),
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


def test_reduction_with_shift_on_second_arg(reduction_with_shift_on_second_arg):
    expected = _expected(reduction_with_shift_on_second_arg, "dim", 3, False)

    offset_provider = {"dim": SimpleNamespace(max_neighbors=3, has_skip_values=False)}
    actual = UnrollReduce().visit(
        reduction_with_shift_on_second_arg, offset_provider=offset_provider
    )
    assert actual == expected


def test_reduction_with_irrelevant_full_shift(reduction_with_irrelevant_full_shift):
    expected = _expected(reduction_with_irrelevant_full_shift, "dim", 3, False)

    offset_provider = {
        "dim": SimpleNamespace(max_neighbors=3, has_skip_values=False),
        "irrelevant_dim": SimpleNamespace(
            max_neighbors=1, has_skip_values=True
        ),  # different max_neighbors and skip value to trigger error
    }
    actual = UnrollReduce().visit(
        reduction_with_irrelevant_full_shift, offset_provider=offset_provider
    )
    assert actual == expected


@pytest.mark.parametrize(
    "offset_provider",
    [
        {
            "dim": SimpleNamespace(max_neighbors=3, has_skip_values=False),
            "dim2": SimpleNamespace(max_neighbors=2, has_skip_values=False),
        },
        {
            "dim": SimpleNamespace(max_neighbors=3, has_skip_values=False),
            "dim2": SimpleNamespace(max_neighbors=3, has_skip_values=True),
        },
        {
            "dim": SimpleNamespace(max_neighbors=3, has_skip_values=False),
            "dim2": SimpleNamespace(max_neighbors=2, has_skip_values=True),
        },
    ],
)
def test_reduction_with_incompatible_shifts(reduction_with_incompatible_shifts, offset_provider):
    offset_provider = {
        "dim": SimpleNamespace(max_neighbors=3, has_skip_values=False),
        "dim2": SimpleNamespace(max_neighbors=2, has_skip_values=False),
    }
    with pytest.raises(RuntimeError, match="incompatible"):
        UnrollReduce().visit(reduction_with_incompatible_shifts, offset_provider=offset_provider)
