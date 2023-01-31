# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from types import SimpleNamespace

import pytest

from gt4py.eve.utils import UIDs
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.unroll_reduce import UnrollReduce, _get_partial_offset_tags

from .conftest import DummyConnectivity


@pytest.fixture(params=[True, False])
def has_skip_values(request):
    return request.param


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
                fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="Dim")]),
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
                fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="Dim")]),
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
                fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="Dim")]),
                args=[ir.SymRef(id="x")],
            ),
            ir.FunCall(
                fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="Dim2")]),
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
                        ir.OffsetLiteral(value="IrrelevantDim"),
                        ir.OffsetLiteral(value="0"),
                        ir.OffsetLiteral(value="Dim"),
                    ],
                ),
                args=[ir.SymRef(id="x")],
            ),
            ir.FunCall(
                fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="Dim")]),
                args=[ir.SymRef(id="y")],
            ),
        ],
    )


@pytest.mark.parametrize(
    "reduction",
    [
        "basic_reduction",
        "reduction_with_irrelevant_full_shift",
        "reduction_with_shift_on_second_arg",
    ],
)
def test_get_partial_offsets(reduction, request):
    offset_provider = {"Dim": SimpleNamespace(max_neighbors=3, has_skip_values=False)}
    partial_offsets = _get_partial_offset_tags(request.getfixturevalue(reduction).args)

    assert set(partial_offsets) == {"Dim"}


def _expected(red, dim, max_neighbors, has_skip_values, shifted_arg=0):
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
                    args=[red.args[shifted_arg]],
                )
            ],
        )
        step_expr = ir.FunCall(fun=ir.SymRef(id="if_"), args=[can_deref, step_expr, acc])
    step_fun = ir.Lambda(params=[ir.Sym(id=acc.id), ir.Sym(id=offset.id)], expr=step_expr)

    step_app = red_init
    for i in range(max_neighbors):
        step_app = ir.FunCall(fun=step, args=[step_app, ir.OffsetLiteral(value=i)])

    return ir.FunCall(fun=ir.Lambda(params=[ir.Sym(id=step.id)], expr=step_app), args=[step_fun])


def test_basic(basic_reduction, has_skip_values):
    expected = _expected(basic_reduction, "Dim", 3, has_skip_values)

    offset_provider = {"Dim": DummyConnectivity(max_neighbors=3, has_skip_values=has_skip_values)}
    actual = UnrollReduce.apply(basic_reduction, offset_provider=offset_provider)
    assert actual == expected


def test_reduction_with_shift_on_second_arg(reduction_with_shift_on_second_arg, has_skip_values):
    expected = _expected(reduction_with_shift_on_second_arg, "Dim", 1, has_skip_values, 1)

    offset_provider = {"Dim": DummyConnectivity(max_neighbors=1, has_skip_values=has_skip_values)}
    actual = UnrollReduce.apply(reduction_with_shift_on_second_arg, offset_provider=offset_provider)
    assert actual == expected


def test_reduction_with_irrelevant_full_shift(reduction_with_irrelevant_full_shift):
    expected = _expected(reduction_with_irrelevant_full_shift, "Dim", 3, False)

    offset_provider = {
        "Dim": DummyConnectivity(max_neighbors=3, has_skip_values=False),
        "IrrelevantDim": DummyConnectivity(
            max_neighbors=1, has_skip_values=True
        ),  # different max_neighbors and skip value to trigger error
    }
    actual = UnrollReduce.apply(
        reduction_with_irrelevant_full_shift, offset_provider=offset_provider
    )
    assert actual == expected


@pytest.mark.parametrize(
    "offset_provider",
    [
        {
            "Dim": DummyConnectivity(max_neighbors=3, has_skip_values=False),
            "Dim2": DummyConnectivity(max_neighbors=2, has_skip_values=False),
        },
        {
            "Dim": DummyConnectivity(max_neighbors=3, has_skip_values=False),
            "Dim2": DummyConnectivity(max_neighbors=3, has_skip_values=True),
        },
        {
            "Dim": DummyConnectivity(max_neighbors=3, has_skip_values=False),
            "Dim2": DummyConnectivity(max_neighbors=2, has_skip_values=True),
        },
    ],
)
def test_reduction_with_incompatible_shifts(reduction_with_incompatible_shifts, offset_provider):
    offset_provider = {
        "Dim": DummyConnectivity(max_neighbors=3, has_skip_values=False),
        "Dim2": DummyConnectivity(max_neighbors=2, has_skip_values=False),
    }
    with pytest.raises(RuntimeError, match="incompatible"):
        UnrollReduce.apply(reduction_with_incompatible_shifts, offset_provider=offset_provider)
