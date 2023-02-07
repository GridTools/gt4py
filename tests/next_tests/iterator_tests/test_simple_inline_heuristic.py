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

import pytest

from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.simple_inline_heuristic import is_eligible_for_inlining


@pytest.fixture
def scan():
    return ir.FunCall(
        fun=ir.SymRef(id="scan"),
        args=[
            ir.Lambda(
                params=[ir.Sym(id="acc"), ir.Sym(id="x")],
                expr=ir.FunCall(
                    fun=ir.SymRef(id="plus"),
                    args=[
                        ir.SymRef(id="acc"),
                        ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x")]),
                    ],
                ),
            ),
            ir.Literal(value="True", type="bool"),
            ir.Literal(value="0.0", type="float"),
        ],
    )


@pytest.mark.parametrize("is_scan_context", [True, False])
def test_trivial(is_scan_context):
    # `is_scan_context == True` covers this pattern:
    # `↑(scan(λ(acc, it) → acc + ·↑(deref)(it)))(...)` where the inner lift should not be inlined.
    expected = not is_scan_context
    testee = ir.FunCall(
        fun=ir.FunCall(
            fun=ir.SymRef(id="lift"),
            args=[ir.SymRef(id="deref")],
        ),
        args=[ir.SymRef(id="it")],
    )
    assert expected == is_eligible_for_inlining(testee, is_scan_context)


def test_scan(scan):
    testee = ir.FunCall(
        fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[scan]), args=[ir.SymRef(id="it")]
    )
    assert not is_eligible_for_inlining(testee, False)


def test_scan_with_lifted_arg(scan):
    testee = ir.FunCall(
        fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[scan]),
        args=[
            ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="lift"),
                    args=[ir.SymRef(id="deref")],
                ),
                args=[ir.SymRef(id="x")],
            )
        ],
    )
    assert not is_eligible_for_inlining(testee, False)
