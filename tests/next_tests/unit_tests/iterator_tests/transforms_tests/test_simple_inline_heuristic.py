# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.simple_inline_heuristic import is_eligible_for_inlining
from gt4py.next.iterator.ir_utils import ir_makers as im


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
            im.literal("True", "bool"),
            im.literal("0.0", "float64"),
        ],
    )


@pytest.mark.parametrize("is_scan_context", [True, False])
def test_trivial(is_scan_context):
    # `is_scan_context == True` covers this pattern:
    # `↑(scan(λ(acc, it) → acc + ·↑(deref)(it)))(...)` where the inner lift should not be inlined.
    expected = not is_scan_context
    testee = ir.FunCall(
        fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="deref")]),
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
                fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="deref")]),
                args=[ir.SymRef(id="x")],
            )
        ],
    )
    assert not is_eligible_for_inlining(testee, False)
