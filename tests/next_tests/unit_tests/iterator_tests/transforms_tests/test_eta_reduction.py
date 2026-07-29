# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.eta_reduction import EtaReduction


def test_simple():
    testee = ir.Lambda(
        params=[ir.Sym(id="x")],
        expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x")]),
    )
    expected = ir.SymRef(id="deref")
    actual = EtaReduction().visit(testee)
    assert actual == expected


def test_param_referenced_in_fun():
    # `λ(x) → (λ(y) → x)(x)` must not be reduced: `x` occurs free in the called function.
    testee = ir.Lambda(
        params=[ir.Sym(id="x")],
        expr=ir.FunCall(
            fun=ir.Lambda(params=[ir.Sym(id="y")], expr=ir.SymRef(id="x")),
            args=[ir.SymRef(id="x")],
        ),
    )
    expected = testee
    actual = EtaReduction().visit(testee)
    assert actual == expected
