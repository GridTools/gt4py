# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.scan_eta_reduction import ScanEtaReduction
from gt4py.next.iterator.ir_utils import ir_makers as im


def _make_scan(*args: list[str]):
    return ir.FunCall(
        fun=ir.SymRef(id="scan"),
        args=[
            ir.Lambda(
                params=[ir.Sym(id="state")] + [ir.Sym(id=f"{arg}") for arg in args],
                expr=ir.SymRef(id="foo"),
            ),
            im.literal("0.0", "float64"),
            im.literal("True", "bool"),
        ],
    )


def test_scan_eta_reduction():
    testee = ir.Lambda(
        params=[ir.Sym(id="x"), ir.Sym(id="y")],
        expr=ir.FunCall(
            fun=_make_scan("param_y", "param_x"), args=[ir.SymRef(id="y"), ir.SymRef(id="x")]
        ),
    )
    expected = _make_scan("param_x", "param_y")
    actual = ScanEtaReduction().visit(testee)
    assert actual == expected
