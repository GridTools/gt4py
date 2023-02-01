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


def _make_scan(*args: list[str]):
    return ir.FunCall(
        fun=ir.SymRef(id="scan"),
        args=[
            ir.Lambda(
                params=[ir.Sym(id="state")] + [ir.Sym(id=f"{arg}") for arg in args],
                expr=ir.SymRef(id="foo"),
            ),
            ir.Literal(value="0.0", type="float"),
            ir.Literal(value="True", type="bool"),
        ],
    )


def test_extended_eta_reduction_for_scan():
    testee = ir.Lambda(
        params=[ir.Sym(id="x"), ir.Sym(id="y")],
        expr=ir.FunCall(
            fun=_make_scan("param_y", "param_x"),
            args=[ir.SymRef(id="y"), ir.SymRef(id="x")],
        ),
    )
    expected = _make_scan("param_x", "param_y")
    actual = EtaReduction().visit(testee)
    assert actual == expected
