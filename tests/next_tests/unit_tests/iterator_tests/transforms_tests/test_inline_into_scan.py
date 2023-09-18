# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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
from gt4py.next.iterator.transforms.inline_into_scan import InlineIntoScan


# TODO(havogt): remove duplication with test_eta_reduction
def _make_scan(*args: list[str], scanpass_body: ir.Expr) -> ir.Expr:
    return ir.FunCall(
        fun=ir.SymRef(id="scan"),
        args=[
            ir.Lambda(
                params=[ir.Sym(id="state")] + [ir.Sym(id=f"{arg}") for arg in args],
                expr=scanpass_body,
            ),
            ir.Literal(value="0.0", type="float64"),
            ir.Literal(value="True", type="bool"),
        ],
    )


def _lift(fun: ir.Expr, *args: list[ir.Expr]) -> ir.Expr:
    return ir.FunCall(fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[fun]), args=[*args])


def _deref(expr: ir.Expr) -> ir.Expr:
    return ir.FunCall(fun=ir.SymRef(id="deref"), args=[expr])


def test_simple():
    testee = ir.Lambda(  # the lambda around is currently required because we need a symbol table to lookup symrefs of user-defined symbols in the args
        params=[ir.Sym(id="x")],
        expr=ir.FunCall(
            fun=_make_scan("foo", scanpass_body=_deref(ir.SymRef(id="foo"))),
            args=[_lift(ir.SymRef(id="deref"), ir.SymRef(id="x"))],
        ),
    )
    expected = ir.Lambda(
        params=[ir.Sym(id="x")],
        expr=ir.FunCall(
            fun=_make_scan("x", scanpass_body=_deref(ir.SymRef(id="x"))), args=[ir.SymRef(id="x")]
        ),
    )
    actual = InlineIntoScan().visit(testee)
    assert actual == expected


def test_respect_scoping():
    testee = ir.Lambda(
        params=[ir.Sym(id="x"), ir.Sym(id="unused_in_outer_scope")],
        expr=ir.FunCall(
            fun=_make_scan("foo", "bar", scanpass_body=_deref(ir.SymRef(id="foo"))),
            args=[
                _lift(ir.SymRef(id="deref"), ir.SymRef(id="x")),
                _lift(
                    ir.Lambda(
                        params=[ir.Sym(id="unused_in_outer_scope")],
                        expr=_deref(ir.SymRef(id="unused_in_outer_scope")),
                    ),
                    ir.SymRef(id="x"),
                ),
            ],
        ),
    )
    expected = ir.Lambda(
        params=[ir.Sym(id="x"), ir.Sym(id="unused_in_outer_scope")],
        expr=ir.FunCall(
            fun=_make_scan("x", scanpass_body=_deref(ir.SymRef(id="x"))), args=[ir.SymRef(id="x")]
        ),
    )
    actual = InlineIntoScan().visit(testee)
    assert actual == expected
