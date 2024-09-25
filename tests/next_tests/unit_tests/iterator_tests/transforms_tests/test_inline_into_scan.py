# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.inline_into_scan import InlineIntoScan
from gt4py.next.iterator.ir_utils import ir_makers as im


# TODO(havogt): remove duplication with test_eta_reduction
def _make_scan(*args: list[str], scanpass_body: ir.Expr) -> ir.Expr:
    return ir.FunCall(
        fun=ir.SymRef(id="scan"),
        args=[
            ir.Lambda(
                params=[ir.Sym(id="state")] + [ir.Sym(id=f"{arg}") for arg in args],
                expr=scanpass_body,
            ),
            im.literal("0.0", "float64"),
            im.literal("True", "bool"),
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
