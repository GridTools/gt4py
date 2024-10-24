# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.prune_closure_inputs import PruneClosureInputs


def test_simple():
    testee = ir.StencilClosure(
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
        stencil=ir.Lambda(
            params=[ir.Sym(id="x"), ir.Sym(id="y"), ir.Sym(id="z")],
            expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="y")]),
        ),
        output=ir.SymRef(id="out"),
        inputs=[ir.SymRef(id="foo"), ir.SymRef(id="bar"), ir.SymRef(id="baz")],
    )
    expected = ir.StencilClosure(
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
        stencil=ir.Lambda(
            params=[ir.Sym(id="y")],
            expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="y")]),
        ),
        output=ir.SymRef(id="out"),
        inputs=[ir.SymRef(id="bar")],
    )
    actual = PruneClosureInputs().visit(testee)
    assert actual == expected


def test_shadowing():
    testee = ir.StencilClosure(
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
        stencil=ir.Lambda(
            params=[ir.Sym(id="x"), ir.Sym(id="y"), ir.Sym(id="z")],
            expr=ir.FunCall(
                fun=ir.Lambda(
                    params=[ir.Sym(id="z")],
                    expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="z")]),
                ),
                args=[ir.SymRef(id="y")],
            ),
        ),
        output=ir.SymRef(id="out"),
        inputs=[ir.SymRef(id="foo"), ir.SymRef(id="bar"), ir.SymRef(id="baz")],
    )
    expected = ir.StencilClosure(
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
        stencil=ir.Lambda(
            params=[ir.Sym(id="y")],
            expr=ir.FunCall(
                fun=ir.Lambda(
                    params=[ir.Sym(id="z")],
                    expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="z")]),
                ),
                args=[ir.SymRef(id="y")],
            ),
        ),
        output=ir.SymRef(id="out"),
        inputs=[ir.SymRef(id="bar")],
    )
    actual = PruneClosureInputs().visit(testee)
    assert actual == expected
