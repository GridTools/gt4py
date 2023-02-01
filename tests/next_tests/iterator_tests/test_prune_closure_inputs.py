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
