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
from gt4py.next.iterator.transforms.collapse_list_get import CollapseListGet


def _list_get(index: ir.Expr, lst: ir.Expr) -> ir.FunCall:
    return ir.FunCall(fun=ir.SymRef(id="list_get"), args=[index, lst])


def _neighbors(offset: ir.Expr, it: ir.Expr) -> ir.FunCall:
    return ir.FunCall(fun=ir.SymRef(id="neighbors"), args=[offset, it])


def test_list_get_neighbors():
    testee = _list_get(
        ir.Literal(value="42", type="int32"),
        _neighbors(ir.OffsetLiteral(value="foo"), ir.SymRef(id="bar")),
    )

    expected = ir.FunCall(
        fun=ir.SymRef(id="deref"),
        args=[
            ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="shift"),
                    args=[
                        ir.OffsetLiteral(value="foo"),
                        ir.OffsetLiteral(value=42),
                    ],
                ),
                args=[ir.SymRef(id="bar")],
            )
        ],
    )

    actual = CollapseListGet().visit(testee)
    assert expected == actual


def test_list_get_make_const_list():
    testee = _list_get(
        ir.Literal(value="42", type="int32"),
        ir.FunCall(
            fun=ir.SymRef(id="make_const_list"), args=[ir.Literal(value="3.14", type="float64")]
        ),
    )

    expected = ir.Literal(value="3.14", type="float64")

    actual = CollapseListGet().visit(testee)
    assert expected == actual
