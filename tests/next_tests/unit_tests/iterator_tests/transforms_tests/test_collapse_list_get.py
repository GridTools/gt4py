# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.collapse_list_get import CollapseListGet
from gt4py.next.iterator.ir_utils import ir_makers as im


def _list_get(index: ir.Expr, lst: ir.Expr) -> ir.FunCall:
    return ir.FunCall(fun=ir.SymRef(id="list_get"), args=[index, lst])


def _neighbors(offset: ir.Expr, it: ir.Expr) -> ir.FunCall:
    return ir.FunCall(fun=ir.SymRef(id="neighbors"), args=[offset, it])


def test_list_get_neighbors():
    testee = _list_get(
        im.literal("42", "int32"),
        _neighbors(ir.OffsetLiteral(value="foo"), ir.SymRef(id="bar")),
    )

    expected = ir.FunCall(
        fun=ir.SymRef(id="deref"),
        args=[
            ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="shift"),
                    args=[ir.OffsetLiteral(value="foo"), ir.OffsetLiteral(value=42)],
                ),
                args=[ir.SymRef(id="bar")],
            )
        ],
    )

    actual = CollapseListGet().visit(testee)
    assert expected == actual


def test_list_get_make_const_list():
    testee = _list_get(
        im.literal("42", "int32"),
        ir.FunCall(fun=ir.SymRef(id="make_const_list"), args=[im.literal("3.14", "float64")]),
    )

    expected = im.literal("3.14", "float64")

    actual = CollapseListGet().visit(testee)
    assert expected == actual
