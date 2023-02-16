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
from gt4py.next.iterator.transforms.collect_shifts import ALL_NEIGHBORS, CollectShifts


def test_trivial():
    testee = ir.FunCall(
        fun=ir.SymRef(id="deref"),
        args=[
            ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="shift"),
                    args=[ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1)],
                ),
                args=[ir.SymRef(id="x")],
            )
        ],
    )
    expected = {"x": [(ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1))]}

    actual = dict()
    CollectShifts().visit(testee, shifts=actual)
    assert actual == expected


def test_reduce():
    testee = ir.FunCall(
        fun=ir.FunCall(
            fun=ir.SymRef(id="reduce"),
            args=[ir.SymRef(id="plus"), ir.Literal(value="0.0", type="float")],
        ),
        args=[
            ir.FunCall(
                fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="V2E")]),
                args=[ir.SymRef(id="x")],
            )
        ],
    )

    expected = {"x": [(ir.OffsetLiteral(value="V2E"), ALL_NEIGHBORS)]}

    actual = dict()
    CollectShifts().visit(testee, shifts=actual)
    assert actual == expected
