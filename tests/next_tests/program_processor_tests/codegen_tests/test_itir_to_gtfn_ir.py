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

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors.codegens.gtfn import gtfn_ir, itir_to_gtfn_ir as it2gtfn


def test_funcall_to_op():
    testee = itir.FunCall(
        fun=itir.SymRef(id="plus"),
        args=[itir.SymRef(id="foo"), itir.SymRef(id="bar")],
    )
    expected = gtfn_ir.BinaryExpr(
        op="+",
        lhs=gtfn_ir.SymRef(id="foo"),
        rhs=gtfn_ir.SymRef(id="bar"),
    )

    actual = it2gtfn.GTFN_lowering(
        grid_type=common.GridType.CARTESIAN, offset_provider={}, column_axis=None
    ).visit(testee)

    assert expected == actual


def test_unapplied_funcall_to_function_object():
    testee = itir.SymRef(id="plus")
    expected = gtfn_ir.SymRef(id="plus")

    actual = it2gtfn.GTFN_lowering(
        grid_type=common.GridType.CARTESIAN, offset_provider={}, column_axis=None
    ).visit(testee)

    assert expected == actual
