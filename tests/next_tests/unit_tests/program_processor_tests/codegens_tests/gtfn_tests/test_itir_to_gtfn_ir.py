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

import gt4py.next as gtx
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.codegens.gtfn import gtfn_ir, itir_to_gtfn_ir as it2gtfn


def test_funcall_to_op():
    testee = itir.FunCall(
        fun=itir.SymRef(id="plus"), args=[itir.SymRef(id="foo"), itir.SymRef(id="bar")]
    )
    expected = gtfn_ir.BinaryExpr(
        op="+", lhs=gtfn_ir.SymRef(id="foo"), rhs=gtfn_ir.SymRef(id="bar")
    )

    actual = it2gtfn.GTFN_lowering(
        grid_type=gtx.GridType.CARTESIAN, offset_provider={}, column_axis=None
    ).visit(testee)

    assert expected == actual


def test_unapplied_funcall_to_function_object():
    testee = itir.SymRef(id="plus")
    expected = gtfn_ir.SymRef(id="plus")

    actual = it2gtfn.GTFN_lowering(
        grid_type=gtx.GridType.CARTESIAN, offset_provider={}, column_axis=None
    ).visit(testee)

    assert expected == actual


def test_get_domains():
    domain = im.call("cartesian_domain")(im.call("named_range")(itir.AxisLiteral(value="D"), 1, 2))
    testee = itir.Program(
        id="foo",
        function_definitions=[],
        params=[itir.Sym(id="bar")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(im.call("as_fieldop")("deref"))(),
                domain=domain,
                target=itir.SymRef(id="bar"),
            )
        ],
    )

    result = list(it2gtfn._get_domains(testee.body))
    assert result == [domain]
