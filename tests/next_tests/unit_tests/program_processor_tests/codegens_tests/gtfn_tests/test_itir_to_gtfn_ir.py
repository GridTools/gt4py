# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
        grid_type=gtx.GridType.CARTESIAN, offset_provider_type={}, column_axis=None
    ).visit(testee)

    assert expected == actual


def test_unapplied_funcall_to_function_object():
    testee = itir.SymRef(id="plus")
    expected = gtfn_ir.SymRef(id="plus")

    actual = it2gtfn.GTFN_lowering(
        grid_type=gtx.GridType.CARTESIAN, offset_provider_type={}, column_axis=None
    ).visit(testee)

    assert expected == actual


def test_get_domains():
    domain = im.call("cartesian_domain")(im.named_range(itir.AxisLiteral(value="D"), 1, 2))
    testee = itir.Program(
        id="foo",
        function_definitions=[],
        params=[itir.Sym(id="bar")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop("deref")(),
                domain=domain,
                target=itir.SymRef(id="bar"),
            )
        ],
    )

    result = list(it2gtfn._get_domains(testee.body))
    assert result == [domain]
