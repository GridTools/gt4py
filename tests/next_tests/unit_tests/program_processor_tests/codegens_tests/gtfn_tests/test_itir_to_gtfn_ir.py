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


def _program_over_domains(domains: list[itir.FunCall]) -> itir.Program:
    return itir.Program(
        id="foo",
        function_definitions=[],
        params=[itir.Sym(id="bar")],
        declarations=[],
        body=[
            itir.SetAt(expr=im.as_fieldop("deref")(), domain=domain, target=itir.SymRef(id="bar"))
            for domain in domains
        ],
    )


def test_get_domains_preserves_ir_order():
    # The order of the domains fixes the order of the generated dimension tag declarations.
    domains = [
        im.call("cartesian_domain")(im.named_range(itir.AxisLiteral(value=name), 1, 2))
        for name in ["D", "C", "B", "A", "F", "E"]
    ]

    result = list(it2gtfn._get_domains(_program_over_domains(domains).body))

    assert result == domains


def test_get_domains_deduplicates():
    domain = im.call("cartesian_domain")(im.named_range(itir.AxisLiteral(value="D"), 1, 2))
    other = im.call("cartesian_domain")(im.named_range(itir.AxisLiteral(value="E"), 1, 2))

    result = list(it2gtfn._get_domains(_program_over_domains([domain, other, domain]).body))

    assert result == [domain, other]
