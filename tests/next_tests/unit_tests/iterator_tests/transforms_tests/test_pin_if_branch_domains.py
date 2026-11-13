# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import domain_utils, ir_makers as im
from gt4py.next.iterator.transforms import infer_domain
from gt4py.next.iterator.transforms.pin_if_branch_domains import PinIfBranchDomains
from gt4py.next.type_system import type_specifications as ts


IDim = common.Dimension(value="IDim")
float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
bool_type = ts.ScalarType(kind=ts.ScalarKind.BOOL)
field_type = ts.FieldType(dims=[IDim], dtype=float_type)
domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 10)})
symbolic_domain = domain_utils.SymbolicDomain.from_expr(domain)


def _inferred(expr: itir.Expr, target_domain) -> itir.Expr:
    return infer_domain.infer_expr(expr, target_domain, offset_provider={})[0]


def test_reference_branch_is_pinned():
    computed = im.as_fieldop("deref", domain)(im.ref("b", field_type))
    testee = _inferred(
        im.if_(im.ref("cond", bool_type), im.ref("a", field_type), computed), symbolic_domain
    )

    actual = PinIfBranchDomains.apply(testee)

    expected = im.if_("cond", im.as_fieldop("deref", domain)("a"), computed)
    assert actual == expected
    assert actual.annex.domain == symbolic_domain


def test_tuple_branch_elements_are_pinned():
    testee = _inferred(
        im.if_(
            im.ref("cond", bool_type),
            im.make_tuple(im.ref("a", field_type), im.ref("b", field_type)),
            im.make_tuple(im.ref("b", field_type), im.ref("a", field_type)),
        ),
        (symbolic_domain, symbolic_domain),
    )

    actual = PinIfBranchDomains.apply(testee)

    pin = lambda name: im.as_fieldop("deref", domain)(name)  # noqa: E731
    expected = im.if_("cond", im.make_tuple(pin("a"), pin("b")), im.make_tuple(pin("b"), pin("a")))
    assert actual == expected
    assert actual.args[1].annex.domain == (symbolic_domain, symbolic_domain)


def test_scalar_and_stencil_ifs_are_not_pinned():
    stencil_if = im.as_fieldop(
        im.lambda_("c", "x", "y")(im.if_(im.deref("c"), im.deref("x"), im.deref("y"))), domain
    )(im.ref("cond", bool_type), im.ref("a", field_type), im.ref("b", field_type))
    scalar_if = im.if_(im.ref("cond", bool_type), im.ref("s", float_type), im.ref("t", float_type))
    testee = _inferred(im.make_tuple(stencil_if, scalar_if), (symbolic_domain, symbolic_domain))

    actual = PinIfBranchDomains.apply(testee)

    assert actual == testee
