# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.inline_center_deref_lift_vars import InlineCenterDerefLiftVars


def wrap_in_fencil(expr: itir.Expr) -> itir.FencilDefinition:
    return itir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[im.sym("d"), im.sym("inp"), im.sym("out")],
        closures=[
            itir.StencilClosure(
                domain=im.call("cartesian_domain")(),
                stencil=im.lambda_("it")(expr),
                output=im.ref("out"),
                inputs=[im.ref("inp")],
            )
        ],
    )


def unwrap_from_fencil(fencil: itir.FencilDefinition) -> itir.Expr:
    return fencil.closures[0].stencil.expr


def test_simple():
    testee = im.let("var", im.lift("deref")("it"))(im.deref("var"))
    expected = "(λ(_icdlv_1) → ·(↑(λ() → _icdlv_1))())(·it)"

    actual = unwrap_from_fencil(InlineCenterDerefLiftVars.apply(wrap_in_fencil(testee)))
    assert str(actual) == expected


def test_double_deref():
    testee = im.let("var", im.lift("deref")("it"))(im.plus(im.deref("var"), im.deref("var")))
    expected = "(λ(_icdlv_1) → ·(↑(λ() → _icdlv_1))() + ·(↑(λ() → _icdlv_1))())(·it)"

    actual = unwrap_from_fencil(InlineCenterDerefLiftVars.apply(wrap_in_fencil(testee)))
    assert str(actual) == expected


def test_deref_at_non_center_different_pos():
    testee = im.let("var", im.lift("deref")("it"))(im.deref(im.shift("I", 1)("var")))

    actual = unwrap_from_fencil(InlineCenterDerefLiftVars.apply(wrap_in_fencil(testee)))
    assert testee == actual


def test_deref_at_multiple_pos():
    testee = im.let("var", im.lift("deref")("it"))(
        im.plus(im.deref("var"), im.deref(im.shift("I", 1)("var")))
    )

    actual = unwrap_from_fencil(InlineCenterDerefLiftVars.apply(wrap_in_fencil(testee)))
    assert testee == actual
