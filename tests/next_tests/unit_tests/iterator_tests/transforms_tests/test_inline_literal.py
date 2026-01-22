# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import next as gtx
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import inline_literal
from gt4py.next.type_system import type_specifications as ts


def test_inline_literal_fieldop():
    IDim = gtx.Dimension("IDim")
    x_ref = im.ref("x", ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)))
    testee = im.op_as_fieldop("plus")(x_ref, 1.0)
    expected = im.as_fieldop(im.lambda_("__arg0")(im.plus(im.deref("__arg0"), 1.0)))(x_ref)
    actual = inline_literal.InlineLiteral.apply(testee)
    assert actual == expected


def test_inline_literal_scan():
    IDim = gtx.Dimension("IDim")
    x_ref = im.ref("x", ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)))
    testee = im.as_fieldop(
        im.scan(
            im.lambda_("state", "inp", "val")(
                im.plus("state", im.multiplies_(im.deref("inp"), "val"))
            ),
            True,
            0.0,
        )
    )(x_ref, 2.0)
    expected = im.as_fieldop(
        im.scan(
            im.lambda_("state", "inp")(im.plus("state", im.multiplies_(im.deref("inp"), 2.0))),
            True,
            0.0,
        )
    )(x_ref)
    actual = inline_literal.InlineLiteral.apply(testee)
    assert actual == expected
