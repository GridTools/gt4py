# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Callable, Optional

from gt4py import next as gtx
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import inline_dynamic_shifts
from gt4py.next.type_system import type_specifications as ts

IDim = gtx.Dimension("IDim")
field_type = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))


def test_inline_dynamic_shift_as_fieldop_arg():
    testee = im.as_fieldop(im.lambda_("a", "b")(im.deref(im.shift("IOff", im.deref("b"))("a"))))(
        im.as_fieldop("deref")("inp"), "offset_field"
    )
    expected = im.as_fieldop(
        im.lambda_("inp", "offset_field")(
            im.deref(im.shift("IOff", im.deref("offset_field"))("inp"))
        )
    )("inp", "offset_field")

    actual = inline_dynamic_shifts.InlineDynamicShifts.apply(testee)
    assert actual == expected


def test_inline_dynamic_shift_let_var():
    testee = im.let("tmp", im.as_fieldop("deref")("inp"))(
        im.as_fieldop(im.lambda_("a", "b")(im.deref(im.shift("IOff", im.deref("b"))("a"))))(
            "tmp", "offset_field"
        )
    )

    expected = im.as_fieldop(
        im.lambda_("inp", "offset_field")(
            im.deref(im.shift("IOff", im.deref("offset_field"))("inp"))
        )
    )("inp", "offset_field")

    actual = inline_dynamic_shifts.InlineDynamicShifts.apply(testee)
    assert actual == expected
