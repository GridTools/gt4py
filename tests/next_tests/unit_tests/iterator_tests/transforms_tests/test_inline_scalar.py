# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.iterator.transforms import inline_scalar
from gt4py.next.iterator.ir_utils import ir_makers as im

TDim = common.Dimension(value="TDim")
int_type = ts.ScalarType(kind=ts.ScalarKind.INT32)


def program_factory(expr: itir.Expr) -> itir.Program:
    return itir.Program(
        id="testee",
        function_definitions=[],
        params=[im.sym("out", ts.FieldType(dims=[TDim], dtype=int_type))],
        declarations=[],
        body=[
            itir.SetAt(
                expr=expr,
                target=im.ref("out"),
                domain=im.domain(common.GridType.CARTESIAN, {TDim: (0, 1)}),
            )
        ],
    )


def test_simple():
    testee = program_factory(im.let("a", 1)(im.op_as_fieldop("plus")("a", "a")))
    expected = program_factory(im.op_as_fieldop("plus")(1, 1))
    actual = inline_scalar.InlineScalar.apply(testee, offset_provider_type={})
    assert actual == expected


def test_fo_inline_only():
    scalar_expr = im.let("a", 1)(im.plus("a", "a"))
    testee = program_factory(im.as_fieldop(im.lambda_()(scalar_expr))())
    actual = inline_scalar.InlineScalar.apply(testee, offset_provider_type={})
    assert actual == testee
