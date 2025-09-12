# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py import next as gtx
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im, misc
from gt4py.next.iterator.transforms import inline_lambdas
from gt4py.next.type_system import type_specifications as ts


@pytest.mark.parametrize(
    "value,expected",
    [
        (itir.Literal(value="True", type=ts.ScalarType(kind=ts.ScalarKind.BOOL)), True),
        (itir.Literal(value="False", type=ts.ScalarType(kind=ts.ScalarKind.BOOL)), False),
        (itir.Literal(value="1", type=ts.ScalarType(kind=ts.ScalarKind.INT8)), gtx.int8(1)),
        (itir.Literal(value="1", type=ts.ScalarType(kind=ts.ScalarKind.UINT8)), gtx.uint8(1)),
        (itir.Literal(value="1", type=ts.ScalarType(kind=ts.ScalarKind.INT16)), gtx.int16(1)),
        (itir.Literal(value="1", type=ts.ScalarType(kind=ts.ScalarKind.UINT16)), gtx.uint16(1)),
        (itir.Literal(value="1", type=ts.ScalarType(kind=ts.ScalarKind.INT32)), gtx.int32(1)),
        (itir.Literal(value="1", type=ts.ScalarType(kind=ts.ScalarKind.UINT32)), gtx.uint32(1)),
        (itir.Literal(value="1", type=ts.ScalarType(kind=ts.ScalarKind.INT64)), gtx.int64(1)),
        (itir.Literal(value="1", type=ts.ScalarType(kind=ts.ScalarKind.UINT64)), gtx.uint64(1)),
        (
            itir.Literal(value="0.1", type=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)),
            gtx.float32("0.1"),
        ),
        (
            itir.Literal(value="0.1", type=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
            gtx.float64("0.1"),
        ),
        (
            itir.Literal(value="1", type=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
            gtx.float64("1.0"),
        ),
    ],
    ids=lambda param: f"Literal[{param.value}, {param.type}]"
    if isinstance(param, itir.Literal)
    else str(param),
)
def test_value_from_literal(value, expected):
    result = misc.value_from_literal(value)

    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(
    "expr,  expected_expr",
    [
        (
            im.let("a", "b")(im.make_tuple(im.tuple_get(0, "a"), im.tuple_get(42, "a"))),
            im.ref("b"),
        ),
        (
            im.tuple_get(1, im.ref("a")),
            im.ref("a"),
        ),
        (
            im.make_tuple(im.tuple_get(0, im.ref("a")), im.tuple_get(42, im.ref("a"))),
            im.ref("a"),
        ),
        (
            im.make_tuple(
                im.tuple_get(1, im.ref("a")),
                im.make_tuple(im.tuple_get(2, im.ref("a")), im.tuple_get(0, im.ref("a"))),
            ),
            im.ref("a"),
        ),
        (
            im.tuple_get(3, im.tuple_get(0, im.ref("a"))),
            im.ref("a"),
        ),
        (
            im.tuple_get(
                1,
                im.as_fieldop("scan")(
                    im.lambda_("state", "val")(
                        im.make_tuple("val", im.plus(im.tuple_get(0, "state"), "val"))
                    )
                ),
            ),
            im.as_fieldop("scan")(
                im.lambda_("state", "val")(
                    im.make_tuple("val", im.plus(im.tuple_get(0, "state"), "val"))
                )
            ),
        ),
        (
            im.plus(im.ref("a"), im.ref("b")),
            im.plus(im.ref("a"), im.ref("b")),
        ),
        (
            im.call("as_fieldop")(im.ref("a")),
            im.call("as_fieldop")(im.ref("a")),
        ),
    ],
)
def test_extract_projector(expr, expected_expr):
    actual_projector, actual_expr = misc.extract_projector(expr)
    assert actual_expr == expected_expr

    if expr == expected_expr:
        assert actual_projector is None

    if actual_projector is not None:
        applied_projector = im.call(actual_projector)(actual_expr)

        # simplify original expression and applied projector for comparison
        applied_projector = inline_lambdas.InlineLambdas.apply(applied_projector)
        inlined_expr = inline_lambdas.InlineLambdas.apply(expr)
        assert applied_projector == inlined_expr
