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
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import constant_folding
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
    result = constant_folding._value_from_literal(value)

    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(
    "test_case",
    (
        # expr, simplified expr
        (im.plus(1, 1), 2),
        (im.not_(True), False),
        (im.not_(False), True),
        (im.plus(4, im.plus(7, im.minus(7, 5))), 13),
        (im.if_(True, im.plus(im.ref("a"), 2), im.minus(9, 5)), im.plus("a", 2)),
        (im.minimum("a", "a"), "a"),
        (im.maximum(1, 2), 2),
        # canonicalization
        (im.plus("a", 1), im.plus("a", 1)),
        (im.plus(1, "a"), im.plus("a", 1)),
        # nested plus
        (im.plus(im.plus("a", 1), 1), im.plus("a", 2)),
        (im.plus(1, im.plus("a", 1)), im.plus("a", 2)),
        # nested maximum
        (im.maximum(im.maximum("a", 1), 1), im.maximum("a", 1)),
        (im.maximum(im.maximum(1, "a"), 1), im.maximum("a", 1)),
        (im.maximum("a", im.maximum(1, "a")), im.maximum("a", 1)),
        (im.maximum(im.maximum(1, "a"), im.maximum(1, "a")), im.maximum("a", 1)),
        (im.maximum(im.maximum(1, "a"), im.maximum("a", 1)), im.maximum("a", 1)),
        (im.maximum(im.minimum("a", 1), "a"), im.maximum(im.minimum("a", 1), "a")),
        # maximum & plus
        (im.maximum(im.plus("a", 1), im.plus("a", 0)), im.plus("a", 1)),
        (
            im.maximum(im.plus("a", 1), im.plus(im.plus("a", 1), 0)),
            im.plus("a", 1),
        ),
        (im.maximum("a", im.plus("a", 1)), im.plus("a", 1)),
        (im.maximum("a", im.plus("a", im.literal_from_value(-1))), im.ref("a")),
        (
            im.plus("a", im.maximum(0, im.literal_from_value(-1))),
            im.ref("a"),
        ),
        # plus & minus
        (im.minus(im.plus("a", 1), im.plus(1, 1)), im.minus("a", 1)),
        (im.plus(im.minus("a", 1), 2), im.plus("a", 1)),
        (im.plus(im.minus(1, "a"), 1), im.minus(2, "a")),
        # nested plus
        (im.plus(im.plus("a", 1), im.plus(1, 1)), im.plus("a", 3)),
        (
            im.plus(im.plus("a", im.literal_from_value(-1)), im.plus("a", 3)),
            im.plus(im.minus("a", 1), im.plus("a", 3)),
        ),
        # maximum & minus
        (im.maximum(im.minus("a", 1), "a"), im.ref("a")),
        (im.maximum("a", im.minus("a", im.literal_from_value(-1))), im.plus("a", 1)),
        (
            im.maximum(im.plus("a", im.literal_from_value(-1)), 1),
            im.maximum(im.minus("a", 1), 1),
        ),
        # minimum & plus & minus
        (im.minimum(im.plus("a", 1), "a"), im.ref("a")),
        (im.minimum("a", im.plus("a", im.literal_from_value(-1))), im.minus("a", 1)),
        (im.minimum(im.minus("a", 1), "a"), im.minus("a", 1)),
        (im.minimum("a", im.minus("a", im.literal_from_value(-1))), im.ref("a")),
        # nested maximum
        (im.maximum("a", im.maximum("b", "a")), im.maximum("b", "a")),
        # maximum & plus on complicated expr (tuple_get)
        (
            im.maximum(
                im.plus(im.tuple_get(1, "a"), 1),
                im.maximum(im.tuple_get(1, "a"), im.plus(im.tuple_get(1, "a"), 1)),
            ),
            im.plus(im.tuple_get(1, "a"), 1),
        ),
        # nested maximum & plus
        (
            im.maximum(im.maximum(im.plus(1, "a"), 1), im.plus(1, "a")),
            im.maximum(im.plus("a", 1), 1),
        ),
        # sanity check that no strange things happen
        # complex tests
        (
            # 1 - max(max(1, max(1, sym), min(1, sym), sym), 1 + (min(-1, 2) + max(-1, 1 - sym)))
            im.minus(
                1,
                im.maximum(
                    im.maximum(
                        im.maximum(1, im.maximum(1, "a")),
                        im.maximum(im.maximum(1, "a"), "a"),
                    ),
                    im.plus(
                        1,
                        im.plus(
                            im.minimum(im.literal_from_value(-1), 2),
                            im.maximum(im.literal_from_value(-1), im.minus(1, "a")),
                        ),
                    ),
                ),
            ),
            # 1 - maximum(maximum(sym, 1), maximum(1 - sym, -1))
            im.minus(
                1,
                im.maximum(
                    im.maximum("a", 1),
                    im.maximum(im.minus(1, "a"), im.literal_from_value(-1)),
                ),
            ),
        ),
        (
            # maximum(sym, 1 + sym) + (maximum(1, maximum(1, sym)) + (sym - 1 + (1 + (sym + 1) + 1))) - 2
            im.minus(
                im.plus(
                    im.maximum("a", im.plus(1, "a")),
                    im.plus(
                        im.maximum(1, im.maximum(1, "a")),
                        im.plus(im.minus("a", 1), im.plus(im.plus(1, im.plus("a", 1)), 1)),
                    ),
                ),
                2,
            ),
            # sym + 1 + (maximum(sym, 1) + (sym - 1 + (sym + 3))) - 2
            im.minus(
                im.plus(
                    im.plus("a", 1),
                    im.plus(
                        im.maximum("a", 1),
                        im.plus(im.minus("a", 1), im.plus("a", 3)),
                    ),
                ),
                2,
            ),
        ),
        (
            # minimum(1 - sym, 1 + sym) + (maximum(maximum(1 - sym, 1 + sym), 1 - sym) + maximum(1 - sym, 1 - sym))
            im.plus(
                im.minimum(im.minus(1, "a"), im.plus(1, "a")),
                im.plus(
                    im.maximum(im.maximum(im.minus(1, "a"), im.plus(1, "a")), im.minus(1, "a")),
                    im.maximum(im.minus(1, "a"), im.minus(1, "a")),
                ),
            ),
            # minimum(1 - sym, sym + 1) + (maximum(1 - sym, sym + 1) + (1 - sym))
            im.plus(
                im.minimum(im.minus(1, "a"), im.plus("a", 1)),
                im.plus(im.maximum(im.minus(1, "a"), im.plus("a", 1)), im.minus(1, "a")),
            ),
        ),
    ),
    ids=lambda x: str(x[0]),
)
def test_constant_folding(test_case):
    testee, expected = test_case
    actual = constant_folding.ConstantFolding.apply(testee)
    assert actual == im.ensure_expr(expected)
