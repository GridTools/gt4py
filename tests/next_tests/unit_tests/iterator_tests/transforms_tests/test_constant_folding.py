# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import constant_folding


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
        # InfinityLiteral folding
        (
            im.call("maximum")(im.literal_from_value(1), itir.InfinityLiteral.POSITIVE),
            itir.InfinityLiteral.POSITIVE,
        ),
        (
            im.call("maximum")(itir.InfinityLiteral.POSITIVE, im.literal_from_value(1)),
            itir.InfinityLiteral.POSITIVE,
        ),
        (
            im.call("maximum")(im.literal_from_value(1), itir.InfinityLiteral.NEGATIVE),
            im.literal_from_value(1),
        ),
        (
            im.call("maximum")(itir.InfinityLiteral.NEGATIVE, im.literal_from_value(1)),
            im.literal_from_value(1),
        ),
        (
            im.call("minimum")(im.literal_from_value(1), itir.InfinityLiteral.POSITIVE),
            im.literal_from_value(1),
        ),
        (
            im.call("minimum")(itir.InfinityLiteral.POSITIVE, im.literal_from_value(1)),
            im.literal_from_value(1),
        ),
        (
            im.call("minimum")(im.literal_from_value(1), itir.InfinityLiteral.NEGATIVE),
            itir.InfinityLiteral.NEGATIVE,
        ),
        (
            im.call("minimum")(itir.InfinityLiteral.NEGATIVE, im.literal_from_value(1)),
            itir.InfinityLiteral.NEGATIVE,
        ),
        (
            im.call("greater")(im.literal_from_value(1), itir.InfinityLiteral.POSITIVE),
            im.literal_from_value(False),
        ),
        (
            im.call("greater")(im.literal_from_value(1), itir.InfinityLiteral.NEGATIVE),
            im.literal_from_value(True),
        ),
        (
            im.call("less")(im.literal_from_value(1), itir.InfinityLiteral.POSITIVE),
            im.literal_from_value(True),
        ),
        (
            im.call("less")(im.literal_from_value(1), itir.InfinityLiteral.NEGATIVE),
            im.literal_from_value(False),
        ),
        (
            im.call("greater")(itir.InfinityLiteral.POSITIVE, im.literal_from_value(1)),
            im.literal_from_value(True),
        ),
        (
            im.call("greater")(itir.InfinityLiteral.NEGATIVE, im.literal_from_value(1)),
            im.literal_from_value(False),
        ),
        (
            im.call("less")(itir.InfinityLiteral.POSITIVE, im.literal_from_value(1)),
            im.literal_from_value(False),
        ),
        (
            im.call("less")(itir.InfinityLiteral.NEGATIVE, im.literal_from_value(1)),
            im.literal_from_value(True),
        ),
    ),
    ids=lambda x: str(x[0]),
)
def test_constant_folding(test_case):
    testee, expected = test_case
    actual = constant_folding.ConstantFolding.apply(testee)
    assert actual == im.ensure_expr(expected)
