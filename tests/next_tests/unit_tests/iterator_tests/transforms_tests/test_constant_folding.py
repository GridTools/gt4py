# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator import ir

import pytest
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding


def test_cases():
    return (
        # expr, simplified expr
        (im.plus(1, 1), 2),
        (im.not_(True), False),
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
    )


@pytest.mark.parametrize("test_case", test_cases())
def test_constant_folding(test_case):
    testee, expected = test_case
    actual = ConstantFolding.apply(testee)
    assert actual == im.ensure_expr(expected)


# TODO: integrate into test structure above
def test_constant_folding_inf_maximum():
    testee = im.call("maximum")(im.literal_from_value(1), ir.InfinityLiteral.POSITIVE)
    expected = ir.InfinityLiteral.POSITIVE
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("maximum")(ir.InfinityLiteral.POSITIVE, im.literal_from_value(1))
    expected = ir.InfinityLiteral.POSITIVE
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("maximum")(im.literal_from_value(1), ir.InfinityLiteral.NEGATIVE)
    expected = im.literal_from_value(1)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("maximum")(ir.InfinityLiteral.NEGATIVE, im.literal_from_value(1))
    expected = im.literal_from_value(1)
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_inf_minimum():
    testee = im.call("minimum")(im.literal_from_value(1), ir.InfinityLiteral.POSITIVE)
    expected = im.literal_from_value(1)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("minimum")(ir.InfinityLiteral.POSITIVE, im.literal_from_value(1))
    expected = im.literal_from_value(1)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("minimum")(im.literal_from_value(1), ir.InfinityLiteral.NEGATIVE)
    expected = ir.InfinityLiteral.NEGATIVE
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("minimum")(ir.InfinityLiteral.NEGATIVE, im.literal_from_value(1))
    expected = ir.InfinityLiteral.NEGATIVE
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_greater_less():
    testee = im.call("greater")(im.literal_from_value(1), ir.InfinityLiteral.POSITIVE)
    expected = im.literal_from_value(False)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("greater")(im.literal_from_value(1), ir.InfinityLiteral.NEGATIVE)
    expected = im.literal_from_value(True)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("less")(im.literal_from_value(1), ir.InfinityLiteral.POSITIVE)
    expected = im.literal_from_value(True)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("less")(im.literal_from_value(1), ir.InfinityLiteral.NEGATIVE)
    expected = im.literal_from_value(False)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("greater")(ir.InfinityLiteral.POSITIVE, im.literal_from_value(1))
    expected = im.literal_from_value(True)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("greater")(ir.InfinityLiteral.NEGATIVE, im.literal_from_value(1))
    expected = im.literal_from_value(False)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("less")(ir.InfinityLiteral.POSITIVE, im.literal_from_value(1))
    expected = im.literal_from_value(False)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("less")(ir.InfinityLiteral.NEGATIVE, im.literal_from_value(1))
    expected = im.literal_from_value(True)
    actual = ConstantFolding.apply(testee)
    assert actual == expected
