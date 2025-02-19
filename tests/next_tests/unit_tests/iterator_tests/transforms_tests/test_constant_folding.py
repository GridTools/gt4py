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


def test_constant_folding_boolean():
    testee = im.not_(im.literal_from_value(True))
    expected = im.literal_from_value(False)

    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_math_op():
    expected = im.literal_from_value(13)
    testee = im.plus(
        im.literal_from_value(4),
        im.plus(
            im.literal_from_value(7), im.minus(im.literal_from_value(7), im.literal_from_value(5))
        ),
    )
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_if():
    expected = im.plus("a", 2)
    testee = im.if_(
        im.literal_from_value(True),
        im.plus(im.ref("a"), im.literal_from_value(2)),
        im.minus(im.literal_from_value(9), im.literal_from_value(5)),
    )
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_minimum():
    testee = im.minimum("a", "a")
    expected = im.ref("a")
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_literal():
    testee = im.plus(im.literal_from_value(1), im.literal_from_value(2))
    expected = im.literal_from_value(3)
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_literal_maximum():
    testee = im.maximum(im.literal_from_value(1), im.literal_from_value(2))
    expected = im.literal_from_value(2)
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_inf_maximum():
    testee = im.call("maximum")(im.literal_from_value(1), ir.InfinityLiteral())
    expected = ir.InfinityLiteral()
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("maximum")(ir.InfinityLiteral(), im.literal_from_value(1))
    expected = ir.InfinityLiteral()
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("maximum")(im.literal_from_value(1), ir.NegInfinityLiteral())
    expected = im.literal_from_value(1)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("maximum")(ir.NegInfinityLiteral(), im.literal_from_value(1))
    expected = im.literal_from_value(1)
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_inf_minimum():
    testee = im.call("minimum")(im.literal_from_value(1), ir.InfinityLiteral())
    expected = im.literal_from_value(1)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("minimum")(ir.InfinityLiteral(), im.literal_from_value(1))
    expected = im.literal_from_value(1)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("minimum")(im.literal_from_value(1), ir.NegInfinityLiteral())
    expected = ir.NegInfinityLiteral()
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("minimum")(ir.NegInfinityLiteral(), im.literal_from_value(1))
    expected = ir.NegInfinityLiteral()
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_greater_less():
    testee = im.call("greater")(im.literal_from_value(1), ir.InfinityLiteral())
    expected = im.literal_from_value(False)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("greater")(im.literal_from_value(1), ir.NegInfinityLiteral())
    expected = im.literal_from_value(True)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("less")(im.literal_from_value(1), ir.InfinityLiteral())
    expected = im.literal_from_value(True)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("less")(im.literal_from_value(1), ir.NegInfinityLiteral())
    expected = im.literal_from_value(False)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("greater")(ir.InfinityLiteral(), im.literal_from_value(1))
    expected = im.literal_from_value(True)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("greater")(ir.NegInfinityLiteral(), im.literal_from_value(1))
    expected = im.literal_from_value(False)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("less")(ir.InfinityLiteral(), im.literal_from_value(1))
    expected = im.literal_from_value(False)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.call("less")(ir.NegInfinityLiteral(), im.literal_from_value(1))
    expected = im.literal_from_value(True)
    actual = ConstantFolding.apply(testee)
    assert actual == expected
