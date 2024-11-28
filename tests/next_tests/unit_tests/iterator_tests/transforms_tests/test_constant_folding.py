# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding


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
    expected = im.call("plus")("a", 2)
    testee = im.if_(
        im.literal_from_value(True),
        im.plus(im.ref("a"), im.literal_from_value(2)),
        im.minus(im.literal_from_value(9), im.literal_from_value(5)),
    )
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_minimum():
    testee = im.call("minimum")("a", "a")
    expected = im.ref("a")
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_literal():
    testee = im.plus(im.literal_from_value(1), im.literal_from_value(2))
    expected = im.literal_from_value(3)
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_literal_maximum():
    testee = im.call("maximum")(im.literal_from_value(1), im.literal_from_value(2))
    expected = im.literal_from_value(2)
    actual = ConstantFolding.apply(testee)
    assert actual == expected
