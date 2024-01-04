# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.power_unrolling import PowerUnrolling


def test_power_unrolling_zero():
    testee = im.call("power")("x", 0)
    expected = im.literal_from_value(1)

    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_one():
    testee = im.call("power")("x", 1)
    expected = "x"

    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_two():
    testee = im.call("power")("x", 2)
    expected = im.multiplies_("x", "x")

    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_three():
    testee = im.call("power")("x", 3)
    expected = im.multiplies_(im.multiplies_("x", "x"), "x")

    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_four():
    testee = im.call("power")("x", 4)
    tmp = im.multiplies_("x", "x")
    expected = im.multiplies_(tmp, tmp)

    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_five():
    testee = im.call("power")("x", 5)
    tmp2 = im.multiplies_("x", "x")
    expected = im.multiplies_(im.multiplies_(tmp2, tmp2), "x")

    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_x_plus_two():
    testee = im.call("power")(im.plus("x", 2), 2)
    tmp = im.plus("x", 2)
    expected = im.let("tmp", tmp)(im.multiplies_("tmp", "tmp"))
    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_x_plus_one_times_three():
    testee = im.call("power")(im.multiplies_(im.plus("x", 1), 3), 2)
    tmp = im.multiplies_(im.plus("x", 1), 3)
    expected = im.let("tmp", tmp)(im.multiplies_("tmp", "tmp"))
    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_seven():
    testee = im.call("power")("x", 7)
    expected = im.call("power")("x", 7)

    actual = PowerUnrolling.apply(testee, max_unroll=5)
    assert actual == expected


def test_power_unrolling_seven_unrolled():
    testee = im.call("power")("x", 7)
    tmp2 = im.multiplies_("x", "x")
    tmp4 = im.multiplies_(tmp2, tmp2)
    expected = im.multiplies_(im.multiplies_(tmp4, tmp2), "x")

    actual = PowerUnrolling.apply(testee, max_unroll=7)
    assert actual == expected


def test_power_unrolling_eight():
    testee = im.call("power")("x", 8)
    expected = im.call("power")("x", 8)

    actual = PowerUnrolling.apply(testee, max_unroll=5)
    assert actual == expected


def test_power_unrolling_eight_unrolled():
    testee = im.call("power")("x", 8)
    tmp2 = im.multiplies_("x", "x")
    tmp4 = im.multiplies_(tmp2, tmp2)
    expected = im.multiplies_(tmp4, tmp4)

    actual = PowerUnrolling.apply(testee, max_unroll=8)
    assert actual == expected
