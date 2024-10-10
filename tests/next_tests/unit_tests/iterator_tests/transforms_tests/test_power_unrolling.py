# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.eve import SymbolRef
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.power_unrolling import PowerUnrolling


def test_power_unrolling_zero():
    pytest.xfail(
        "Not implementeds we don't have an easy way to determine the type of the one literal (type inference is to expensive)."
    )
    testee = im.call("power")("x", 0)
    expected = im.literal_from_value(1)

    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_one():
    testee = im.call("power")("x", 1)
    expected = ir.SymRef(id=SymbolRef("x"))

    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_two():
    testee = im.call("power")("x", 2)
    expected = im.multiplies_("x", "x")

    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_two_x_plus_two():
    testee = im.call("power")(im.plus("x", 2), 2)
    expected = im.let("power_1", im.plus("x", 2))(
        im.let("power_2", im.multiplies_("power_1", "power_1"))("power_2")
    )

    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_two_x_plus_one_times_three():
    testee = im.call("power")(im.multiplies_(im.plus("x", 1), 3), 2)
    expected = im.let("power_1", im.multiplies_(im.plus("x", 1), 3))(
        im.let("power_2", im.multiplies_("power_1", "power_1"))("power_2")
    )

    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_three():
    testee = im.call("power")("x", 3)
    expected = im.multiplies_(im.multiplies_("x", "x"), "x")

    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_four():
    testee = im.call("power")("x", 4)
    expected = im.let("power_2", im.multiplies_("x", "x"))(im.multiplies_("power_2", "power_2"))

    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_five():
    testee = im.call("power")("x", 5)
    tmp2 = im.multiplies_("x", "x")
    expected = im.multiplies_(im.multiplies_(tmp2, tmp2), "x")
    expected = im.let("power_2", im.multiplies_("x", "x"))(
        im.multiplies_(im.multiplies_("power_2", "power_2"), "x")
    )

    actual = PowerUnrolling.apply(testee)
    assert actual == expected


def test_power_unrolling_seven():
    testee = im.call("power")("x", 7)
    expected = im.call("power")("x", 7)

    actual = PowerUnrolling.apply(testee, max_unroll=5)
    assert actual == expected


def test_power_unrolling_seven_unrolled():
    testee = im.call("power")("x", 7)
    expected = im.let("power_2", im.multiplies_("x", "x"))(
        im.multiplies_(im.multiplies_(im.multiplies_("power_2", "power_2"), "power_2"), "x")
    )

    actual = PowerUnrolling.apply(testee, max_unroll=7)
    assert actual == expected


def test_power_unrolling_seven_x_plus_one_unrolled():
    testee = im.call("power")(im.plus("x", 1), 7)
    expected = im.let("power_1", im.plus("x", 1))(
        im.let("power_2", im.multiplies_("power_1", "power_1"))(
            im.let("power_4", im.multiplies_("power_2", "power_2"))(
                im.multiplies_(im.multiplies_("power_4", "power_2"), "power_1")
            )
        )
    )

    actual = PowerUnrolling.apply(testee, max_unroll=7)
    assert actual == expected


def test_power_unrolling_eight():
    testee = im.call("power")("x", 8)
    expected = im.call("power")("x", 8)

    actual = PowerUnrolling.apply(testee, max_unroll=5)
    assert actual == expected


def test_power_unrolling_eight_unrolled():
    testee = im.call("power")("x", 8)
    expected = im.let("power_2", im.multiplies_("x", "x"))(
        im.let("power_4", im.multiplies_("power_2", "power_2"))(
            im.multiplies_("power_4", "power_4")
        )
    )

    actual = PowerUnrolling.apply(testee, max_unroll=8)
    assert actual == expected


def test_power_unrolling_eight_x_plus_one_unrolled():
    testee = im.call("power")(im.plus("x", 1), 8)
    expected = im.let("power_1", im.plus("x", 1))(
        im.let("power_2", im.multiplies_("power_1", "power_1"))(
            im.let("power_4", im.multiplies_("power_2", "power_2"))(
                im.let("power_8", im.multiplies_("power_4", "power_4"))("power_8")
            )
        )
    )

    actual = PowerUnrolling.apply(testee, max_unroll=8)
    assert actual == expected
