# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding

one = im.literal_from_value(1)


def test_constant_folding_plus():
    testee = im.plus(one, one)
    expected = im.literal_from_value(2)
    actual = ConstantFolding.apply(testee)
    assert actual == expected


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
    expected = im.plus("sym", 2)
    testee = im.if_(
        im.literal_from_value(True),
        im.plus("sym", im.literal_from_value(2)),
        im.minus(im.literal_from_value(9), im.literal_from_value(5)),
    )
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_maximum_literal():
    testee = im.maximum(one, im.literal_from_value(2))
    expected = im.literal_from_value(2)
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_minimum():
    testee = im.minimum("sym", "sym")
    expected = im.ref("sym")
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.maximum(0, 0)
    expected = im.literal_from_value(0)
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_cannonicalize_plus_funcall_symref_literal():
    testee = im.plus("sym", one)
    expected = im.plus("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.plus(one, "sym")
    expected = im.plus("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_canonicalize_minus():
    testee = im.minus("sym", one)
    expected = im.minus("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.minus(one, "sym")
    expected = im.minus(one, "sym")
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_canonicalize_fold_op_funcall_symref_literal():
    testee = im.plus(im.plus("sym", one), one)
    expected = im.plus("sym", im.literal_from_value(2))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.plus(one, im.plus("sym", one))
    expected = im.plus("sym", im.literal_from_value(2))
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_nested_maximum():
    testee = im.maximum(im.maximum("sym", one), one)
    expected = im.maximum("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.maximum(im.maximum(one, "sym"), one)
    expected = im.maximum("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.maximum("sym", im.maximum(one, "sym"))
    expected = im.maximum("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.maximum(im.maximum(one, "sym"), im.maximum(one, "sym"))
    expected = im.maximum("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.maximum(im.maximum(one, "sym"), im.maximum("sym", one))
    expected = im.maximum("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.maximum(im.minimum("sym", 1), "sym")
    expected = im.maximum(im.minimum("sym", 1), "sym")
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_maximum_plus():
    testee = im.maximum(im.plus("sym", one), im.plus("sym", im.literal_from_value(0)))
    expected = im.plus("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.maximum(im.plus("sym", one), im.plus(im.plus("sym", one), im.literal_from_value(0)))
    expected = im.plus("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.maximum("sym", im.plus("sym", one))
    expected = im.plus("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.maximum("sym", im.plus("sym", im.literal_from_value(-1)))
    expected = im.ref("sym")
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.plus("sym", im.maximum(im.literal_from_value(0), im.literal_from_value(-1)))
    expected = im.ref("sym")
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_plus_minus():
    testee = im.minus(im.plus("sym", one), im.plus(one, one))
    expected = im.minus("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.plus(im.minus("sym", one), im.literal_from_value(2))
    expected = im.plus("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.plus(im.minus(one, "sym"), one)
    expected = im.minus(im.literal_from_value(2), "sym")
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_nested_plus():
    testee = im.plus(im.plus("sym", one), im.plus(one, one))
    expected = im.plus("sym", im.literal_from_value(3))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.plus(
        im.plus("sym", im.literal_from_value(-1)), im.plus("sym", im.literal_from_value(3))
    )
    expected = im.plus(im.minus("sym", one), im.plus("sym", im.literal_from_value(3)))
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_maximum_minus():
    testee = im.maximum(im.minus("sym", one), "sym")
    expected = im.ref("sym")
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.maximum("sym", im.minus("sym", im.literal_from_value(-1)))
    expected = im.plus("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.maximum(im.plus("sym", im.literal_from_value(-1)), one)
    expected = im.maximum(im.minus("sym", one), one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_minimum_plus_minus():
    testee = im.minimum(im.plus("sym", one), "sym")
    expected = im.ref("sym")
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.minimum("sym", im.plus("sym", im.literal_from_value(-1)))
    expected = im.minus("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.minimum(im.minus("sym", one), "sym")
    expected = im.minus("sym", one)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.minimum("sym", im.minus("sym", im.literal_from_value(-1)))
    expected = im.ref("sym")
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_max_syms():
    testee = im.maximum("sym1", im.maximum("sym2", "sym1"))
    expected = im.maximum("sym2", "sym1")
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_max_tuple_get():
    testee = im.maximum(
        im.plus(im.tuple_get(1, "sym"), 1),
        im.maximum(im.tuple_get(1, "sym"), im.plus(im.tuple_get(1, "sym"), 1)),
    )
    expected = im.plus(im.tuple_get(1, "sym"), 1)
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_nested_max_plus():
    # maximum(maximum(1 + sym, 1), 1 + sym)
    testee = im.maximum(im.maximum(im.plus(one, "sym"), 1), im.plus(one, "sym"))

    # maximum(sym + 1, 1)
    expected = im.maximum(im.plus("sym", one), 1)
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_divides_float32():
    sym = im.ref("sym", "float32")
    testee = im.divides_(
        im.minus(im.literal("1", "float32"), sym), im.plus(im.literal("2", "float32"), sym)
    )
    expected = im.divides_(
        im.minus(im.literal("1", "float32"), sym), im.plus(sym, im.literal("2", "float32"))
    )
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_complex():
    # 1 - max(max(1, max(1, sym), min(1, sym), sym), 1 + (min(-1, 2) + max(-1, 1 - sym)))
    testee = im.minus(
        one,
        im.maximum(
            im.maximum(
                im.maximum(one, im.maximum(one, "sym")),
                im.maximum(im.maximum(one, "sym"), "sym"),
            ),
            im.plus(
                one,
                im.plus(
                    im.minimum(im.literal_from_value(-1), 2),
                    im.maximum(im.literal_from_value(-1), im.minus(one, "sym")),
                ),
            ),
        ),
    )
    # 1 - maximum(maximum(sym, 1), maximum(1 - sym, -1))
    expected = im.minus(
        one,
        im.maximum(
            im.maximum("sym", one),
            im.maximum(im.minus(one, "sym"), im.literal_from_value(-1)),
        ),
    )
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_complex_1():
    # maximum(sym, 1 + sym) + (maximum(1, maximum(1, sym)) + (sym - 1 + (1 + (sym + 1) + 1))) - 2
    testee = im.minus(
        im.plus(
            im.maximum("sym", im.plus(one, "sym")),
            im.plus(
                im.maximum(one, im.maximum(one, "sym")),
                im.plus(im.minus("sym", one), im.plus(im.plus(one, im.plus("sym", one)), one)),
            ),
        ),
        im.literal_from_value(2),
    )
    # sym + 1 + (maximum(sym, 1) + (sym - 1 + (sym + 3))) - 2
    expected = im.minus(
        im.plus(
            im.plus("sym", 1),
            im.plus(
                im.maximum("sym", one),
                im.plus(im.minus("sym", one), im.plus("sym", im.literal_from_value(3))),
            ),
        ),
        im.literal_from_value(2),
    )
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_complex_3():
    # minimum(1 - sym, 1 + sym) + (maximum(maximum(1 - sym, 1 + sym), 1 - sym) + maximum(1 - sym, 1 - sym))
    testee = im.plus(
        im.minimum(im.minus(one, "sym"), im.plus(one, "sym")),
        im.plus(
            im.maximum(im.maximum(im.minus(one, "sym"), im.plus(one, "sym")), im.minus(one, "sym")),
            im.maximum(im.minus(one, "sym"), im.minus(one, "sym")),
        ),
    )
    # minimum(1 - sym, sym + 1) + (maximum(1 - sym, sym + 1) + (1 - sym))
    expected = im.plus(
        im.minimum(im.minus(one, "sym"), im.plus("sym", one)),
        im.plus(im.maximum(im.minus(one, "sym"), im.plus("sym", one)), im.minus(one, "sym")),
    )
    actual = ConstantFolding.apply(testee)
    assert actual == expected
