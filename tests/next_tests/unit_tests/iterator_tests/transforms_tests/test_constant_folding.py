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
        (im.maximum(im.plus("a", one), im.plus("a", im.literal_from_value(0))), im.plus("a", one)),
        (
            im.maximum(im.plus("a", one), im.plus(im.plus("a", one), im.literal_from_value(0))),
            im.plus("a", one),
        ),
        (im.maximum("a", im.plus("a", one)), im.plus("a", one)),
        (im.maximum("a", im.plus("a", im.literal_from_value(-1))), im.ref("a")),
        (
            im.plus("a", im.maximum(im.literal_from_value(0), im.literal_from_value(-1))),
            im.ref("a"),
        ),
        # plus & minus
        (im.minus(im.plus("sym", one), im.plus(one, one)), im.minus("sym", one)),
        (im.plus(im.minus("sym", one), im.literal_from_value(2)), im.plus("sym", one)),
        (im.plus(im.minus(one, "sym"), one), im.minus(im.literal_from_value(2), "sym")),
        # nested plus
        (im.plus(im.plus("sym", one), im.plus(one, one)), im.plus("sym", im.literal_from_value(3))),
        (
            im.plus(
                im.plus("sym", im.literal_from_value(-1)), im.plus("sym", im.literal_from_value(3))
            ),
            im.plus(im.minus("sym", one), im.plus("sym", im.literal_from_value(3))),
        ),
        # maximum & minus
        (im.maximum(im.minus("sym", one), "sym"), im.ref("sym")),
        (im.maximum("sym", im.minus("sym", im.literal_from_value(-1))), im.plus("sym", one)),
        (
            im.maximum(im.plus("sym", im.literal_from_value(-1)), one),
            im.maximum(im.minus("sym", one), one),
        ),
        # minimum & plus & minus
        (im.minimum(im.plus("sym", one), "sym"), im.ref("sym")),
        (im.minimum("sym", im.plus("sym", im.literal_from_value(-1))), im.minus("sym", one)),
        (im.minimum(im.minus("sym", one), "sym"), im.minus("sym", one)),
        (im.minimum("sym", im.minus("sym", im.literal_from_value(-1))), im.ref("sym")),
        # nested maximum
        (im.maximum("sym1", im.maximum("sym2", "sym1")), im.maximum("sym2", "sym1")),
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
            im.maximum(im.maximum(im.plus(1, "sym"), 1), im.plus(1, "sym")),
            im.maximum(im.plus("sym", 1), 1),
        ),
        # sanity check that no strange things happen
        # complex tests
        (
            # 1 - max(max(1, max(1, sym), min(1, sym), sym), 1 + (min(-1, 2) + max(-1, 1 - sym)))
            im.minus(
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
            ),
            # 1 - maximum(maximum(sym, 1), maximum(1 - sym, -1))
            im.minus(
                one,
                im.maximum(
                    im.maximum("sym", one),
                    im.maximum(im.minus(one, "sym"), im.literal_from_value(-1)),
                ),
            ),
        ),
        (
            # maximum(sym, 1 + sym) + (maximum(1, maximum(1, sym)) + (sym - 1 + (1 + (sym + 1) + 1))) - 2
            im.minus(
                im.plus(
                    im.maximum("sym", im.plus(one, "sym")),
                    im.plus(
                        im.maximum(one, im.maximum(one, "sym")),
                        im.plus(
                            im.minus("sym", one), im.plus(im.plus(one, im.plus("sym", one)), one)
                        ),
                    ),
                ),
                im.literal_from_value(2),
            ),
            # sym + 1 + (maximum(sym, 1) + (sym - 1 + (sym + 3))) - 2
            im.minus(
                im.plus(
                    im.plus("sym", 1),
                    im.plus(
                        im.maximum("sym", one),
                        im.plus(im.minus("sym", one), im.plus("sym", im.literal_from_value(3))),
                    ),
                ),
                im.literal_from_value(2),
            ),
        ),
        (
            # minimum(1 - sym, 1 + sym) + (maximum(maximum(1 - sym, 1 + sym), 1 - sym) + maximum(1 - sym, 1 - sym))
            im.plus(
                im.minimum(im.minus(one, "sym"), im.plus(one, "sym")),
                im.plus(
                    im.maximum(
                        im.maximum(im.minus(one, "sym"), im.plus(one, "sym")), im.minus(one, "sym")
                    ),
                    im.maximum(im.minus(one, "sym"), im.minus(one, "sym")),
                ),
            ),
            # minimum(1 - sym, sym + 1) + (maximum(1 - sym, sym + 1) + (1 - sym))
            im.plus(
                im.minimum(im.minus(one, "sym"), im.plus("sym", one)),
                im.plus(
                    im.maximum(im.minus(one, "sym"), im.plus("sym", one)), im.minus(one, "sym")
                ),
            ),
        ),
    )


@pytest.mark.parametrize("test_case", test_cases())
def test_constant_folding(test_case):
    testee, expected = test_case
    actual = ConstantFolding.apply(testee)
    assert actual == im.ensure_expr(expected)
