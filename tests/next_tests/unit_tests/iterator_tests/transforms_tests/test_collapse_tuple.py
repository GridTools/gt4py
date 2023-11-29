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
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple


def test_simple_make_tuple_tuple_get():
    tuple_of_size_2 = im.make_tuple("first", "second")
    testee = im.make_tuple(im.tuple_get(0, tuple_of_size_2), im.tuple_get(1, tuple_of_size_2))

    actual = CollapseTuple.apply(testee, flags=CollapseTuple.Flag.COLLAPSE_MAKE_TUPLE_TUPLE_GET)

    expected = tuple_of_size_2
    assert actual == expected


def test_nested_make_tuple_tuple_get():
    tup_of_size2_from_lambda = im.call(im.lambda_()(im.make_tuple("first", "second")))()
    testee = im.make_tuple(
        im.tuple_get(0, tup_of_size2_from_lambda), im.tuple_get(1, tup_of_size2_from_lambda)
    )

    actual = CollapseTuple.apply(testee, flags=CollapseTuple.Flag.COLLAPSE_MAKE_TUPLE_TUPLE_GET)

    assert actual == tup_of_size2_from_lambda


def test_different_tuples_make_tuple_tuple_get():
    t0 = im.make_tuple("foo0", "bar0")
    t1 = im.make_tuple("foo1", "bar1")
    testee = im.make_tuple(im.tuple_get(0, t0), im.tuple_get(1, t1))

    actual = CollapseTuple.apply(testee, flags=CollapseTuple.Flag.COLLAPSE_MAKE_TUPLE_TUPLE_GET)

    assert actual == testee  # did nothing


def test_incompatible_order_make_tuple_tuple_get():
    tuple_of_size_2 = im.make_tuple("first", "second")
    testee = im.make_tuple(im.tuple_get(1, tuple_of_size_2), im.tuple_get(0, tuple_of_size_2))
    actual = CollapseTuple.apply(testee, flags=CollapseTuple.Flag.COLLAPSE_MAKE_TUPLE_TUPLE_GET)
    assert actual == testee  # did nothing


def test_incompatible_size_make_tuple_tuple_get():
    testee = im.make_tuple(im.tuple_get(0, im.make_tuple("first", "second")))
    actual = CollapseTuple.apply(testee, flags=CollapseTuple.Flag.COLLAPSE_MAKE_TUPLE_TUPLE_GET)
    assert actual == testee  # did nothing


def test_merged_with_smaller_outer_size_make_tuple_tuple_get():
    testee = im.make_tuple(im.tuple_get(0, im.make_tuple("first", "second")))
    actual = CollapseTuple.apply(
        testee, ignore_tuple_size=True, flags=CollapseTuple.Flag.COLLAPSE_MAKE_TUPLE_TUPLE_GET
    )
    assert actual == im.make_tuple("first", "second")


def test_simple_tuple_get_make_tuple():
    expected = im.ref("bar")
    testee = im.tuple_get(1, im.make_tuple("foo", expected))
    actual = CollapseTuple.apply(testee, flags=CollapseTuple.Flag.COLLAPSE_TUPLE_GET_MAKE_TUPLE)
    assert expected == actual


def test_propagate_tuple_get():
    expected = im.let("el1", 1, "el2", 2)(im.tuple_get(0, im.make_tuple("el1", "el2")))
    testee = im.tuple_get(0, im.let("el1", 1, "el2", 2)(im.make_tuple("el1", "el2")))
    actual = CollapseTuple.apply(testee, flags=CollapseTuple.Flag.PROPAGATE_TUPLE_GET)
    assert expected == actual


def test_letify_make_tuple_elements():
    opaque_call = im.call("opaque")()
    testee = im.make_tuple(opaque_call, opaque_call)
    expected = im.let("_tuple_el_1", opaque_call, "_tuple_el_2", opaque_call)(
        im.make_tuple("_tuple_el_1", "_tuple_el_2")
    )

    actual = CollapseTuple.apply(testee, flags=CollapseTuple.Flag.LETIFY_MAKE_TUPLE_ELEMENTS)
    assert actual == expected


def test_letify_make_tuple_with_trivial_elements():
    testee = im.let("a", 1, "b", 2)(im.make_tuple("a", "b"))
    expected = testee  # did nothing

    actual = CollapseTuple.apply(testee, flags=CollapseTuple.Flag.LETIFY_MAKE_TUPLE_ELEMENTS)
    assert actual == expected


def test_inline_trivial_make_tuple():
    testee = im.let("tup", im.make_tuple("a", "b"))("tup")
    expected = im.make_tuple("a", "b")

    actual = CollapseTuple.apply(testee, flags=CollapseTuple.Flag.INLINE_TRIVIAL_MAKE_TUPLE)
    assert actual == expected


def test_propagate_to_if_on_tuples():
    testee = im.tuple_get(0, im.if_("cond", im.make_tuple(1, 2), im.make_tuple(3, 4)))
    expected = im.if_(
        "cond", im.tuple_get(0, im.make_tuple(1, 2)), im.tuple_get(0, im.make_tuple(3, 4))
    )
    actual = CollapseTuple.apply(testee, flags=CollapseTuple.Flag.PROPAGATE_TO_IF_ON_TUPLES)
    assert actual == expected


def test_propagate_to_if_on_tuples_with_let():
    testee = im.let("val", im.if_("cond", im.make_tuple(1, 2), im.make_tuple(3, 4)))(
        im.tuple_get(0, "val")
    )
    expected = im.if_(
        "cond", im.tuple_get(0, im.make_tuple(1, 2)), im.tuple_get(0, im.make_tuple(3, 4))
    )
    actual = CollapseTuple.apply(
        testee,
        flags=CollapseTuple.Flag.PROPAGATE_TO_IF_ON_TUPLES
        | CollapseTuple.Flag.LETIFY_MAKE_TUPLE_ELEMENTS
        | CollapseTuple.Flag.REMOVE_LETIFIED_MAKE_TUPLE_ELEMENTS,
    )
    assert actual == expected


def test_propagate_nested_lift():
    testee = im.let("a", im.let("b", 1)("a_val"))("a")
    expected = im.let("b", 1)(im.let("a", "a_val")("a"))
    actual = CollapseTuple.apply(testee, flags=CollapseTuple.Flag.PROPAGATE_NESTED_LET)
    assert actual == expected


def test_collapse_complicated_():
    # TODO: fuse with test_propagate_to_if_on_tuples_with_let
    testee = im.let("val", im.if_("cond", im.make_tuple(1, 2), im.make_tuple(3, 4)))(
        im.tuple_get(0, "val")
    )
    expected = im.if_("cond", 1, 3)
    actual = CollapseTuple.apply(
        testee,
        # flags=CollapseTuple.Flag.PROPAGATE_TO_IF_ON_TUPLES
    )
    assert actual == expected
