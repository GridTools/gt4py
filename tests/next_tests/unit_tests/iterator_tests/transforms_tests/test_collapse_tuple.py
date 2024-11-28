# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple
from gt4py.next.type_system import type_specifications as ts


def test_simple_make_tuple_tuple_get():
    tuple_of_size_2 = im.make_tuple("first", "second")
    testee = im.make_tuple(im.tuple_get(0, tuple_of_size_2), im.tuple_get(1, tuple_of_size_2))

    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        flags=CollapseTuple.Flag.COLLAPSE_MAKE_TUPLE_TUPLE_GET,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )

    expected = tuple_of_size_2
    assert actual == expected


def test_nested_make_tuple_tuple_get():
    tup_of_size2_from_lambda = im.call(im.lambda_()(im.make_tuple("first", "second")))()
    testee = im.make_tuple(
        im.tuple_get(0, tup_of_size2_from_lambda), im.tuple_get(1, tup_of_size2_from_lambda)
    )

    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        flags=CollapseTuple.Flag.COLLAPSE_MAKE_TUPLE_TUPLE_GET,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )

    assert actual == tup_of_size2_from_lambda


def test_different_tuples_make_tuple_tuple_get():
    t0 = im.make_tuple("foo0", "bar0")
    t1 = im.make_tuple("foo1", "bar1")
    testee = im.make_tuple(im.tuple_get(0, t0), im.tuple_get(1, t1))

    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        flags=CollapseTuple.Flag.COLLAPSE_MAKE_TUPLE_TUPLE_GET,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )

    assert actual == testee  # did nothing


def test_incompatible_order_make_tuple_tuple_get():
    tuple_of_size_2 = im.make_tuple("first", "second")
    testee = im.make_tuple(im.tuple_get(1, tuple_of_size_2), im.tuple_get(0, tuple_of_size_2))
    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        flags=CollapseTuple.Flag.COLLAPSE_MAKE_TUPLE_TUPLE_GET,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == testee  # did nothing


def test_incompatible_size_make_tuple_tuple_get():
    testee = im.make_tuple(im.tuple_get(0, im.make_tuple("first", "second")))
    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        flags=CollapseTuple.Flag.COLLAPSE_MAKE_TUPLE_TUPLE_GET,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == testee  # did nothing


def test_merged_with_smaller_outer_size_make_tuple_tuple_get():
    testee = im.make_tuple(im.tuple_get(0, im.make_tuple("first", "second")))
    actual = CollapseTuple.apply(
        testee,
        ignore_tuple_size=True,
        flags=CollapseTuple.Flag.COLLAPSE_MAKE_TUPLE_TUPLE_GET,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == im.make_tuple("first", "second")


def test_simple_tuple_get_make_tuple():
    expected = im.ref("bar")
    testee = im.tuple_get(1, im.make_tuple("foo", expected))
    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        flags=CollapseTuple.Flag.COLLAPSE_TUPLE_GET_MAKE_TUPLE,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert expected == actual


def test_propagate_tuple_get():
    expected = im.let(("el1", 1), ("el2", 2))(im.tuple_get(0, im.make_tuple("el1", "el2")))
    testee = im.tuple_get(0, im.let(("el1", 1), ("el2", 2))(im.make_tuple("el1", "el2")))
    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        flags=CollapseTuple.Flag.PROPAGATE_TUPLE_GET,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert expected == actual


def test_letify_make_tuple_elements():
    # anything that is not trivial, i.e. a SymRef, works here
    el1, el2 = im.let("foo", "foo")("foo"), im.let("bar", "bar")("bar")
    testee = im.make_tuple(el1, el2)
    expected = im.let(("_tuple_el_1", el1), ("_tuple_el_2", el2))(
        im.make_tuple("_tuple_el_1", "_tuple_el_2")
    )

    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        flags=CollapseTuple.Flag.LETIFY_MAKE_TUPLE_ELEMENTS,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_letify_make_tuple_with_trivial_elements():
    testee = im.let(("a", 1), ("b", 2))(im.make_tuple("a", "b"))
    expected = testee  # did nothing

    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        flags=CollapseTuple.Flag.LETIFY_MAKE_TUPLE_ELEMENTS,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_inline_trivial_make_tuple():
    testee = im.let("tup", im.make_tuple("a", "b"))("tup")
    expected = im.make_tuple("a", "b")

    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        flags=CollapseTuple.Flag.INLINE_TRIVIAL_MAKE_TUPLE,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_propagate_to_if_on_tuples():
    testee = im.tuple_get(
        0, im.if_(im.ref("pred", "bool"), im.make_tuple(1, 2), im.make_tuple(3, 4))
    )
    expected = im.if_(
        im.ref("pred", "bool"),
        im.tuple_get(0, im.make_tuple(1, 2)),
        im.tuple_get(0, im.make_tuple(3, 4)),
    )
    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        flags=CollapseTuple.Flag.PROPAGATE_TO_IF_ON_TUPLES,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_propagate_to_if_on_tuples_with_let():
    testee = im.let(
        "val", im.if_(im.ref("pred", "bool"), im.make_tuple(1, 2), im.make_tuple(3, 4))
    )(im.tuple_get(0, "val"))
    expected = im.if_(
        im.ref("pred"), im.tuple_get(0, im.make_tuple(1, 2)), im.tuple_get(0, im.make_tuple(3, 4))
    )
    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=True,
        flags=CollapseTuple.Flag.PROPAGATE_TO_IF_ON_TUPLES
        | CollapseTuple.Flag.LETIFY_MAKE_TUPLE_ELEMENTS,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_propagate_nested_lift():
    testee = im.let("a", im.let("b", 1)("a_val"))("a")
    expected = im.let("b", 1)(im.let("a", "a_val")("a"))
    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        flags=CollapseTuple.Flag.PROPAGATE_NESTED_LET,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_if_on_tuples_with_let():
    testee = im.let(
        "val", im.if_(im.ref("pred", "bool"), im.make_tuple(1, 2), im.make_tuple(3, 4))
    )(im.tuple_get(0, "val"))
    expected = im.if_("pred", 1, 3)
    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_tuple_get_on_untyped_ref():
    # test pass gracefully handles untyped nodes.
    testee = im.tuple_get(0, im.ref("val", ts.DeferredType(constraint=None)))

    actual = CollapseTuple.apply(testee, allow_undeclared_symbols=True, within_stencil=False)
    assert actual == testee
