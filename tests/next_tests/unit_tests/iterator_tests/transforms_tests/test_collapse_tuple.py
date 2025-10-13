# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

from gt4py.next import common
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple
from gt4py.next.iterator.type_system import type_specifications as it_ts
from gt4py.next.type_system import type_specifications as ts


int_type = ts.ScalarType(kind=ts.ScalarKind.INT32)
Vertex = common.Dimension(value="Vertex", kind=common.DimensionKind.HORIZONTAL)


def test_simple_make_tuple_tuple_get():
    tuple_of_size_2 = im.make_tuple("first", "second")
    testee = im.make_tuple(im.tuple_get(0, tuple_of_size_2), im.tuple_get(1, tuple_of_size_2))

    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        enabled_transformations=CollapseTuple.Transformation.COLLAPSE_MAKE_TUPLE_TUPLE_GET,
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
        enabled_transformations=CollapseTuple.Transformation.COLLAPSE_MAKE_TUPLE_TUPLE_GET,
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
        enabled_transformations=CollapseTuple.Transformation.COLLAPSE_MAKE_TUPLE_TUPLE_GET,
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
        enabled_transformations=CollapseTuple.Transformation.COLLAPSE_MAKE_TUPLE_TUPLE_GET,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == testee  # did nothing


def test_incompatible_size_make_tuple_tuple_get():
    testee = im.make_tuple(im.tuple_get(0, im.make_tuple("first", "second")))
    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        enabled_transformations=CollapseTuple.Transformation.COLLAPSE_MAKE_TUPLE_TUPLE_GET,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == testee  # did nothing


def test_simple_tuple_get_make_tuple():
    expected = im.ref("bar")
    testee = im.tuple_get(1, im.make_tuple("foo", expected))
    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        enabled_transformations=CollapseTuple.Transformation.COLLAPSE_TUPLE_GET_MAKE_TUPLE,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert expected == actual


@pytest.mark.parametrize("fun", ["if_", "concat_where"])
def test_propagate_tuple_get(fun):
    testee = im.tuple_get(
        0, im.call(fun)("cond", im.make_tuple("el1", "el2"), im.make_tuple("el1", "el2"))
    )
    expected = im.call(fun)(
        "cond",
        im.tuple_get(0, im.make_tuple("el1", "el2")),
        im.tuple_get(0, im.make_tuple("el1", "el2")),
    )
    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        enabled_transformations=CollapseTuple.Transformation.PROPAGATE_TUPLE_GET,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert expected == actual


def test_propagate_tuple_get_let():
    expected = im.let(("el1", 1), ("el2", 2))(im.tuple_get(0, im.make_tuple("el1", "el2")))
    testee = im.tuple_get(0, im.let(("el1", 1), ("el2", 2))(im.make_tuple("el1", "el2")))
    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        enabled_transformations=CollapseTuple.Transformation.PROPAGATE_TUPLE_GET,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert expected == actual


def test_letify_make_tuple_elements():
    fun_type = ts.FunctionType(
        pos_only_args=[], pos_or_kw_args={}, kw_only_args={}, returns=int_type
    )
    # anything that is not trivial, works here
    el1, el2 = im.call(im.ref("foo", fun_type))(), im.call(im.ref("bar", fun_type))()
    testee = im.make_tuple(el1, el2)
    expected = im.let(("__ct_el_1", el1), ("__ct_el_2", el2))(
        im.make_tuple("__ct_el_1", "__ct_el_2")
    )

    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        enabled_transformations=CollapseTuple.Transformation.LETIFY_MAKE_TUPLE_ELEMENTS,
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
        enabled_transformations=CollapseTuple.Transformation.LETIFY_MAKE_TUPLE_ELEMENTS,
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
        enabled_transformations=CollapseTuple.Transformation.INLINE_TRIVIAL_MAKE_TUPLE,
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
        enabled_transformations=CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
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
        enabled_transformations=CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES
        | CollapseTuple.Transformation.LETIFY_MAKE_TUPLE_ELEMENTS,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_propagate_nested_let():
    testee = im.let("a", im.let("b", 1)("a_val"))("a")
    expected = im.let("b", 1)(im.let("a", "a_val")("a"))
    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        enabled_transformations=CollapseTuple.Transformation.PROPAGATE_NESTED_LET,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_propagate_nested_let_collision_between_args():
    testee = im.let(("a", im.let("c", 1)("c")), ("b", im.let("c", 2)("c")))(
        im.call("plus")("a", "b")
    )
    expected = im.let(("c", 1), ("c_", 2))(
        im.let(("a", "c"), ("b", "c_"))(im.call("plus")("a", "b"))
    )
    actual = CollapseTuple.apply(
        testee,
        remove_letified_make_tuple_elements=False,
        enabled_transformations=CollapseTuple.Transformation.PROPAGATE_NESTED_LET,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_propagate_nested_let_collision_between_args2():
    ir = im.let(("a", im.let("c", 1)("c")), ("b", "c"))(im.make_tuple("a", "b"))
    expected = im.make_tuple(1, "c")
    actual = CollapseTuple.apply(
        ir,
        enabled_transformations=CollapseTuple.Transformation.PROPAGATE_NESTED_LET,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_propagate_nested_let_collision_with_body():
    ir = im.let(("a", im.let("c", 1)("c")))(im.make_tuple("a", "c"))
    expected = im.make_tuple(1, "c")
    actual = CollapseTuple.apply(
        ir,
        enabled_transformations=CollapseTuple.Transformation.PROPAGATE_NESTED_LET,
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


def test_if_make_tuple_reorder_cps():
    testee = im.let("t", im.if_(True, im.make_tuple(1, 2), im.make_tuple(3, 4)))(
        im.make_tuple(im.tuple_get(1, "t"), im.tuple_get(0, "t"))
    )
    expected = im.if_(True, im.make_tuple(2, 1), im.make_tuple(4, 3))
    actual = CollapseTuple.apply(
        testee,
        enabled_transformations=~CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_nested_if_make_tuple_reorder_cps():
    testee = im.let(
        ("t1", im.if_(True, im.make_tuple(1, 2), im.make_tuple(3, 4))),
        ("t2", im.if_(False, im.make_tuple(5, 6), im.make_tuple(7, 8))),
    )(
        im.make_tuple(
            im.tuple_get(1, "t1"),
            im.tuple_get(0, "t1"),
            im.tuple_get(1, "t2"),
            im.tuple_get(0, "t2"),
        )
    )
    expected = im.if_(
        True,
        im.if_(False, im.make_tuple(2, 1, 6, 5), im.make_tuple(2, 1, 8, 7)),
        im.if_(False, im.make_tuple(4, 3, 6, 5), im.make_tuple(4, 3, 8, 7)),
    )
    actual = CollapseTuple.apply(
        testee,
        enabled_transformations=~CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_if_make_tuple_reorder_cps_nested():
    testee = im.let("t", im.if_(True, im.make_tuple(1, 2), im.make_tuple(3, 4)))(
        im.let("c", im.tuple_get(0, "t"))(
            im.make_tuple(im.tuple_get(1, "t"), im.tuple_get(0, "t"), "c")
        )
    )
    expected = im.if_(True, im.make_tuple(2, 1, 1), im.make_tuple(4, 3, 3))
    actual = CollapseTuple.apply(
        testee,
        enabled_transformations=~CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_if_make_tuple_reorder_cps_external():
    external_ref = im.tuple_get(0, im.ref("external", ts.TupleType(types=[int_type])))
    testee = im.let("t", im.if_(True, im.make_tuple(1, 2), im.make_tuple(3, 4)))(
        im.make_tuple(external_ref, im.tuple_get(1, "t"), im.tuple_get(0, "t"))
    )
    expected = im.if_(True, im.make_tuple(external_ref, 2, 1), im.make_tuple(external_ref, 4, 3))
    actual = CollapseTuple.apply(
        testee,
        enabled_transformations=~CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_flatten_as_fieldop_args():
    it_type = it_ts.IteratorType(
        position_dims=[],
        defined_dims=[],
        element_type=ts.TupleType(types=[int_type, int_type]),
    )
    testee = im.as_fieldop(im.lambda_(im.sym("it", it_type))(im.tuple_get(1, im.deref("it"))))(
        im.make_tuple(1, 2)
    )
    expected = im.as_fieldop(
        im.lambda_("__ct_flat_el_0_it", "__ct_flat_el_1_it")(im.deref("__ct_flat_el_1_it"))
    )(1, 2)
    actual = CollapseTuple.apply(
        testee,
        enabled_transformations=~CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_flatten_as_fieldop_args_nested():
    it_type = it_ts.IteratorType(
        position_dims=[],
        defined_dims=[],
        element_type=ts.TupleType(
            types=[
                int_type,
                ts.TupleType(types=[int_type, int_type]),
            ]
        ),
    )
    testee = im.as_fieldop(
        im.lambda_(im.sym("it", it_type))(im.tuple_get(1, im.tuple_get(1, im.deref("it"))))
    )(im.make_tuple(1, im.make_tuple(2, 3)))
    expected = im.as_fieldop(
        im.lambda_("__ct_flat_el_0_it", "__ct_flat_el_1_0_it", "__ct_flat_el_1_1_it")(
            im.deref("__ct_flat_el_1_1_it")
        )
    )(1, 2, 3)
    actual = CollapseTuple.apply(
        testee,
        enabled_transformations=~CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected


def test_flatten_as_fieldop_args_scan():
    it_type = it_ts.IteratorType(
        position_dims=[],
        defined_dims=[],
        element_type=ts.TupleType(types=[int_type, int_type]),
    )
    testee = im.as_fieldop(
        im.scan(
            im.lambda_("state", im.sym("it", it_type))(im.tuple_get(1, im.deref("it"))), True, 0
        )
    )(im.make_tuple(1, 2))
    expected = im.as_fieldop(
        im.scan(
            im.lambda_("state", "__ct_flat_el_0_it", "__ct_flat_el_1_it")(
                im.deref("__ct_flat_el_1_it")
            ),
            True,
            0,
        )
    )(1, 2)
    actual = CollapseTuple.apply(
        testee,
        enabled_transformations=~CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
        allow_undeclared_symbols=True,
        within_stencil=False,
    )
    assert actual == expected
