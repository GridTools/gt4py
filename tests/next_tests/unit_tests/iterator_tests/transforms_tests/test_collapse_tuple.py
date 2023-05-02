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

import pytest

import gt4py.next.iterator.ir_makers as im
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple


def _tuple_get(i: int, t: ir.Expr) -> ir.Expr:
    return im.tuple_get(im.literal(str(i), "int"), t)


def _tup_of_size_2(first=ir.SymRef(id="first_elem"), second=ir.SymRef(id="second_elem")) -> ir.Expr:
    return im.make_tuple(first, second)


def test_simple_make_tuple_tuple_get():
    t = _tup_of_size_2()
    testee = im.make_tuple(_tuple_get(0, t), _tuple_get(1, t))

    actual = CollapseTuple.apply(testee, collapse_tuple_get_make_tuple=False)

    expected = _tup_of_size_2()
    assert actual == expected


def test_nested_make_tuple_tuple_get():
    t = im.call(im.lambda_()(_tup_of_size_2()))()
    testee = im.make_tuple(_tuple_get(0, t), _tuple_get(1, t))

    actual = CollapseTuple.apply(testee, collapse_tuple_get_make_tuple=False)

    assert actual == t


def test_different_tuples_make_tuple_tuple_get():
    t0 = im.make_tuple("foo0", "bar0")
    t1 = im.make_tuple("foo1", "bar1")
    testee = im.make_tuple(_tuple_get(0, t0), _tuple_get(1, t1))

    actual = CollapseTuple.apply(testee, collapse_tuple_get_make_tuple=False)

    assert actual == testee  # did nothing


def test_incompatible_order_make_tuple_tuple_get():
    t = _tup_of_size_2()
    testee = ir.FunCall(
        fun=ir.SymRef(id="make_tuple"),
        args=[_tuple_get(1, t), _tuple_get(0, t)],
    )
    actual = CollapseTuple.apply(testee, collapse_tuple_get_make_tuple=False)
    assert actual == testee  # did nothing


def test_incompatible_size_make_tuple_tuple_get():
    testee = ir.FunCall(fun=ir.SymRef(id="make_tuple"), args=[_tuple_get(0, _tup_of_size_2())])
    actual = CollapseTuple.apply(testee, collapse_tuple_get_make_tuple=False)
    assert actual == testee  # did nothing


def test_merged_with_smaller_outer_size_make_tuple_tuple_get():
    testee = ir.FunCall(fun=ir.SymRef(id="make_tuple"), args=[_tuple_get(0, _tup_of_size_2())])
    actual = CollapseTuple.apply(testee, ignore_tuple_size=True)
    assert actual == _tup_of_size_2()


def test_simple_tuple_get_make_tuple():
    expected = ir.SymRef(id="bar")
    testee = _tuple_get(1, _tup_of_size_2(ir.SymRef(id="foo"), expected))
    actual = CollapseTuple.apply(testee, collapse_make_tuple_tuple_get=False)
    assert expected == actual


def test_oob_tuple_get_make_tuple():
    with pytest.raises(IndexError, match=r"CollapseTuple:.*out of bounds.*tuple of size 2"):
        CollapseTuple.apply(_tuple_get(2, _tup_of_size_2()), collapse_make_tuple_tuple_get=False)
