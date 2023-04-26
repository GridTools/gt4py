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

from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple


def _tuple_get(i: int, t: ir.Expr) -> ir.Expr:
    return ir.FunCall(fun=ir.SymRef(id="tuple_get"), args=[ir.Literal(value=str(i), type="int"), t])


def _tup_of_size_2(first=ir.SymRef(id="first_elem"), second=ir.SymRef(id="second_elem")) -> ir.Expr:
    return ir.FunCall(fun=ir.SymRef(id="make_tuple"), args=[first, second])


def _nested_tup_of_size_2() -> ir.Expr:
    # nested in a lambda to
    # - let type inference compute the size of the tuple
    # - avoid that this expression wrapped in `tuple_get` can be trivially simplified
    return ir.FunCall(
        fun=ir.Lambda(
            params=[],
            expr=_tup_of_size_2(),
        ),
        args=[],
    )


def test_simple():
    testee = ir.FunCall(
        fun=ir.SymRef(id="make_tuple"),
        args=[_tuple_get(0, _nested_tup_of_size_2()), _tuple_get(1, _nested_tup_of_size_2())],
    )
    expected = _nested_tup_of_size_2()
    actual = CollapseTuple.apply(testee)
    assert actual == expected


def test_incompatible_order():
    testee = ir.FunCall(
        fun=ir.SymRef(id="make_tuple"),
        args=[_tuple_get(1, _nested_tup_of_size_2()), _tuple_get(0, _nested_tup_of_size_2())],
    )
    actual = CollapseTuple.apply(testee)
    assert actual == testee  # did nothing


def test_incompatible_size():
    testee = ir.FunCall(
        fun=ir.SymRef(id="make_tuple"), args=[_tuple_get(0, _nested_tup_of_size_2())]
    )
    actual = CollapseTuple.apply(testee)
    assert actual == testee  # did nothing


def test_merged_with_smaller_outer_size():
    testee = ir.FunCall(
        fun=ir.SymRef(id="make_tuple"), args=[_tuple_get(0, _nested_tup_of_size_2())]
    )
    actual = CollapseTuple.apply(testee, ignore_tuple_size=True)
    assert actual == _nested_tup_of_size_2()


def test_merged_tuple_get_make_tuple():
    expected = ir.SymRef(id="bar")
    testee = _tuple_get(1, _tup_of_size_2(ir.SymRef(id="foo"), expected))
    actual = CollapseTuple.apply(testee)
    assert expected == actual


def test_merged_tuple_get_make_tuple_oob():
    with pytest.raises(IndexError, match=r"CollapseTuple:.*out of bounds.*tuple of size 2"):
        CollapseTuple.apply(_tuple_get(2, _tup_of_size_2()))
