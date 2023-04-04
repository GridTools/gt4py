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


def _tuple_get(i: int, t: ir.Expr):
    return ir.FunCall(fun=ir.SymRef(id="tuple_get"), args=[ir.Literal(value=str(i), type="int"), t])


@pytest.fixture
def tup_of_size_2():
    return ir.FunCall(
        fun=ir.SymRef(id="make_tuple"), args=[ir.SymRef(id="foo"), ir.SymRef(id="foo")]
    )


def test_simple(tup_of_size_2):
    testee = ir.FunCall(
        fun=ir.SymRef(id="make_tuple"),
        args=[_tuple_get(0, tup_of_size_2), _tuple_get(1, tup_of_size_2)],
    )
    expected = tup_of_size_2
    actual = CollapseTuple.apply(testee)
    assert actual == expected


def test_incompatible_order(tup_of_size_2):
    testee = ir.FunCall(
        fun=ir.SymRef(id="make_tuple"),
        args=[_tuple_get(1, tup_of_size_2), _tuple_get(0, tup_of_size_2)],
    )
    actual = CollapseTuple.apply(testee)
    assert actual == testee  # did nothing


def test_incompatible_size(tup_of_size_2):
    testee = ir.FunCall(fun=ir.SymRef(id="make_tuple"), args=[_tuple_get(0, tup_of_size_2)])
    actual = CollapseTuple.apply(testee)
    assert actual == testee  # did nothing


def test_merged_with_smaller_outer_size(tup_of_size_2):
    testee = ir.FunCall(fun=ir.SymRef(id="make_tuple"), args=[_tuple_get(0, tup_of_size_2)])
    actual = CollapseTuple.apply(testee, ignore_tuple_size=True)
    assert actual == tup_of_size_2
