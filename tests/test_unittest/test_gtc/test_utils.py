# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from gtc.utils import ListTuple


@pytest.mark.parametrize(
    "valid_list_tuple,expected_lengths",
    [
        (ListTuple([]), [0]),
        (ListTuple([], []), [0, 0]),
        (ListTuple([1, 2]), [2]),
        (ListTuple([1, 2], [3, 4, 5]), [2, 3]),
    ],
)
def test_ListTuple_creation(valid_list_tuple, expected_lengths):
    assert len(valid_list_tuple) == len(expected_lengths)
    for lst, length in zip(valid_list_tuple, expected_lengths):
        assert len(lst) == length


def test_invalid_ListTuple():
    with pytest.raises(AssertionError):
        ListTuple()


@pytest.mark.parametrize(
    "list_tuple_expr,expected_list_tuple",
    [
        (ListTuple([]) + ListTuple([]), ListTuple([])),
        (ListTuple([1]) + ListTuple([2]), ListTuple([1, 2])),
        (ListTuple([], []) + ListTuple([], []), ListTuple([], [])),
        (ListTuple([1], [2]) + ListTuple([3], [4]), ListTuple([1, 3], [2, 4])),
        (
            ListTuple([1], [2, 3], [4, 5, 6]) + ListTuple([7, 8, 9], [10, 11], [12]),
            ListTuple([1, 7, 8, 9], [2, 3, 10, 11], [4, 5, 6, 12]),
        ),
    ],
)
def test_ListTuple_add(list_tuple_expr, expected_list_tuple):
    assert len(list_tuple_expr) == len(expected_list_tuple)
    for result, expected in zip(list_tuple_expr, expected_list_tuple):
        assert result == expected


@pytest.mark.parametrize(
    "list_tuple_expr,expected_list_tuple",
    [
        (ListTuple([]) + [ListTuple([]), ListTuple([])], ListTuple([])),
        (ListTuple([1]) + [ListTuple([2]), ListTuple([3])], ListTuple([1, 2, 3])),
        (ListTuple([1]) + [ListTuple([2]), [ListTuple([3])]], ListTuple([1, 2, 3])),
    ],
)
def test_ListTuple_add_list(list_tuple_expr, expected_list_tuple):
    assert len(list_tuple_expr) == len(expected_list_tuple)
    for result, expected in zip(list_tuple_expr, expected_list_tuple):
        assert result == expected


@pytest.mark.parametrize(
    "expr",
    [
        lambda: ListTuple([1]) + ListTuple([2], [3]),
        lambda: ListTuple([1]) + [ListTuple([2], [3])],
        lambda: ListTuple([1]) + 1,
    ],
)
def test_ListTuple_add_incompatible(expr):
    with pytest.raises(ValueError):
        expr()
