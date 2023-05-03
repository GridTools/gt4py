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

import contextlib
from typing import Optional

import numpy as np
import pytest

from gt4py.next.iterator import embedded


@contextlib.contextmanager
def column_range(column_range: Optional[range]) -> None:
    token = embedded.column_range.set(None)
    yield
    embedded.column_range.reset(token)


def test_column_ufunc():
    with column_range(None):
        a = embedded.Column(1, np.asarray(range(0, 3)))
        b = embedded.Column(1, np.asarray(range(3, 6)))

        res = a + b
        assert isinstance(res, embedded.Column)
        assert np.array_equal(res.data, a.data + b.data)
        assert res.kstart == 1


def test_column_ufunc_with_scalar():
    with column_range(None):
        a = embedded.Column(1, np.asarray(range(0, 3)))
        res = 1.0 + a
        assert isinstance(res, embedded.Column)
        assert np.array_equal(res.data, a.data + 1.0)
        assert res.kstart == 1


def test_column_ufunc_wrong_kstart():
    with column_range(None):
        a = embedded.Column(1, np.asarray(range(0, 3)))
        wrong_kstart = embedded.Column(2, np.asarray(range(3, 6)))

        with pytest.raises(ValueError):
            a + wrong_kstart


def test_column_ufunc_wrong_shape():
    with column_range(None):
        a = embedded.Column(1, np.asarray(range(0, 3)))
        wrong_shape = embedded.Column(1, np.asarray([1, 2]))

        with pytest.raises(ValueError):
            a + wrong_shape


def test_column_array_function():
    with column_range(None):
        cond = embedded.Column(1, np.asarray([True, False]))
        a = embedded.Column(1, np.asarray([1, 1]))
        b = embedded.Column(1, np.asarray([2, 2]))

        res = np.where(cond, a, b)
        ref = np.asarray([1, 2])

        assert isinstance(res, embedded.Column)
        assert np.array_equal(res.data, ref)
        assert res.kstart == 1


def test_column_array_function_with_scalar():
    with column_range(None):
        cond = embedded.Column(1, np.asarray([True, False]))
        a = 1
        b = embedded.Column(1, np.asarray([2, 2]))

        res = np.where(cond, a, b)
        ref = np.asarray([1, 2])

        assert isinstance(res, embedded.Column)
        assert np.array_equal(res.data, ref)
        assert res.kstart == 1


def test_column_array_function_wrong_kstart():
    with column_range(None):
        cond = embedded.Column(1, np.asarray([True, False]))
        wrong_kstart = embedded.Column(2, np.asarray([1, 1]))
        b = embedded.Column(1, np.asarray([2, 2]))

        with pytest.raises(ValueError):
            np.where(cond, wrong_kstart, b)


def test_column_array_function_wrong_shape():
    with column_range(None):
        cond = embedded.Column(1, np.asarray([True, False]))
        wrong_shape = embedded.Column(2, np.asarray([1, 1, 1]))
        b = embedded.Column(1, np.asarray([2, 2]))

        with pytest.raises(ValueError):
            np.where(cond, wrong_shape, b)
