# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

import numpy as np
import pytest

from gt4py.next.iterator.embedded import Column


def test_column_ufunc():
    a = Column(1, np.asarray(range(0, 3)))
    b = Column(1, np.asarray(range(3, 6)))

    res = a + b
    assert isinstance(res, Column)
    assert np.array_equal(res.data, a.data + b.data)
    assert res.kstart == 1


def test_column_ufunc_with_scalar():
    a = Column(1, np.asarray(range(0, 3)))
    res = 1.0 + a
    assert isinstance(res, Column)
    assert np.array_equal(res.data, a.data + 1.0)
    assert res.kstart == 1


def test_column_ufunc_wrong_kstart():
    a = Column(1, np.asarray(range(0, 3)))
    wrong_kstart = Column(2, np.asarray(range(3, 6)))

    with pytest.raises(ValueError):
        a + wrong_kstart


def test_column_ufunc_wrong_shape():
    a = Column(1, np.asarray(range(0, 3)))
    wrong_shape = Column(1, np.asarray([1, 2]))

    with pytest.raises(ValueError):
        a + wrong_shape


def test_column_array_function():
    cond = Column(1, np.asarray([True, False]))
    a = Column(1, np.asarray([1, 1]))
    b = Column(1, np.asarray([2, 2]))

    res = np.where(cond, a, b)
    ref = np.asarray([1, 2])

    assert isinstance(res, Column)
    assert np.array_equal(res.data, ref)
    assert res.kstart == 1


def test_column_array_function_with_scalar():
    cond = Column(1, np.asarray([True, False]))
    a = 1
    b = Column(1, np.asarray([2, 2]))

    res = np.where(cond, a, b)
    ref = np.asarray([1, 2])

    assert isinstance(res, Column)
    assert np.array_equal(res.data, ref)
    assert res.kstart == 1


def test_column_array_function_wrong_kstart():
    cond = Column(1, np.asarray([True, False]))
    wrong_kstart = Column(2, np.asarray([1, 1]))
    b = Column(1, np.asarray([2, 2]))

    with pytest.raises(ValueError):
        np.where(cond, wrong_kstart, b)


def test_column_array_function_wrong_shape():
    cond = Column(1, np.asarray([True, False]))
    wrong_shape = Column(2, np.asarray([1, 1, 1]))
    b = Column(1, np.asarray([2, 2]))

    with pytest.raises(ValueError):
        np.where(cond, wrong_shape, b)
