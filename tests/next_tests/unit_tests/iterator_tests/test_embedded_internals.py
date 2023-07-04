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

import contextvars as cvars
import threading
from typing import Any, Callable, Optional

import numpy as np
import pytest

from gt4py.next.iterator import embedded


def _run_within_context(
    func: Callable[[], Any],
    *,
    column_range: Optional[range] = None,
    offset_provider: Optional[embedded.OffsetProvider] = None,
) -> Any:
    def wrapped_func():
        embedded.column_range_cvar.set(column_range)
        embedded.offset_provider_cvar.set(offset_provider)
        func()

    cvars.copy_context().run(wrapped_func)


def test_column_ufunc():
    def test_func():
        a = embedded.Column(1, np.asarray(range(0, 3)))
        b = embedded.Column(1, np.asarray(range(3, 6)))
        res = a + b

        assert isinstance(res, embedded.Column)
        assert np.array_equal(res.data, a.data + b.data)
        assert res.kstart == 1

    _run_within_context(test_func)

    def test_func(data_a: int, data_b: int):
        a = embedded.Column(1, data_a)
        b = embedded.Column(1, data_b)
        res = a + b

        assert isinstance(res, embedded.Column)
        assert np.array_equal(res.data, a.data + b.data)
        assert res.kstart == 1

    # Setting an invalid column_range here shouldn't affect other contexts
    embedded.column_range_cvar.set(range(2, 999))
    _run_within_context(lambda: test_func(2, 3), column_range=range(0, 3))


def test_column_ufunc_with_scalar():
    def test_func():
        a = embedded.Column(1, np.asarray(range(0, 3)))
        res = 1.0 + a
        assert isinstance(res, embedded.Column)
        assert np.array_equal(res.data, a.data + 1.0)
        assert res.kstart == 1


def test_column_ufunc_wrong_kstart():
    def test_func():
        a = embedded.Column(1, np.asarray(range(0, 3)))
        wrong_kstart = embedded.Column(2, np.asarray(range(3, 6)))

        with pytest.raises(ValueError):
            a + wrong_kstart

    _run_within_context(test_func)


def test_column_ufunc_wrong_shape():
    def test_func():
        a = embedded.Column(1, np.asarray(range(0, 3)))
        wrong_shape = embedded.Column(1, np.asarray([1, 2]))

        with pytest.raises(ValueError):
            a + wrong_shape

    _run_within_context(test_func)


def test_column_array_function():
    def test_func():
        cond = embedded.Column(1, np.asarray([True, False]))
        a = embedded.Column(1, np.asarray([1, 1]))
        b = embedded.Column(1, np.asarray([2, 2]))

        res = np.where(cond, a, b)
        ref = np.asarray([1, 2])

        assert isinstance(res, embedded.Column)
        assert np.array_equal(res.data, ref)
        assert res.kstart == 1

    _run_within_context(test_func)


def test_column_array_function_with_scalar():
    def test_func():
        cond = embedded.Column(1, np.asarray([True, False]))
        a = 1
        b = embedded.Column(1, np.asarray([2, 2]))

        res = np.where(cond, a, b)
        ref = np.asarray([1, 2])

        assert isinstance(res, embedded.Column)
        assert np.array_equal(res.data, ref)
        assert res.kstart == 1

    _run_within_context(test_func)


def test_column_array_function_wrong_kstart():
    def test_func():
        cond = embedded.Column(1, np.asarray([True, False]))
        wrong_kstart = embedded.Column(2, np.asarray([1, 1]))
        b = embedded.Column(1, np.asarray([2, 2]))

        with pytest.raises(ValueError):
            np.where(cond, wrong_kstart, b)

    _run_within_context(test_func)


def test_column_array_function_wrong_shape():
    def test_func():
        cond = embedded.Column(1, np.asarray([True, False]))
        wrong_shape = embedded.Column(2, np.asarray([1, 1, 1]))
        b = embedded.Column(1, np.asarray([2, 2]))

        with pytest.raises(ValueError):
            np.where(cond, wrong_shape, b)

    _run_within_context(test_func)
