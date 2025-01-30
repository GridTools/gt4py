# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import contextvars as cvars
from typing import Any, Callable, Optional

import numpy as np
import pytest

from gt4py.next import common
from gt4py.next.embedded import context as embedded_context
from gt4py.next.iterator import embedded


def _run_within_context(
    func: Callable[[], Any],
    *,
    column_range: Optional[common.NamedRange] = None,
    offset_provider: Optional[embedded.OffsetProvider] = None,
) -> Any:
    def wrapped_func():
        embedded_context.closure_column_range.set(column_range)
        embedded_context.offset_provider.set(offset_provider)
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
    embedded_context.closure_column_range.set(range(2, 999))
    _run_within_context(
        lambda: test_func(2, 3),
        column_range=common.NamedRange(
            common.Dimension("K", kind=common.DimensionKind.VERTICAL), range(0, 3)
        ),
    )


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
