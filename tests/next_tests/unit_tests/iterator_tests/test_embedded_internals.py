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

from next_tests.unit_tests.conftest import strict_array_on_device1


def test_column_ufunc():
    def test_func():
        a = embedded.Column(1, np.asarray(range(0, 3)))
        b = embedded.Column(1, np.asarray(range(3, 6)))
        res = a + b

        assert isinstance(res, embedded.Column)
        assert np.array_equal(res.data, a.data + b.data)
        assert res.kstart == 1

    with embedded_context.update(offset_provider={}, closure_column_range="SOMETHING"):
        test_func()

    def test_func(data_a: int, data_b: int):
        a = embedded.Column(1, data_a)
        b = embedded.Column(1, data_b)
        res = a + b

        assert isinstance(res, embedded.Column)
        assert np.array_equal(res.data, a.data + b.data)
        assert res.kstart == 1

    with embedded_context.update(closure_column_range=range(2, 999)):
        # Setting an invalid column_range here shouldn't affect other contexts

        with embedded_context.update(
            offset_provider={},
            closure_column_range=common.NamedRange(
                common.Dimension("K", kind=common.DimensionKind.VERTICAL), range(0, 3)
            ),
        ):
            test_func(2, 3)


def test_column_ufunc_with_scalar():
    def test_func():
        a = embedded.Column(1, np.asarray(range(0, 3)))
        res = 1.0 + a
        assert isinstance(res, embedded.Column)
        assert np.array_equal(res.data, a.data + 1.0)
        assert res.kstart == 1

    with embedded_context.update(offset_provider={}, closure_column_range="SOMETHING"):
        test_func()


def test_column_ufunc_wrong_kstart():
    def test_func():
        a = embedded.Column(1, np.asarray(range(0, 3)))
        wrong_kstart = embedded.Column(2, np.asarray(range(3, 6)))

        with pytest.raises(ValueError):
            a + wrong_kstart

    with embedded_context.update(offset_provider={}, closure_column_range="SOMETHING"):
        test_func()


def test_column_ufunc_wrong_shape():
    def test_func():
        a = embedded.Column(1, np.asarray(range(0, 3)))
        wrong_shape = embedded.Column(1, np.asarray([1, 2]))

        with pytest.raises(ValueError):
            a + wrong_shape

    with embedded_context.update(offset_provider={}, closure_column_range="SOMETHING"):
        test_func()


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

    with embedded_context.update(offset_provider={}, closure_column_range="SOMETHING"):
        test_func()


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

    with embedded_context.update(offset_provider={}, closure_column_range="SOMETHING"):
        test_func()


def test_column_array_function_wrong_kstart():
    def test_func():
        cond = embedded.Column(1, np.asarray([True, False]))
        wrong_kstart = embedded.Column(2, np.asarray([1, 1]))
        b = embedded.Column(1, np.asarray([2, 2]))

        with pytest.raises(ValueError):
            np.where(cond, wrong_kstart, b)

    with embedded_context.update(offset_provider={}, closure_column_range="SOMETHING"):
        test_func()


def test_column_array_function_wrong_shape():
    def test_func():
        cond = embedded.Column(1, np.asarray([True, False]))
        wrong_shape = embedded.Column(2, np.asarray([1, 1, 1]))
        b = embedded.Column(1, np.asarray([2, 2]))

        with pytest.raises(ValueError):
            np.where(cond, wrong_shape, b)

    with embedded_context.update(offset_provider={}, closure_column_range="SOMETHING"):
        test_func()


def test_lift_accepts_cartesian_dimension_offset():
    K = common.Dimension("K", kind=common.DimensionKind.VERTICAL)
    lifted = embedded.lift(lambda *args: 0)()
    lifted.shift(common.CartesianConnectivity(K), 1)  # must not raise


@pytest.mark.parametrize(
    "make_array",
    [
        pytest.param(lambda: np.arange(3.0), id="numpy"),
        pytest.param(
            lambda: strict_array_on_device1(np.arange(3.0)), id="array_api_strict-device1"
        ),
    ],
)
def test_as_fieldop_result_in_namespace_and_on_device_of_inputs(make_array):
    import array_api_compat

    from gt4py.next.iterator import builtins, runtime

    I = common.Dimension("I")
    data = make_array()
    inp = common._field(data, domain=common.domain({I: 3}))

    with embedded_context.update(offset_provider={}):
        result = builtins.as_fieldop(builtins.deref, runtime.CartesianDomain({I: range(3)}))(inp)

    assert array_api_compat.array_namespace(result.ndarray) is array_api_compat.array_namespace(
        data
    )
    assert array_api_compat.device(result.ndarray) == array_api_compat.device(data)
    np.testing.assert_array_equal(result.asnumpy(), [0.0, 1.0, 2.0])


@pytest.mark.requires_torch
@pytest.mark.requires_torch_cuda
@pytest.mark.requires_gpu
def test_as_fieldop_result_on_device_of_inputs():
    import torch

    from gt4py.next.iterator import builtins, runtime

    I = common.Dimension("I")
    inp = common._field(
        torch.arange(3.0, dtype=torch.float64, device="cuda"), domain=common.domain({I: 3})
    )

    with embedded_context.update(offset_provider={}):
        result = builtins.as_fieldop(builtins.deref, runtime.CartesianDomain({I: range(3)}))(inp)

    assert result.ndarray.device.type == "cuda"
    np.testing.assert_array_equal(result.asnumpy(), [0.0, 1.0, 2.0])
