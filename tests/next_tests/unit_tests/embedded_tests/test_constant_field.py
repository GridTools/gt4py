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

import operator

import numpy as np
import pytest

from gt4py._core.definitions import float64
from gt4py.next import common
from gt4py.next.common import Dimension, UnitRange
from gt4py.next.embedded import constant_field


IDim = Dimension("IDim")
JDim = Dimension("JDim")
KDim = Dimension("KDim")


def rfloordiv(x, y):
    return operator.floordiv(y, x)


@pytest.mark.parametrize(
    "op_func, expected_result",
    [
        (operator.add, 10 + 20),
        (operator.sub, 10 - 20),
        (operator.mul, 10 * 20),
        (operator.truediv, 10 / 20),
        (operator.floordiv, 10 // 20),
        (rfloordiv, 20 // 10),
        (operator.pow, 10**20),
        (lambda x, y: operator.truediv(y, x), 20 / 10),
        (operator.add, 10 + 20),
        (operator.mul, 10 * 20),
        (lambda x, y: operator.sub(y, x), 20 - 10),
    ],
)
def test_constant_field_binary_op_with_constant_field(op_func, expected_result):
    cf1 = constant_field.ConstantField(10)
    cf2 = constant_field.ConstantField(20)
    result = op_func(cf1, cf2)
    assert result.value == expected_result


@pytest.mark.parametrize(
    "cf1,cf2,expected",
    [
        (constant_field.ConstantField(10.0), constant_field.ConstantField(20), 30.0),
        (constant_field.ConstantField(10.0), 10, 20.0),
    ],
)
def test_constant_field_binary_op_float(cf1, cf2, expected):
    res = cf1 + cf2
    assert res.value == expected
    assert res.dtype.dtype == float64


@pytest.mark.parametrize(
    "index", [((IDim, UnitRange(0, 10)),), common.Domain(dims=(IDim,), ranges=(UnitRange(0, 10),))]
)
def test_constant_field_getitem_missing_domain(index):
    cf = constant_field.ConstantField(10)
    with pytest.raises(IndexError):
        cf[index]


@pytest.mark.parametrize(
    "domain,expected_shape",
    [
        (common.Domain(dims=(IDim, JDim), ranges=(UnitRange(3, 13), UnitRange(-5, 5))), (10, 10)),
        (
            common.Domain(
                dims=(IDim, JDim, KDim),
                ranges=(UnitRange(-6, -3), UnitRange(-5, 10), UnitRange(1, 2)),
            ),
            (3, 15, 1),
        ),
    ],
)
def test_constant_field_ndarray(domain, expected_shape):
    cf = constant_field.ConstantField(10, domain)
    assert cf.ndarray.shape == expected_shape
    assert np.all(cf.ndarray == 10)


def test_constant_field_empty_domain_op():
    domain = common.Domain(dims=(IDim, JDim), ranges=(UnitRange(3, 13), UnitRange(-5, 5)))
    field = common.field(np.ones((10, 10)), domain=domain)
    cf = constant_field.ConstantField(10)

    result = cf + field
    assert result.ndarray.shape == (10, 10)
    assert np.all(result.ndarray == 11)


binary_op_field_intersection_cases = [
    (
        common.Domain(dims=(IDim, JDim), ranges=(UnitRange(3, 13), UnitRange(-5, 5))),
        np.ones((10, 10)),
        common.Domain(dims=(IDim, JDim), ranges=(UnitRange(3, 5), UnitRange(0, 5))),
        2,
        (2, 5),
        3,
    ),
    (
        common.Domain(dims=(IDim, JDim), ranges=(UnitRange(-5, 2), UnitRange(3, 8))),
        np.ones((7, 5)),
        common.Domain(dims=(IDim,), ranges=(UnitRange(-5, 0),)),
        5,
        (5, 5),
        6,
    ),
]


@pytest.mark.parametrize(
    "domain1, field_data, domain2, constant_value, expected_shape, expected_value",
    binary_op_field_intersection_cases,
)
def test_constant_field_non_empty_domain_op(
    domain1, field_data, domain2, constant_value, expected_shape, expected_value
):
    field = common.field(field_data, domain=domain1)
    cf = constant_field.ConstantField(constant_value, domain2)

    result = cf + field
    assert result.ndarray.shape == expected_shape
    assert np.all(result.ndarray == expected_value)


def adder(i, j, k=None):
    return i + j


@pytest.mark.parametrize(
    "domain,expected_shape",
    [
        (common.Domain(dims=(IDim, JDim), ranges=(UnitRange(3, 13), UnitRange(-5, 5))), (10, 10)),
        (
                common.Domain(
                    dims=(IDim, JDim, KDim),
                    ranges=(UnitRange(-6, -3), UnitRange(-5, 10), UnitRange(1, 2)),
                ),
                (3, 15, 1),
        ),
    ],
)
def test_function_field_ndarray(domain, expected_shape):
    ff = constant_field.FunctionField(adder, domain)
    assert ff.ndarray.shape == expected_shape

    ff_func = lambda *indices: adder(*indices)
    expected_values = np.fromfunction(ff_func, ff.ndarray.shape)
    assert np.allclose(ff.ndarray, expected_values)


@pytest.mark.parametrize(
    "domain",
    [
        common.Domain(dims=(IDim, JDim), ranges=(UnitRange(3, 13), UnitRange(-5, 5))),
    ],
)
def test_function_field_unary(domain):
    ff = constant_field.FunctionField(adder, domain)

    # Test negation and absolute value
    for operation in [lambda x: -x, abs]:
        modified_ff = operation(ff)

        ff_func = lambda *indices: operation(adder(*indices))
        expected_values = np.fromfunction(ff_func, ff.ndarray.shape)

        assert np.allclose(modified_ff.ndarray, expected_values)


# TODO: add more tests with domain intersection
@pytest.mark.parametrize(
    "domain",
    [
        common.Domain(dims=(IDim, JDim), ranges=(UnitRange(3, 13), UnitRange(-5, 5))),
    ],
)
def test_function_field_with_field(domain):
    ff = constant_field.FunctionField(adder, domain)
    field = common.field(np.ones((10, 10)), domain=domain)

    result = ff + field

    ff_func = lambda *indices: adder(*indices) + 1
    expected_values = np.fromfunction(ff_func, result.ndarray.shape)

    assert result.ndarray.shape == (10, 10)
    assert np.allclose(result.ndarray, expected_values)
