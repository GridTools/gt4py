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
import math
import operator

import numpy as np
import pytest

from gt4py.next import common
from gt4py.next.common import Dimension, UnitRange
from gt4py.next.embedded import exceptions as embedded_exceptions, function_field as funcf
from gt4py.next import fbuiltins

from .test_common import get_infinite_domain, get_mixed_domain



I = Dimension("I")
J = Dimension("J")
K = Dimension("K")


def rfloordiv(x, y):
    return operator.floordiv(y, x)


operators = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
    lambda x, y: operator.truediv(y, x),  # Reverse true division
    lambda x, y: operator.add(y, x),  # Reverse addition
    lambda x, y: operator.mul(y, x),  # Reverse multiplication
    lambda x, y: operator.sub(y, x),  # Reverse subtraction
    lambda x, y: operator.floordiv(y, x),  # Reverse floor division
]

logical_operators = [
    operator.xor,
    operator.and_,
    operator.or_,
]


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
def test_constant_field_no_domain(op_func, expected_result):
    cf1 = funcf.constant_field(10)
    cf2 = funcf.constant_field(20)
    result = op_func(cf1, cf2)
    assert result.func() == expected_result


@pytest.fixture(
    params=[((I, UnitRange(0, 10)),), common.Domain(dims=(I,), ranges=(UnitRange(0, 10),))]
)
def test_index(request):
    return request.param


def test_constant_field_getitem_missing_domain(test_index):
    cf = funcf.constant_field(10)
    with pytest.raises(embedded_exceptions.IndexOutOfBounds):
        cf[test_index]


def test_constant_field_getitem_missing_domain_ellipsis(test_index):
    cf = funcf.constant_field(10)
    cf[...].domain == cf.domain


@pytest.mark.parametrize(
    "domain",
    [
        common.Domain(dims=(I, J), ranges=(UnitRange(3, 13), UnitRange(-5, 5))),
        common.Domain(
            dims=(I, J, K), ranges=(UnitRange(-6, -3), UnitRange(-5, 10), UnitRange(1, 2))
        ),
    ],
)
def test_constant_field_ndarray(domain):
    cf = funcf.constant_field(10, domain)
    assert isinstance(cf.ndarray, int)
    assert cf.ndarray == 10


def test_constant_field_and_field_op():
    domain = common.Domain(dims=(I, J), ranges=(UnitRange(3, 13), UnitRange(-5, 5)))
    field = common.field(np.ones((10, 10)), domain=domain)
    cf = funcf.constant_field(10)

    result = cf + field
    assert np.allclose(result.ndarray, 11)
    assert result.domain == domain


binary_op_field_intersection_cases = [
    (
        common.Domain(dims=(I, J), ranges=(UnitRange(3, 13), UnitRange(-5, 5))),
        np.ones((10, 10)),
        common.Domain(dims=(I, J), ranges=(UnitRange(3, 5), UnitRange(0, 5))),
        2.0,
        (2, 5),
        3,
    ),
    (
        common.Domain(dims=(I, J), ranges=(UnitRange(-5, 2), UnitRange(3, 8))),
        np.ones((7, 5)),
        common.Domain(dims=(I,), ranges=(UnitRange(-5, 0),)),
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
    cf = funcf.constant_field(constant_value, domain2)

    result = cf + field
    assert result.ndarray.shape == expected_shape
    assert np.all(result.ndarray == expected_value)


def adder(i, j):
    return i + j


@pytest.mark.parametrize(
    "op_func",
    operators,
)
def test_function_field_broadcast(op_func):
    func1 = lambda x, y: x + y
    func2 = lambda y: 2 * y

    domain1 = common.Domain(dims=(I, J), ranges=(common.UnitRange(1, 10), common.UnitRange(5, 10)))
    domain2 = common.Domain(dims=(J,), ranges=(common.UnitRange(7, 15),))

    ff1 = funcf.FunctionField(func1, domain1)
    ff2 = funcf.FunctionField(func2, domain2)

    result = op_func(ff1, ff2)

    assert result.func(5, 10) == op_func(func1(5, 10), func2(10))
    assert isinstance(result.ndarray, np.ndarray)


@pytest.mark.parametrize("op_func", logical_operators)
def test_function_field_logical_operators(op_func):
    func1 = lambda x, y: x > 5
    func2 = lambda y: y < 10

    domain1 = common.Domain(dims=(I, J), ranges=(common.UnitRange(1, 10), common.UnitRange(5, 10)))
    domain2 = common.Domain(dims=(J,), ranges=(common.UnitRange(7, 15),))

    ff1 = funcf.FunctionField(func1, domain1)
    ff2 = funcf.FunctionField(func2, domain2)

    result = op_func(ff1, ff2)

    assert result.func(5, 10) == op_func(func1(5, 10), func2(10))
    assert isinstance(result.ndarray, np.ndarray)


@pytest.mark.parametrize(
    "domain,expected_shape",
    [
        (common.Domain(dims=(I, J), ranges=(UnitRange(3, 13), UnitRange(-5, 5))), (10, 10)),
    ],
)
def test_function_field_ndarray(domain, expected_shape):
    ff = funcf.FunctionField(adder, domain)
    assert ff.ndarray.shape == expected_shape

    ff_func = lambda *indices: adder(*indices)
    expected_values = np.fromfunction(ff_func, ff.ndarray.shape)
    assert np.allclose(ff.ndarray, expected_values)


@pytest.mark.parametrize(
    "domain",
    [
        common.Domain(dims=(I, J), ranges=(UnitRange(3, 13), UnitRange(-5, 5))),
    ],
)
def test_function_field_with_field(domain):
    ff = funcf.FunctionField(adder, domain)
    field = common.field(np.ones((10, 10)), domain=domain)

    result = ff + field
    ff_func = lambda *indices: adder(*indices) + 1
    expected_values = np.fromfunction(ff_func, result.ndarray.shape)

    assert result.ndarray.shape == (10, 10)
    assert np.allclose(result.ndarray, expected_values)


def test_function_field_function_field_op():
    res = funcf.FunctionField(
        lambda x, y: x + 42 * y,
        domain=common.Domain(
            dims=(I, J), ranges=(common.UnitRange(1, 10), common.UnitRange(5, 10))
        ),
    ) + funcf.FunctionField(
        lambda y: 2 * y, domain=common.Domain(dims=(J,), ranges=(common.UnitRange(7, 15),))
    )

    assert res.func(1, 2) == 89


@pytest.fixture
def function_field():
    return funcf.FunctionField(
        adder,
        domain=common.Domain(
            dims=(I, J), ranges=(common.UnitRange(1, 10), common.UnitRange(5, 10))
        ),
    )


def test_function_field_unary(function_field):
    pos_result = +function_field
    assert pos_result.func(1, 2) == 3

    neg_result = -function_field
    assert neg_result.func(1, 2) == -3

    abs_result = abs(function_field)
    assert abs_result.func(1, 2) == 3


def test_function_field_scalar_op(function_field):
    new = function_field * 5.0
    assert new.func(1, 2) == 15


@pytest.mark.parametrize("func", ["foo", 1.0, 1])
def test_function_field_invalid_func(func):
    with pytest.raises(embedded_exceptions.FunctionFieldError, match="Invalid first argument type"):
        funcf.FunctionField(func)


@pytest.mark.parametrize(
    "domain",
    [
        common.Domain(),
        common.Domain(*((I, UnitRange(1, 10)), (J, UnitRange(5, 10)))),
    ],
)
def test_function_field_invalid_invariant(domain):
    with pytest.raises(embedded_exceptions.FunctionFieldError, match="Invariant violation"):
        funcf.FunctionField(lambda *args, x: x, domain)


def test_function_field_infinite_range(get_infinite_domain, get_mixed_domain):
    domains = [get_infinite_domain, get_mixed_domain]
    for d in domains:
        with pytest.raises(embedded_exceptions.InfiniteRangeNdarrayError):
            ff = funcf.FunctionField(adder, d)
            ff.ndarray


@pytest.mark.parametrize("builtin_name", funcf._BUILTINS)
def test_function_field_builtins(function_field, builtin_name):
    if builtin_name in ["abs", "power", "gamma"]:
        pytest.skip(f"Skipping '{builtin_name}'")

    fbuiltin_func = getattr(fbuiltins, builtin_name)

    result = fbuiltin_func(function_field).func(1, 2)

    if math.isnan(result):
        assert math.isnan(np.__getattribute__(builtin_name)(3))
    else:
        assert result == np.__getattribute__(builtin_name)(3)


def test_unary_logical_op_boolean():
    boolean_func = lambda x: x % 2 != 0
    field = funcf.FunctionField(boolean_func, common.Domain((I, UnitRange(1, 10))))
    assert np.allclose(~field.ndarray, np.invert(np.fromfunction(boolean_func, (9,))))


def test_unary_logical_op_scalar():
    scalar_func = lambda x: x % 2
    field = funcf.FunctionField(scalar_func, common.Domain((I, UnitRange(1, 10))))
    with pytest.raises(NotImplementedError):
        ~field
