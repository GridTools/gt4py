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

from gt4py.next import common
from gt4py.next.common import Dimension, UnitRange
from gt4py.next.embedded import function_field
from gt4py.next.embedded.nd_array_field import _get_slices_from_domain_slice


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
def test_binary_operations_constant_field(op_func, expected_result):
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
    assert res.dtype == float


@pytest.mark.parametrize(
    "index", [((IDim, UnitRange(0, 10)),), common.Domain(dims=(IDim,), ranges=(UnitRange(0, 10),))]
)
def test_constant_field_getitem_missing_domain(index):
    cf = constant_field.ConstantField(10)
    with pytest.raises(IndexError):
        cf[index]


def test_constant_field_ndarray():
    domain = common.Domain(dims=(IDim, JDim), ranges=(UnitRange(3, 13), UnitRange(-5, 5)))
    cf = constant_field.ConstantField(10, domain)
    assert cf.ndarray.shape == (10, 10)
    assert np.all(cf.ndarray == 10)


def test_constant_field_binary_op_with_field():
    domain = common.Domain(dims=(IDim, JDim), ranges=(UnitRange(3, 13), UnitRange(-5, 5)))
    field = common.field(np.ones((10, 10)), domain=domain)
    cf = constant_field.ConstantField(10)

    result = cf + field
    assert result.ndarray.shape == (10, 10)
    assert np.all(result.ndarray == 11)


test_cases = [
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
    "domain1, field_data, domain2, constant_value, expected_shape, expected_value", test_cases
)
def test_constant_field_binary_op_with_field_intersection(
    domain1, field_data, domain2, constant_value, expected_shape, expected_value
):
    field = common.field(field_data, domain=domain1)
    cf = constant_field.ConstantField(constant_value, domain2)

    result = cf + field
    assert result.ndarray.shape == expected_shape
    assert np.all(result.ndarray == expected_value)


@pytest.mark.parametrize(
    "index, expected_shape, expected_domain",
    [
        (
            (slice(None, 5), slice(None, 2)),
            (5, 2),
            common.Domain((IDim, JDim), (UnitRange(5, 10), UnitRange(2, 4))),
        ),
        (
            (slice(None, 5),),
            (5, 10),
            common.Domain((IDim, JDim), (UnitRange(5, 10), UnitRange(2, 12))),
        ),
        ((Ellipsis, 1), (10,), common.Domain((IDim,), (UnitRange(5, 15),))),
        (
            (slice(2, 3), slice(5, 7)),
            (1, 2),
            common.Domain((IDim, JDim), (UnitRange(7, 8), UnitRange(7, 9))),
        ),
        (
            (slice(1, 2), 0),
            (1,),
            common.Domain((IDim,), (UnitRange(6, 7),)),
        ),
    ],
)
def test_relative_indexing_slice_2D(index, expected_shape, expected_domain):
    domain = common.Domain(dims=(IDim, JDim), ranges=(UnitRange(5, 15), UnitRange(2, 12)))
    field = constant_field.ConstantField(np.ones((10, 10)), domain)
    indexed_field = field[index]

    assert isinstance(indexed_field, constant_field.ConstantField)
    assert indexed_field.ndarray.shape == expected_shape
    assert indexed_field.domain == expected_domain


@pytest.mark.parametrize(
    "domain_slice,expected_dimensions,expected_shape",
    [
        (
            (
                (IDim, UnitRange(7, 9)),
                (JDim, UnitRange(8, 10)),
            ),
            (IDim, JDim, KDim),
            (2, 2, 15),
        ),
        (
            (
                (IDim, UnitRange(7, 9)),
                (KDim, UnitRange(12, 20)),
            ),
            (IDim, JDim, KDim),
            (2, 10, 8),
        ),
        (common.Domain(dims=(IDim,), ranges=(UnitRange(7, 9),)), (IDim, JDim, KDim), (2, 10, 15)),
        (((IDim, 8),), (JDim, KDim), (10, 15)),
        (((JDim, 9),), (IDim, KDim), (5, 15)),
        (((KDim, 11),), (IDim, JDim), (5, 10)),
        (
            (
                (IDim, 8),
                (JDim, UnitRange(8, 10)),
            ),
            (JDim, KDim),
            (2, 15),
        ),
        ((IDim, 1), (JDim, KDim), (10, 15)),
        ((IDim, UnitRange(5, 7)), (IDim, JDim, KDim), (2, 10, 15)),
    ],
)
def test_absolute_indexing(domain_slice, expected_dimensions, expected_shape):
    domain = common.Domain(
        dims=(IDim, JDim, KDim), ranges=(UnitRange(5, 10), UnitRange(5, 15), UnitRange(10, 25))
    )
    field = constant_field.ConstantField(np.ones((5, 10, 15)), domain)
    indexed_field = field[domain_slice]

    assert isinstance(indexed_field, constant_field.ConstantField)
    assert indexed_field.ndarray.shape == expected_shape
    assert indexed_field.domain.dims == expected_dimensions


def test_absolute_indexing_value_return():
    domain = common.Domain(dims=(IDim, JDim), ranges=(UnitRange(10, 20), UnitRange(5, 15)))
    field = constant_field.ConstantField(np.ones((10, 10), dtype=np.int32), domain)

    named_index = ((IDim, 2), (JDim, 4))
    value = field[named_index]

    assert isinstance(value, np.int32)
    assert value == 1
