import operator

import pytest
import numpy as np
from gt4py.next import common
from gt4py.next.common import UnitRange, Dimension
from gt4py.next.embedded import constant_field

IDim = Dimension("IDim")
JDim = Dimension("JDim")

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


def test_constant_field_binary_op_with_field_intersection():
    domain = common.Domain(dims=(IDim, JDim), ranges=(UnitRange(3, 13), UnitRange(-5, 5)))
    field = common.field(np.ones((10, 10)), domain=domain)

    domain2 = common.Domain(dims=(IDim, JDim), ranges=(UnitRange(3, 5), UnitRange(0, 5)))
    cf = constant_field.ConstantField(2, domain2)

    result = cf + field
    assert result.ndarray.shape == (2, 5)
    assert np.all(result.ndarray == 3)