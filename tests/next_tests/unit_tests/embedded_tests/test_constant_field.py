import operator

import pytest

import gt4py.next
from gt4py.next import common
from gt4py.next.embedded import constant_field
from gt4py.next.common import UnitRange
from tests.next_tests.unit_tests.embedded_tests.test_nd_array_field import IDim


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



# def test_constant_field_binary_op_with_field():
#     domain = common.Domain(dims=(IDim, JDim), ranges=(UnitRange(3, 13), UnitRange(-5, 5)))
#     field = common.field(np.ones((10, 10)), domain=domain)
#
#     cf = gt4py.next.embedded.nd_array_field.ConstantField(10)
#
#     result = cf + field
#     assert result.ndarray.shape == (5, 16)


# def test_constant_field_array():
#     cf = common.ConstantField(10)
#     domain = common.Domain(dims=(IDim,), ranges=(UnitRange(0, 10),))
#
#     cf_broadcasted = gt4py.next.embedded.nd_array_field._broadcast(cf, domain.dims)
#
#     result = cf[nr]
#     assert result.ndarray.shape == (5, 16)
#     assert np.all(result.ndarray == 10)