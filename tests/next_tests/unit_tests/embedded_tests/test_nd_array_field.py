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
import dataclasses
import itertools
import math
import operator
from typing import Callable, Iterable

import numpy as np
import pytest

from gt4py.next import Dimension, common
from gt4py.next.common import Domain, UnitRange
from gt4py.next.embedded import nd_array_field
from gt4py.next.embedded.nd_array_field import _get_slices_from_domain_slice, _slice_range
from gt4py.next.ffront import fbuiltins

from next_tests.integration_tests.feature_tests.math_builtin_test_data import math_builtin_test_data


IDim = Dimension("IDim")
JDim = Dimension("JDim")
KDim = Dimension("KDim")


@pytest.fixture(params=nd_array_field._nd_array_implementations)
def nd_array_implementation(request):
    yield request.param


@pytest.fixture(
    params=[
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.mod,
    ]
)
def binary_arithmetic_op(request):
    yield request.param


@pytest.fixture(
    params=[operator.xor, operator.and_, operator.or_],
)
def binary_logical_op(request):
    yield request.param


@pytest.fixture(params=[operator.neg, operator.pos])
def unary_arithmetic_op(request):
    yield request.param


@pytest.fixture(params=[operator.invert])
def unary_logical_op(request):
    yield request.param


def _make_field(lst: Iterable, nd_array_implementation, *, dtype=None):
    if not dtype:
        dtype = nd_array_implementation.float32
    return common.field(
        nd_array_implementation.asarray(lst, dtype=dtype),
        domain=common.Domain((common.Dimension("foo"),), (common.UnitRange(0, len(lst)),)),
    )


@pytest.mark.parametrize("builtin_name, inputs", math_builtin_test_data())
def test_math_function_builtins(builtin_name: str, inputs, nd_array_implementation):
    if builtin_name == "gamma":
        # numpy has no gamma function
        pytest.xfail("TODO: implement gamma")
        ref_impl: Callable = np.vectorize(math.gamma)
    else:
        ref_impl: Callable = getattr(np, builtin_name)

    expected = ref_impl(*[np.asarray(inp, dtype=np.float32) for inp in inputs])

    field_inputs = [_make_field(inp, nd_array_implementation) for inp in inputs]

    builtin = getattr(fbuiltins, builtin_name)
    result = builtin(*field_inputs)

    assert np.allclose(result.ndarray, expected)


def test_binary_arithmetic_ops(binary_arithmetic_op, nd_array_implementation):
    inp_a = [-1.0, 4.2, 42]
    inp_b = [2.0, 3.0, -3.0]
    inputs = [inp_a, inp_b]

    expected = binary_arithmetic_op(*[np.asarray(inp, dtype=np.float32) for inp in inputs])

    field_inputs = [_make_field(inp, nd_array_implementation) for inp in inputs]

    result = binary_arithmetic_op(*field_inputs)

    assert np.allclose(result.ndarray, expected)


def test_binary_logical_ops(binary_logical_op, nd_array_implementation):
    inp_a = [True, True, False, False]
    inp_b = [True, False, True, False]
    inputs = [inp_a, inp_b]

    expected = binary_logical_op(*[np.asarray(inp) for inp in inputs])

    field_inputs = [_make_field(inp, nd_array_implementation, dtype=bool) for inp in inputs]

    result = binary_logical_op(*field_inputs)

    assert np.allclose(result.ndarray, expected)


def test_unary_logical_ops(unary_logical_op, nd_array_implementation):
    inp = [
        True,
        False,
    ]

    expected = unary_logical_op(np.asarray(inp))

    field_input = _make_field(inp, nd_array_implementation, dtype=bool)

    result = unary_logical_op(field_input)

    assert np.allclose(result.ndarray, expected)


def test_unary_arithmetic_ops(unary_arithmetic_op, nd_array_implementation):
    inp = [1.0, -2.0, 0.0]

    expected = unary_arithmetic_op(np.asarray(inp, dtype=np.float32))

    field_input = _make_field(inp, nd_array_implementation)

    result = unary_arithmetic_op(field_input)

    assert np.allclose(result.ndarray, expected)


@pytest.mark.parametrize(
    "dims,expected_indices",
    [
        ((IDim,), (slice(5, 10), None)),
        ((JDim,), (None, slice(5, 10))),
    ],
)
def test_binary_operations_with_intersection(binary_arithmetic_op, dims, expected_indices):
    arr1 = np.arange(10)
    arr1_domain = common.Domain(dims=dims, ranges=(UnitRange(0, 10),))

    arr2 = np.ones((5, 5))
    arr2_domain = common.Domain(dims=(IDim, JDim), ranges=(UnitRange(5, 10), UnitRange(5, 10)))

    field1 = common.field(arr1, domain=arr1_domain)
    field2 = common.field(arr2, domain=arr2_domain)

    op_result = binary_arithmetic_op(field1, field2)
    expected_result = binary_arithmetic_op(arr1[expected_indices[0], expected_indices[1]], arr2)

    assert op_result.ndarray.shape == (5, 5)
    assert np.allclose(op_result.ndarray, expected_result)


@pytest.fixture(
    params=itertools.product(
        nd_array_field._nd_array_implementations, nd_array_field._nd_array_implementations
    ),
    ids=lambda param: f"{param[0].__name__}-{param[1].__name__}",
)
def product_nd_array_implementation(request):
    yield request.param


def test_mixed_fields(product_nd_array_implementation):
    first_impl, second_impl = product_nd_array_implementation
    if (first_impl.__name__ == "cupy" and second_impl.__name__ == "numpy") or (
        first_impl.__name__ == "numpy" and second_impl.__name__ == "cupy"
    ):
        pytest.skip("Binary operation between CuPy and NumPy requires explicit conversion.")

    inp_a = [-1.0, 4.2, 42]
    inp_b = [2.0, 3.0, -3.0]

    expected = np.asarray(inp_a) + np.asarray(inp_b)

    field_inp_a = _make_field(inp_a, first_impl)
    field_inp_b = _make_field(inp_b, second_impl)

    result = field_inp_a + field_inp_b
    assert np.allclose(result.ndarray, expected)


def test_non_dispatched_function():
    @fbuiltins.builtin_function
    def fma(a: common.Field, b: common.Field, c: common.Field, /) -> common.Field:
        return a * b + c

    inp_a = [-1.0, 4.2, 42]
    inp_b = [2.0, 3.0, -3.0]
    inp_c = [-2.0, -3.0, 3.0]

    expected = np.asarray(inp_a) * np.asarray(inp_b) + np.asarray(inp_c)

    field_inp_a = _make_field(inp_a, np)
    field_inp_b = _make_field(inp_b, np)
    field_inp_c = _make_field(inp_c, np)

    result = fma(field_inp_a, field_inp_b, field_inp_c)
    assert np.allclose(result.ndarray, expected)


@pytest.mark.parametrize(
    "new_dims,field,expected_domain",
    [
        (
            (
                (IDim,),
                common.field(
                    np.arange(10), domain=common.Domain(dims=(IDim,), ranges=(UnitRange(0, 10),))
                ),
                Domain(dims=(IDim,), ranges=(UnitRange(0, 10),)),
            )
        ),
        (
            (
                (IDim, JDim),
                common.field(
                    np.arange(10), domain=common.Domain(dims=(IDim,), ranges=(UnitRange(0, 10),))
                ),
                Domain(dims=(IDim, JDim), ranges=(UnitRange(0, 10), UnitRange.infinity())),
            )
        ),
        (
            (
                (IDim, JDim),
                common.field(
                    np.arange(10), domain=common.Domain(dims=(JDim,), ranges=(UnitRange(0, 10),))
                ),
                Domain(dims=(IDim, JDim), ranges=(UnitRange.infinity(), UnitRange(0, 10))),
            )
        ),
        (
            (
                (IDim, JDim, KDim),
                common.field(
                    np.arange(10), domain=common.Domain(dims=(JDim,), ranges=(UnitRange(0, 10),))
                ),
                Domain(
                    dims=(IDim, JDim, KDim),
                    ranges=(UnitRange.infinity(), UnitRange(0, 10), UnitRange.infinity()),
                ),
            )
        ),
    ],
)
def test_field_broadcast(new_dims, field, expected_domain):
    result = fbuiltins.broadcast(field, new_dims)
    assert result.domain == expected_domain


@pytest.mark.parametrize(
    "domain_slice",
    [
        ((IDim, UnitRange(0, 10)),),
        common.Domain(dims=(IDim,), ranges=(UnitRange(0, 10),)),
    ],
)
def test_get_slices_with_named_indices_3d_to_1d(domain_slice):
    field_domain = common.Domain(
        dims=(IDim, JDim, KDim), ranges=(UnitRange(0, 10), UnitRange(0, 10), UnitRange(0, 10))
    )
    slices = _get_slices_from_domain_slice(field_domain, domain_slice)
    assert slices == (slice(0, 10, None), slice(None), slice(None))


def test_get_slices_with_named_index():
    field_domain = common.Domain(
        dims=(IDim, JDim, KDim), ranges=(UnitRange(0, 10), UnitRange(0, 10), UnitRange(0, 10))
    )
    named_index = ((IDim, UnitRange(0, 10)), (JDim, 2), (KDim, 3))
    slices = _get_slices_from_domain_slice(field_domain, named_index)
    assert slices == (slice(0, 10, None), 2, 3)


def test_get_slices_invalid_type():
    field_domain = common.Domain(
        dims=(IDim, JDim, KDim), ranges=(UnitRange(0, 10), UnitRange(0, 10), UnitRange(0, 10))
    )
    new_domain = ((IDim, "1"),)
    with pytest.raises(ValueError):
        _get_slices_from_domain_slice(field_domain, new_domain)


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
    field = common.field(np.ones((5, 10, 15)), domain=domain)
    indexed_field = field[domain_slice]

    assert common.is_field(indexed_field)
    assert indexed_field.ndarray.shape == expected_shape
    assert indexed_field.domain.dims == expected_dimensions


def test_absolute_indexing_value_return():
    domain = common.Domain(dims=(IDim, JDim), ranges=(UnitRange(10, 20), UnitRange(5, 15)))
    field = common.field(np.ones((10, 10), dtype=np.int32), domain=domain)

    named_index = ((IDim, 2), (JDim, 4))
    value = field[named_index]

    assert isinstance(value, np.int32)
    assert value == 1


@pytest.mark.parametrize(
    "index, expected_shape, expected_domain",
    [
        (
            (slice(None, 5), slice(None, 2)),
            (5, 2),
            Domain((IDim, JDim), (UnitRange(5, 10), UnitRange(2, 4))),
        ),
        ((slice(None, 5),), (5, 10), Domain((IDim, JDim), (UnitRange(5, 10), UnitRange(2, 12)))),
        ((Ellipsis, 1), (10,), Domain((IDim,), (UnitRange(5, 15),))),
        (
            (slice(2, 3), slice(5, 7)),
            (1, 2),
            Domain((IDim, JDim), (UnitRange(7, 8), UnitRange(7, 9))),
        ),
        (
            (slice(1, 2), 0),
            (1,),
            Domain((IDim,), (UnitRange(6, 7),)),
        ),
    ],
)
def test_relative_indexing_slice_2D(index, expected_shape, expected_domain):
    domain = common.Domain(dims=(IDim, JDim), ranges=(UnitRange(5, 15), UnitRange(2, 12)))
    field = common.field(np.ones((10, 10)), domain=domain)
    indexed_field = field[index]

    assert common.is_field(indexed_field)
    assert indexed_field.ndarray.shape == expected_shape
    assert indexed_field.domain == expected_domain


@pytest.mark.parametrize(
    "index, expected_shape, expected_domain",
    [
        ((1, slice(None), 2), (15,), Domain((JDim,), (UnitRange(10, 25),))),
        (
            (slice(None), slice(None), 2),
            (10, 15),
            Domain((IDim, JDim), (UnitRange(5, 15), UnitRange(10, 25))),
        ),
        (
            (slice(None),),
            (10, 15, 10),
            Domain((IDim, JDim, KDim), (UnitRange(5, 15), UnitRange(10, 25), UnitRange(10, 20))),
        ),
        (
            (slice(None), slice(None), slice(None)),
            (10, 15, 10),
            Domain((IDim, JDim, KDim), (UnitRange(5, 15), UnitRange(10, 25), UnitRange(10, 20))),
        ),
        (
            (slice(None)),
            (10, 15, 10),
            Domain((IDim, JDim, KDim), (UnitRange(5, 15), UnitRange(10, 25), UnitRange(10, 20))),
        ),
        ((0, Ellipsis, 0), (15,), Domain((JDim,), (UnitRange(10, 25),))),
        (
            Ellipsis,
            (10, 15, 10),
            Domain((IDim, JDim, KDim), (UnitRange(5, 15), UnitRange(10, 25), UnitRange(10, 20))),
        ),
    ],
)
def test_relative_indexing_slice_3D(index, expected_shape, expected_domain):
    domain = common.Domain(
        dims=(IDim, JDim, KDim), ranges=(UnitRange(5, 15), UnitRange(10, 25), UnitRange(10, 20))
    )
    field = common.field(np.ones((10, 15, 10)), domain=domain)
    indexed_field = field[index]

    assert common.is_field(indexed_field)
    assert indexed_field.ndarray.shape == expected_shape
    assert indexed_field.domain == expected_domain


@pytest.mark.parametrize(
    "index, expected_value",
    [((1, 0), 10), ((0, 1), 1)],
)
def test_relative_indexing_value_return(index, expected_value):
    domain = common.Domain(dims=(IDim, JDim), ranges=(UnitRange(5, 15), UnitRange(2, 12)))
    field = common.field(np.reshape(np.arange(100, dtype=int), (10, 10)), domain=domain)
    indexed_field = field[index]

    assert indexed_field == expected_value


@pytest.mark.parametrize("lazy_slice", [lambda f: f[13], lambda f: f[:5, :3, :2]])
def test_relative_indexing_out_of_bounds(lazy_slice):
    domain = common.Domain(dims=(IDim, JDim), ranges=(UnitRange(3, 13), UnitRange(-5, 5)))
    field = common.field(np.ones((10, 10)), domain=domain)

    with pytest.raises(IndexError):
        lazy_slice(field)


@pytest.mark.parametrize("index", [IDim, "1", (IDim, JDim)])
def test_field_unsupported_index(index):
    domain = common.Domain(dims=(IDim,), ranges=(UnitRange(0, 10),))
    field = common.field(np.ones((10,)), domain=domain)
    with pytest.raises(IndexError, match="Unsupported index type"):
        field[index]


def test_slice_range():
    input_range = UnitRange(2, 10)
    slice_obj = slice(2, -2)
    expected = UnitRange(4, 8)

    result = _slice_range(input_range, slice_obj)
    assert result == expected
