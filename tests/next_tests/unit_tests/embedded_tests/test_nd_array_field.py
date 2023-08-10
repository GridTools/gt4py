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

import itertools
import math
import operator
from typing import Callable, Iterable

import numpy as np
import pytest

from gt4py.next import common, Dimension
from gt4py.next.common import UnitRange
from gt4py.next.embedded import nd_array_field
from gt4py.next.embedded.nd_array_field import _slice_with_domain
from gt4py.next.ffront import fbuiltins

from next_tests.integration_tests.feature_tests.math_builtin_test_data import math_builtin_test_data

IDim = Dimension("IDim")
JDim = Dimension("JDim")
KDim = Dimension("KDim")


@pytest.fixture(params=nd_array_field._nd_array_implementations)
def nd_array_implementation(request):
    yield request.param


@pytest.fixture(
    params=[operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv],
)
def binary_op(request):
    yield request.param


def _make_field(lst: Iterable, nd_array_implementation):
    return common.field(
        nd_array_implementation.asarray(lst, dtype=nd_array_implementation.float32),
        domain=((common.Dimension("foo"), common.UnitRange(0, len(lst))),),
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


def test_binary_ops(binary_op, nd_array_implementation):
    inp_a = [-1.0, 4.2, 42]
    inp_b = [2.0, 3.0, -3.0]
    inputs = [inp_a, inp_b]

    expected = binary_op(*[np.asarray(inp, dtype=np.float32) for inp in inputs])

    field_inputs = [_make_field(inp, nd_array_implementation) for inp in inputs]

    result = binary_op(*field_inputs)

    assert np.allclose(result.ndarray, expected)


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


@pytest.mark.parametrize("arr,domain,target_domain,expected_shape,expected_slices",
                         [(
                                 np.random.rand(10, ),
                                 common.Domain(dims=(IDim,), ranges=(UnitRange(30, 40),)),
                                 common.Domain(dims=(IDim,), ranges=(UnitRange(30, 35),)),
                                 (5,),
                                 (slice(0, 5),)
                         ),
                             (
                                     np.random.rand(10, 6),
                                     common.Domain(dims=(IDim, JDim), ranges=(UnitRange(30, 40), UnitRange(17, 23))),
                                     common.Domain(dims=(IDim, JDim), ranges=(UnitRange(30, 35), UnitRange(20, 23))),
                                     (5, 3),
                                     (slice(0, 5), slice(3, 6))
                             ),
                             (
                                     np.random.rand(10, 6, 4),
                                     common.Domain(dims=(IDim, JDim, KDim),
                                                   ranges=(UnitRange(30, 40), UnitRange(17, 23), UnitRange(2, 6))),
                                     common.Domain(dims=(IDim, JDim, KDim),
                                                   ranges=(UnitRange(30, 35), UnitRange(20, 23), UnitRange(2, 4))),
                                     (5, 3, 2),
                                     (slice(0, 5), slice(3, 6), slice(0, 2))
                             ), ])
def test_slice_with_domain(nd_array_implementation, arr, domain, target_domain, expected_shape, expected_slices):
    f = common.field(arr, domain=domain)
    f_slice = _slice_with_domain(f, target_domain)
    assert f_slice.shape == expected_shape
    assert np.allclose(f_slice, arr[expected_slices])


def test_field_intersection_binary_op(nd_array_implementation):
    arr1 = np.random.rand(10,)
    arr2 = np.random.rand(3, 10)
    d1 = common.Domain(dims=(JDim,), ranges=(UnitRange(0, 10),))
    d2 = common.Domain(dims=(IDim, JDim), ranges=(UnitRange(0, 3), UnitRange(8, 18)))
    f1, f2 = common.field(arr1, domain=d1), common.field(arr2, domain=d2)
    intersection = d1 & d2
    new = _slice_with_domain(f1, intersection) + _slice_with_domain(f2, intersection)

    assert new.shape == (3, 2)
    assert np.allclose(new, arr1[8:10, ] + arr2[:, :2])