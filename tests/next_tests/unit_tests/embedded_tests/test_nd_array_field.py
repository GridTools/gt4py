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

from gt4py.next import common
from gt4py.next.embedded import nd_array_field
from gt4py.next.ffront import fbuiltins

from next_tests.integration_tests.feature_tests.math_builtin_test_data import math_builtin_test_data


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
