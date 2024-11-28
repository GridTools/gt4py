# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math
import operator
from typing import Callable, Iterable, Optional

import numpy as np
import pytest

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.common import Dimension, Domain, NamedIndex, NamedRange, UnitRange
from gt4py.next.embedded import exceptions as embedded_exceptions, nd_array_field
from gt4py.next.embedded.nd_array_field import _get_slices_from_domain_slice
from gt4py.next.ffront import fbuiltins

from next_tests.integration_tests.feature_tests.math_builtin_test_data import math_builtin_test_data


D0 = Dimension("D0")
D1 = Dimension("D1")
D2 = Dimension("D2")


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


@pytest.fixture(params=[operator.xor, operator.and_, operator.or_])
def binary_logical_op(request):
    yield request.param


@pytest.fixture(params=[operator.neg, operator.pos])
def unary_arithmetic_op(request):
    yield request.param


@pytest.fixture(params=[operator.invert])
def unary_logical_op(request):
    yield request.param


def _make_default_domain(shape: tuple[int, ...]) -> Domain:
    return common.Domain(
        dims=tuple(Dimension(f"D{i}") for i in range(len(shape))),
        ranges=tuple(UnitRange(0, s) for s in shape),
    )


def _make_field_or_scalar(
    lst: Iterable | core_defs.Scalar, nd_array_implementation, *, domain=None, dtype=None
):
    """Creates a field from an Iterable or returns a scalar."""
    if not dtype:
        dtype = np.float32
    if isinstance(lst, core_defs.SCALAR_TYPES):
        return dtype(lst)
    buffer = nd_array_implementation.asarray(lst, dtype=dtype)
    if domain is None:
        domain = _make_default_domain(buffer.shape)
    return common._field(buffer, domain=domain)


def _np_asarray_or_scalar(value: Iterable | core_defs.Scalar, dtype=None):
    """Creates a numpy array from an Iterable or returns a scalar."""
    if not dtype:
        dtype = np.float32

    return (
        dtype(value)
        if isinstance(value, core_defs.SCALAR_TYPES)
        else np.asarray(value, dtype=dtype)
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

    field_inputs = [_make_field_or_scalar(inp, nd_array_implementation) for inp in inputs]

    builtin = getattr(fbuiltins, builtin_name)
    result = builtin(*field_inputs)

    assert np.allclose(result.ndarray, expected)


def test_where_builtin(nd_array_implementation):
    cond = np.asarray([True, False])
    true_ = np.asarray([1.0, 2.0], dtype=np.float32)
    false_ = np.asarray([3.0, 4.0], dtype=np.float32)

    field_inputs = [
        _make_field_or_scalar(inp, nd_array_implementation) for inp in [cond, true_, false_]
    ]
    expected = np.where(cond, true_, false_)

    result = fbuiltins.where(*field_inputs)
    assert np.allclose(result.ndarray, expected)


def test_where_builtin_different_domain(nd_array_implementation):
    cond = np.asarray([True, False])
    true_ = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    false_ = np.asarray([7.0, 8.0, 9.0, 10.0], dtype=np.float32)

    cond_field = common._field(nd_array_implementation.asarray(cond), domain=common.domain({D1: 2}))
    true_field = common._field(
        nd_array_implementation.asarray(true_),
        domain=common.domain({D0: common.UnitRange(0, 2), D1: common.UnitRange(-1, 2)}),
    )
    false_field = common._field(
        nd_array_implementation.asarray(false_), domain=common.domain({D1: common.UnitRange(-1, 3)})
    )

    expected = np.where(cond[np.newaxis, :], true_[:, 1:], false_[np.newaxis, 1:-1])

    result = fbuiltins.where(cond_field, true_field, false_field)
    assert np.allclose(result.ndarray, expected)


def test_where_builtin_with_tuple(nd_array_implementation):
    cond = np.asarray([True, False])
    true0 = np.asarray([1.0, 2.0], dtype=np.float32)
    false0 = np.asarray([3.0, 4.0], dtype=np.float32)
    true1 = np.asarray([11.0, 12.0], dtype=np.float32)
    false1 = np.asarray([13.0, 14.0], dtype=np.float32)

    expected0 = np.where(cond, true0, false0)
    expected1 = np.where(cond, true1, false1)

    cond_field = _make_field_or_scalar(cond, nd_array_implementation, dtype=bool)
    field_true = tuple(
        _make_field_or_scalar(inp, nd_array_implementation) for inp in [true0, true1]
    )
    field_false = tuple(
        _make_field_or_scalar(inp, nd_array_implementation) for inp in [false0, false1]
    )

    result = fbuiltins.where(cond_field, field_true, field_false)
    assert np.allclose(result[0].ndarray, expected0)
    assert np.allclose(result[1].ndarray, expected1)


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        ([-1.0, 4.2, 42], [2.0, 3.0, -3.0]),
        (2.0, [2.0, 3.0, -3.0]),  # scalar with field, tests reverse operators
    ],
)
def test_binary_arithmetic_ops(binary_arithmetic_op, nd_array_implementation, lhs, rhs):
    inputs = [lhs, rhs]

    expected = binary_arithmetic_op(*[_np_asarray_or_scalar(inp) for inp in inputs])

    field_inputs = [_make_field_or_scalar(inp, nd_array_implementation) for inp in inputs]

    result = binary_arithmetic_op(*field_inputs)

    assert np.allclose(result.ndarray, expected)


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        ([True, True, False, False], [True, False, True, False]),
        (True, [True, False]),
        (False, [True, False]),
    ],
)
def test_binary_logical_ops(binary_logical_op, nd_array_implementation, lhs, rhs):
    inputs = [lhs, rhs]

    expected = binary_logical_op(*[_np_asarray_or_scalar(inp, dtype=bool) for inp in inputs])

    field_inputs = [
        _make_field_or_scalar(inp, nd_array_implementation, dtype=bool) for inp in inputs
    ]

    result = binary_logical_op(*field_inputs)

    assert np.allclose(result.ndarray, expected)


def test_unary_logical_ops(unary_logical_op, nd_array_implementation):
    inp = [True, False]

    expected = unary_logical_op(np.asarray(inp))

    field_input = _make_field_or_scalar(inp, nd_array_implementation, dtype=bool)

    result = unary_logical_op(field_input)

    assert np.allclose(result.ndarray, expected)


def test_unary_arithmetic_ops(unary_arithmetic_op, nd_array_implementation):
    inp = [1.0, -2.0, 0.0]

    expected = unary_arithmetic_op(np.asarray(inp, dtype=np.float32))

    field_input = _make_field_or_scalar(inp, nd_array_implementation)

    result = unary_arithmetic_op(field_input)

    assert np.allclose(result.ndarray, expected)


@pytest.mark.parametrize(
    "dims,expected_indices", [((D0,), (slice(5, 10), None)), ((D1,), (None, slice(5, 10)))]
)
def test_binary_operations_with_intersection(binary_arithmetic_op, dims, expected_indices):
    arr1 = np.arange(10)
    arr1_domain = common.Domain(dims=dims, ranges=(UnitRange(0, 10),))

    arr2 = np.ones((5, 5))
    arr2_domain = common.Domain(dims=(D0, D1), ranges=(UnitRange(5, 10), UnitRange(5, 10)))

    field1 = common._field(arr1, domain=arr1_domain)
    field2 = common._field(arr2, domain=arr2_domain)

    op_result = binary_arithmetic_op(field1, field2)
    expected_result = binary_arithmetic_op(arr1[expected_indices[0], expected_indices[1]], arr2)

    assert op_result.ndarray.shape == (5, 5)
    assert np.allclose(op_result.ndarray, expected_result)


def test_as_scalar(nd_array_implementation):
    testee = common._field(
        nd_array_implementation.asarray(42.0, dtype=np.float32), domain=common.Domain()
    )

    result = testee.as_scalar()
    assert result == 42.0
    assert isinstance(result, np.float32)


def product_nd_array_implementation_params():
    for xp1 in nd_array_field._nd_array_implementations:
        for xp2 in nd_array_field._nd_array_implementations:
            marks = ()
            if any(hasattr(nd_array_field, "cp") and xp == nd_array_field.cp for xp in (xp1, xp2)):
                marks = pytest.mark.requires_gpu
            yield pytest.param((xp1, xp2), id=f"{xp1.__name__}-{xp2.__name__}", marks=marks)


@pytest.fixture(params=product_nd_array_implementation_params())
def product_nd_array_implementation(request):
    yield request.param


def test_mixed_fields(product_nd_array_implementation):
    first_impl, second_impl = product_nd_array_implementation
    if "numpy" in first_impl.__name__ and "cupy" in second_impl.__name__:
        pytest.skip("Binary operation between NumPy and CuPy requires explicit conversion.")

    inp_a = [-1.0, 4.2, 42]
    inp_b = [2.0, 3.0, -3.0]

    expected = np.asarray(inp_a) + np.asarray(inp_b)

    field_inp_a = _make_field_or_scalar(inp_a, first_impl)
    field_inp_b = _make_field_or_scalar(inp_b, second_impl)

    result = field_inp_a + field_inp_b
    assert np.allclose(result.ndarray, expected)


def test_non_dispatched_function():
    @fbuiltins.BuiltInFunction
    def fma(a: common.Field, b: common.Field, c: common.Field, /) -> common.Field:
        return a * b + c

    inp_a = [-1.0, 4.2, 42]
    inp_b = [2.0, 3.0, -3.0]
    inp_c = [-2.0, -3.0, 3.0]

    expected = np.asarray(inp_a) * np.asarray(inp_b) + np.asarray(inp_c)

    field_inp_a = _make_field_or_scalar(inp_a, np)
    field_inp_b = _make_field_or_scalar(inp_b, np)
    field_inp_c = _make_field_or_scalar(inp_c, np)

    result = fma(field_inp_a, field_inp_b, field_inp_c)
    assert np.allclose(result.ndarray, expected)


def test_domain_premap():
    # Translation case
    I = Dimension("I")
    J = Dimension("J")

    N = 10
    data_field = common._field(
        0.1 * np.arange(N * N).reshape((N, N)),
        domain=common.Domain(
            common.NamedRange(I, common.unit_range(N)), common.NamedRange(J, common.unit_range(N))
        ),
    )
    conn = common.CartesianConnectivity.for_translation(J, +1)

    result = data_field.premap(conn)
    expected = common._field(
        data_field.ndarray,
        domain=common.Domain(
            common.NamedRange(I, common.unit_range(N)),
            common.NamedRange(J, common.unit_range((-1, N - 1))),
        ),
    )

    assert result.domain == expected.domain
    assert np.all(result.ndarray == expected.ndarray)

    # Relocation case
    I_half = Dimension("I_half")

    conn = common.CartesianConnectivity.for_relocation(I, I_half)

    result = data_field.premap(conn)
    expected = common._field(
        data_field.ndarray,
        domain=common.Domain(
            dims=(
                I_half,
                J,
            ),
            ranges=(data_field.domain[I].unit_range, data_field.domain[J].unit_range),
        ),
    )

    assert result.domain == expected.domain
    assert np.all(result.ndarray == expected.ndarray)


def test_reshuffling_premap():
    I = Dimension("I")
    J = Dimension("J")

    ij_field = common._field(
        np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
        domain=common.Domain(dims=(I, J), ranges=(UnitRange(0, 3), UnitRange(0, 3))),
    )
    max_ij_conn = common._connectivity(
        np.fromfunction(lambda i, j: np.maximum(i, j), (3, 3), dtype=int),
        domain=common.Domain(
            dims=ij_field.domain.dims,
            ranges=ij_field.domain.ranges,
        ),
        codomain=I,
    )

    result = ij_field.premap(max_ij_conn)
    expected = common._field(
        np.asarray([[0.0, 4.0, 8.0], [3.0, 4.0, 8.0], [6.0, 7.0, 8.0]]),
        domain=common.Domain(dims=(I, J), ranges=(UnitRange(0, 3), UnitRange(0, 3))),
    )

    assert result.domain == expected.domain
    assert np.all(result.ndarray == expected.ndarray)


def test_remapping_premap():
    V = Dimension("V")
    E = Dimension("E")

    V_START, V_STOP = 2, 7
    E_START, E_STOP = 0, 10
    v_field = common._field(
        -0.1 * np.arange(V_START, V_STOP),
        domain=common.Domain(dims=(V,), ranges=(UnitRange(V_START, V_STOP),)),
    )
    e2v_conn = common._connectivity(
        np.arange(E_START, E_STOP),
        domain=common.Domain(dims=(E,), ranges=[UnitRange(E_START, E_STOP)]),
        codomain=V,
    )

    result = v_field.premap(e2v_conn)
    expected = common._field(
        -0.1 * np.arange(V_START, V_STOP),
        domain=common.Domain(dims=(E,), ranges=(UnitRange(V_START, V_STOP),)),
    )

    assert result.domain == expected.domain
    assert np.all(result.ndarray == expected.ndarray)


def test_identity_connectivity():
    D0 = Dimension("D0")
    D1 = Dimension("D1")
    D2 = Dimension("D2")

    domain = common.Domain(
        dims=(D0, D1, D2),
        ranges=(common.UnitRange(0, 3), common.UnitRange(0, 4), common.UnitRange(0, 5)),
    )
    codomains = [D0, D1, D2]

    expected = {
        D0: nd_array_field.NumPyArrayConnectivityField.from_array(
            np.array(
                [
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                    [[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]],
                ],
                dtype=int,
            ),
            codomain=D0,
            domain=domain,
        ),
        D1: nd_array_field.NumPyArrayConnectivityField.from_array(
            np.array(
                [
                    [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
                    [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
                    [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
                ],
                dtype=int,
            ),
            codomain=D1,
            domain=domain,
        ),
        D2: nd_array_field.NumPyArrayConnectivityField.from_array(
            np.array(
                [
                    [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
                    [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
                    [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
                ],
                dtype=int,
            ),
            codomain=D2,
            domain=domain,
        ),
    }

    for codomain in codomains:
        result = nd_array_field._identity_connectivity(
            domain, codomain, cls=nd_array_field.NumPyArrayConnectivityField
        )
        assert result.codomain == expected[codomain].codomain
        assert result.domain == expected[codomain].domain
        assert result.dtype == expected[codomain].dtype
        assert np.all(result.ndarray == expected[codomain].ndarray)


@pytest.mark.parametrize(
    "new_dims,field,expected_domain",
    [
        (
            (
                (D0,),
                common._field(
                    np.arange(10), domain=common.Domain(dims=(D0,), ranges=(UnitRange(0, 10),))
                ),
                Domain(dims=(D0,), ranges=(UnitRange(0, 10),)),
            )
        ),
        (
            (
                (D0, D1),
                common._field(
                    np.arange(10), domain=common.Domain(dims=(D0,), ranges=(UnitRange(0, 10),))
                ),
                Domain(dims=(D0, D1), ranges=(UnitRange(0, 10), UnitRange.infinite())),
            )
        ),
        (
            (
                (D0, D1),
                common._field(
                    np.arange(10), domain=common.Domain(dims=(D1,), ranges=(UnitRange(0, 10),))
                ),
                Domain(dims=(D0, D1), ranges=(UnitRange.infinite(), UnitRange(0, 10))),
            )
        ),
        (
            (
                (D0, D1, D2),
                common._field(
                    np.arange(10), domain=common.Domain(dims=(D1,), ranges=(UnitRange(0, 10),))
                ),
                Domain(
                    dims=(D0, D1, D2),
                    ranges=(UnitRange.infinite(), UnitRange(0, 10), UnitRange.infinite()),
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
    [(NamedRange(D0, UnitRange(0, 10)),), common.Domain(dims=(D0,), ranges=(UnitRange(0, 10),))],
)
def test_get_slices_with_named_indices_3d_to_1d(domain_slice):
    field_domain = common.Domain(
        dims=(D0, D1, D2), ranges=(UnitRange(0, 10), UnitRange(0, 10), UnitRange(0, 10))
    )
    slices = _get_slices_from_domain_slice(field_domain, domain_slice)
    assert slices == (slice(0, 10, None), slice(None), slice(None))


def test_get_slices_with_named_index():
    field_domain = common.Domain(
        dims=(D0, D1, D2), ranges=(UnitRange(0, 10), UnitRange(0, 10), UnitRange(0, 10))
    )
    named_index = (NamedRange(D0, UnitRange(0, 10)), (D1, 2), (D2, 3))
    slices = _get_slices_from_domain_slice(field_domain, named_index)
    assert slices == (slice(0, 10, None), 2, 3)


def test_get_slices_invalid_type():
    field_domain = common.Domain(
        dims=(D0, D1, D2), ranges=(UnitRange(0, 10), UnitRange(0, 10), UnitRange(0, 10))
    )
    new_domain = ((D0, "1"),)
    with pytest.raises(ValueError):
        _get_slices_from_domain_slice(field_domain, new_domain)


@pytest.mark.parametrize(
    "domain_slice,expected_dimensions,expected_shape",
    [
        (
            (NamedRange(D0, UnitRange(7, 9)), NamedRange(D1, UnitRange(8, 10))),
            (D0, D1, D2),
            (2, 2, 15),
        ),
        (
            (NamedRange(D0, UnitRange(7, 9)), NamedRange(D2, UnitRange(12, 20))),
            (D0, D1, D2),
            (2, 10, 8),
        ),
        (common.Domain(dims=(D0,), ranges=(UnitRange(7, 9),)), (D0, D1, D2), (2, 10, 15)),
        ((NamedIndex(D0, 8),), (D1, D2), (10, 15)),
        ((NamedIndex(D1, 9),), (D0, D2), (5, 15)),
        ((NamedIndex(D2, 11),), (D0, D1), (5, 10)),
        ((NamedIndex(D0, 8), NamedRange(D1, UnitRange(8, 10))), (D1, D2), (2, 15)),
        (NamedIndex(D0, 5), (D1, D2), (10, 15)),
        (NamedRange(D0, UnitRange(5, 7)), (D0, D1, D2), (2, 10, 15)),
    ],
)
def test_absolute_indexing(domain_slice, expected_dimensions, expected_shape):
    domain = common.Domain(
        dims=(D0, D1, D2), ranges=(UnitRange(5, 10), UnitRange(5, 15), UnitRange(10, 25))
    )
    field = common._field(np.ones((5, 10, 15)), domain=domain)
    indexed_field = field[domain_slice]

    assert isinstance(indexed_field, common.Field)
    assert indexed_field.ndarray.shape == expected_shape
    assert indexed_field.domain.dims == expected_dimensions


def test_absolute_indexing_dim_sliced():
    domain = common.Domain(
        dims=(D0, D1, D2), ranges=(UnitRange(5, 10), UnitRange(5, 15), UnitRange(10, 25))
    )
    field = common._field(np.ones((5, 10, 15)), domain=domain)
    indexed_field_1 = field[D1(8) : D1(10), D0(5) : D0(9)]
    expected = field[
        NamedRange(dim=D0, unit_range=UnitRange(5, 9)),
        NamedRange(dim=D1, unit_range=UnitRange(8, 10)),
    ]

    assert isinstance(indexed_field_1, common.Field)
    assert indexed_field_1 == expected


def test_absolute_indexing_dim_sliced_single_slice():
    domain = common.Domain(
        dims=(D0, D1, D2), ranges=(UnitRange(5, 10), UnitRange(5, 15), UnitRange(10, 25))
    )
    field = common._field(np.ones((5, 10, 15)), domain=domain)
    indexed_field_1 = field[D2(11)]
    indexed_field_2 = field[NamedIndex(D2, 11)]

    assert isinstance(indexed_field_1, common.Field)
    assert indexed_field_1 == indexed_field_2


def test_absolute_indexing_wrong_dim_sliced():
    domain = common.Domain(
        dims=(D0, D1, D2), ranges=(UnitRange(5, 10), UnitRange(5, 15), UnitRange(10, 25))
    )
    field = common._field(np.ones((5, 10, 15)), domain=domain)

    with pytest.raises(IndexError, match="Dimensions slicing mismatch between 'D1' and 'D0'."):
        field[D1(8) : D0(10)]


def test_absolute_indexing_empty_dim_sliced():
    domain = common.Domain(
        dims=(D0, D1, D2), ranges=(UnitRange(5, 10), UnitRange(5, 15), UnitRange(10, 25))
    )
    field = common._field(np.ones((5, 10, 15)), domain=domain)
    with pytest.raises(IndexError, match="Lower bound needs to be specified"):
        field[: D0(10)]


def test_absolute_indexing_value_return():
    domain = common.Domain(dims=(D0, D1), ranges=(UnitRange(10, 20), UnitRange(5, 15)))
    field = common._field(np.reshape(np.arange(100, dtype=np.int32), (10, 10)), domain=domain)

    named_index = (NamedIndex(D0, 12), NamedIndex(D1, 6))
    assert isinstance(field, common.Field)
    value = field[named_index]

    assert isinstance(value, common.Field)
    assert value.as_scalar() == 21


@pytest.mark.parametrize(
    "index, expected_shape, expected_domain",
    [
        (
            (slice(None, 5), slice(None, 2)),
            (5, 2),
            Domain(NamedRange(D0, UnitRange(5, 10)), NamedRange(D1, UnitRange(2, 4))),
        ),
        (
            (slice(None, 5),),
            (5, 10),
            Domain(NamedRange(D0, UnitRange(5, 10)), NamedRange(D1, UnitRange(2, 12))),
        ),
        ((Ellipsis, 1), (10,), Domain(NamedRange(D0, UnitRange(5, 15)))),
        (
            (slice(2, 3), slice(5, 7)),
            (1, 2),
            Domain(NamedRange(D0, UnitRange(7, 8)), NamedRange(D1, UnitRange(7, 9))),
        ),
        ((slice(1, 2), 0), (1,), Domain(NamedRange(D0, UnitRange(6, 7)))),
    ],
)
def test_relative_indexing_slice_2D(index, expected_shape, expected_domain):
    domain = common.Domain(dims=(D0, D1), ranges=(UnitRange(5, 15), UnitRange(2, 12)))
    field = common._field(np.ones((10, 10)), domain=domain)
    indexed_field = field[index]

    assert isinstance(indexed_field, common.Field)
    assert indexed_field.ndarray.shape == expected_shape
    assert indexed_field.domain == expected_domain


@pytest.mark.parametrize(
    "index, expected_shape, expected_domain",
    [
        ((1, slice(None), 2), (15,), Domain(dims=(D1,), ranges=(UnitRange(10, 25),))),
        (
            (slice(None), slice(None), 2),
            (10, 15),
            Domain(dims=(D0, D1), ranges=(UnitRange(5, 15), UnitRange(10, 25))),
        ),
        (
            (slice(None),),
            (10, 15, 10),
            Domain(
                dims=(D0, D1, D2), ranges=(UnitRange(5, 15), UnitRange(10, 25), UnitRange(10, 20))
            ),
        ),
        (
            (slice(None), slice(None), slice(None)),
            (10, 15, 10),
            Domain(
                dims=(D0, D1, D2), ranges=(UnitRange(5, 15), UnitRange(10, 25), UnitRange(10, 20))
            ),
        ),
        (
            (slice(None)),
            (10, 15, 10),
            Domain(
                dims=(D0, D1, D2), ranges=(UnitRange(5, 15), UnitRange(10, 25), UnitRange(10, 20))
            ),
        ),
        ((0, Ellipsis, 0), (15,), Domain(dims=(D1,), ranges=(UnitRange(10, 25),))),
        (
            Ellipsis,
            (10, 15, 10),
            Domain(
                dims=(D0, D1, D2), ranges=(UnitRange(5, 15), UnitRange(10, 25), UnitRange(10, 20))
            ),
        ),
    ],
)
def test_relative_indexing_slice_3D(index, expected_shape, expected_domain):
    domain = common.Domain(
        dims=(D0, D1, D2), ranges=(UnitRange(5, 15), UnitRange(10, 25), UnitRange(10, 20))
    )
    field = common._field(np.ones((10, 15, 10)), domain=domain)
    indexed_field = field[index]

    assert isinstance(indexed_field, common.Field)
    assert indexed_field.ndarray.shape == expected_shape
    assert indexed_field.domain == expected_domain


@pytest.mark.parametrize("index, expected_value", [((1, 0), 10), ((0, 1), 1)])
def test_relative_indexing_value_return(index, expected_value):
    domain = common.Domain(dims=(D0, D1), ranges=(UnitRange(5, 15), UnitRange(2, 12)))
    field = common._field(np.reshape(np.arange(100, dtype=int), (10, 10)), domain=domain)
    indexed_field = field[index]

    assert indexed_field.as_scalar() == expected_value


@pytest.mark.parametrize("lazy_slice", [lambda f: f[13], lambda f: f[:5, :3, :2]])
def test_relative_indexing_out_of_bounds(lazy_slice):
    domain = common.Domain(dims=(D0, D1), ranges=(UnitRange(3, 13), UnitRange(-5, 5)))
    field = common._field(np.ones((10, 10)), domain=domain)

    with pytest.raises((embedded_exceptions.IndexOutOfBounds, IndexError)):
        lazy_slice(field)


@pytest.mark.parametrize("index", [D0, "1", (D0, D1)])
def test_field_unsupported_index(index):
    domain = common.Domain(dims=(D0,), ranges=(UnitRange(0, 10),))
    field = common._field(np.ones((10,)), domain=domain)
    with pytest.raises(IndexError, match="Unsupported index type"):
        field[index]


@pytest.mark.parametrize(
    "index, value",
    [
        ((1, 1), 42.0),
        ((1, slice(None)), np.ones((10,)) * 42.0),
        (
            (1, slice(None)),
            common._field(
                np.ones((10,)) * 42.0, domain=common.Domain(NamedRange(D1, UnitRange(0, 10)))
            ),
        ),
    ],
)
def test_setitem(index, value):
    field = common._field(
        np.arange(100).reshape(10, 10),
        domain=common.Domain(dims=(D0, D1), ranges=(UnitRange(0, 10), UnitRange(0, 10))),
    )

    expected = np.copy(field.asnumpy())
    expected[index] = value.asnumpy() if isinstance(value, common.Field) else value

    field[index] = value

    assert np.allclose(field.ndarray, expected)


def test_setitem_wrong_domain():
    field = common._field(
        np.arange(100).reshape(10, 10),
        domain=common.Domain(dims=(D0, D1), ranges=(UnitRange(0, 10), UnitRange(0, 10))),
    )

    value_incompatible = common._field(
        np.ones((10,)) * 42.0, domain=common.Domain(NamedRange(D1, UnitRange(-5, 5)))
    )

    with pytest.raises(ValueError, match=r"Incompatible 'Domain'.*"):
        field[(1, slice(None))] = value_incompatible


def test_connectivity_field_inverse_image():
    V = Dimension("V")
    E = Dimension("E")

    V_START, V_STOP = 2, 7
    E_START, E_STOP = 0, 10

    e2v_conn = common._connectivity(
        np.roll(np.arange(E_START, E_STOP), 1),
        domain=common.domain([common.named_range((E, (E_START, E_STOP)))]),
        codomain=V,
    )

    # Test range
    image_range = UnitRange(V_START, V_STOP)
    result = e2v_conn.inverse_image(image_range)

    assert len(result) == 1
    assert result[0] == (E, UnitRange(V_START + 1, V_STOP + 1))

    # Test cache
    cached_result = e2v_conn.inverse_image(image_range)
    assert result is cached_result  # If the cache is not used, the result would be a new object

    # Test codomain
    with pytest.raises(ValueError, match="does not match the codomain dimension"):
        e2v_conn.inverse_image(NamedRange(E, UnitRange(1, 2)))


def test_connectivity_field_inverse_image_2d_domain():
    V = Dimension("V")
    C = Dimension("C")
    C2V = Dimension("C2V")

    V_START, V_STOP = 0, 3
    C_START, C_STOP = 0, 3
    C2V_START, C2V_STOP = 0, 3

    c2v_conn = common._connectivity(
        np.asarray([[0, 0, 2], [1, 1, 2], [2, 2, 2]]),
        domain=common.domain(
            [
                common.named_range((C, (C_START, C_STOP))),
                common.named_range((C2V, (C2V_START, C2V_STOP))),
            ]
        ),
        codomain=V,
    )

    # c2v_conn:
    #  ---C2V----
    #  |[[0 0 2]
    #  C [1 1 2]
    #  | [2 2 2]]

    # Test contiguous and non-contiguous ranges.
    # For the 'c2v_conn' defined above, the only valid range including 2
    # is [0, 3). Otherwise, the inverse image would be non-contiguous.
    image_range = UnitRange(V_START, V_STOP)
    result = c2v_conn.inverse_image(image_range)

    assert len(result) == 2
    assert result[0] == (C, UnitRange(C_START, C_STOP))
    assert result[1] == (C2V, UnitRange(C2V_START, C2V_STOP))

    result = c2v_conn.inverse_image(UnitRange(0, 2))
    assert len(result) == 2
    assert result[0] == (C, UnitRange(0, 2))
    assert result[1] == (C2V, UnitRange(0, 2))

    result = c2v_conn.inverse_image(UnitRange(0, 1))
    assert len(result) == 2
    assert result[0] == (C, UnitRange(0, 1))
    assert result[1] == (C2V, UnitRange(0, 2))

    result = c2v_conn.inverse_image(UnitRange(1, 2))
    assert len(result) == 2
    assert result[0] == (C, UnitRange(1, 2))
    assert result[1] == (C2V, UnitRange(0, 2))

    with pytest.raises(ValueError, match="generates non-contiguous dimensions"):
        result = c2v_conn.inverse_image(UnitRange(1, 3))

    with pytest.raises(ValueError, match="generates non-contiguous dimensions"):
        result = c2v_conn.inverse_image(UnitRange(2, 3))


def test_connectivity_field_inverse_image_non_contiguous():
    V = Dimension("V")
    E = Dimension("E")

    V_START, V_STOP = 2, 7
    E_START, E_STOP = 0, 10

    e2v_conn = common._connectivity(
        np.asarray([0, 1, 2, 3, 4, 9, 7, 5, 8, 6]),
        domain=common.domain([common.named_range((E, (E_START, E_STOP)))]),
        codomain=V,
    )

    result = e2v_conn.inverse_image(UnitRange(V_START, 5))
    assert result[0] == (E, UnitRange(V_START, 5))

    with pytest.raises(ValueError, match="generates non-contiguous dimensions"):
        e2v_conn.inverse_image(UnitRange(V_START, 6))

    with pytest.raises(ValueError, match="generates non-contiguous dimensions"):
        e2v_conn.inverse_image(UnitRange(V_START, V_STOP))


def test_connectivity_field_inverse_image_2d_domain_skip_values():
    V = Dimension("V")
    C = Dimension("C")
    C2V = Dimension("C2V")

    V_START, V_STOP = 0, 3
    C_START, C_STOP = 0, 4
    C2V_START, C2V_STOP = 0, 4

    c2v_conn = common._connectivity(
        np.asarray([[-1, 0, 2, -1], [1, 1, 2, 2], [2, 2, -1, -1], [-1, 2, -1, -1]]),
        domain=common.domain(
            [
                common.named_range((C, (C_START, C_STOP))),
                common.named_range((C2V, (C2V_START, C2V_STOP))),
            ]
        ),
        codomain=V,
        skip_value=-1,
    )

    # c2v_conn:
    #  ---C2V---------
    #  |[[-1  0  2 -1]
    #  C [ 1  1  2  2]
    #  | [ 2  2 -1 -1]
    #  | [-1  2 -1 -1]]

    image_range = UnitRange(V_START, V_STOP)
    result = c2v_conn.inverse_image(image_range)

    assert len(result) == 2
    assert result[0] == (C, UnitRange(C_START, C_STOP))
    assert result[1] == (C2V, UnitRange(C2V_START, C2V_STOP))

    result = c2v_conn.inverse_image(UnitRange(0, 2))
    assert len(result) == 2
    assert result[0] == (C, UnitRange(0, 2))
    assert result[1] == (C2V, UnitRange(0, 2))

    result = c2v_conn.inverse_image(UnitRange(0, 1))
    assert len(result) == 2
    assert result[0] == (C, UnitRange(0, 1))
    assert result[1] == (C2V, UnitRange(1, 2))

    result = c2v_conn.inverse_image(UnitRange(1, 2))
    assert len(result) == 2
    assert result[0] == (C, UnitRange(1, 2))
    assert result[1] == (C2V, UnitRange(0, 2))

    with pytest.raises(ValueError, match="generates non-contiguous dimensions"):
        result = c2v_conn.inverse_image(UnitRange(1, 3))

    with pytest.raises(ValueError, match="generates non-contiguous dimensions"):
        result = c2v_conn.inverse_image(UnitRange(2, 3))


@pytest.mark.parametrize(
    "index_array, expected",
    [
        ([0, 0, 1], [(0, 2)]),
        ([0, 1, 0], None),
        ([0, -1, 0], [(0, 3)]),
        ([[1, 1, 1], [1, 0, 0]], [(1, 2), (1, 3)]),
        ([[1, 0, -1], [1, 0, 0]], [(0, 2), (1, 3)]),
    ],
)
def test_hyperslice(index_array, expected):
    index_array = np.asarray(index_array)
    image_range = common.UnitRange(0, 1)
    skip_value = -1

    expected = tuple(slice(*e) for e in expected) if expected is not None else None

    result = nd_array_field._hyperslice(index_array, image_range, np, skip_value)

    assert result == expected


@pytest.mark.parametrize(
    "mask_data, true_data, false_data, expected",
    [
        (
            ([True, False, True, False, True], None),
            ([1, 2, 3, 4, 5], None),
            ([6, 7, 8, 9, 10], None),
            ([1, 7, 3, 9, 5], None),
        ),
        (
            ([True, False, True, False], None),
            ([1, 2, 3, 4, 5], {D0: (-2, 3)}),
            ([6, 7, 8, 9], {D0: (1, 5)}),
            ([3, 6, 5, 8], {D0: (0, 4)}),
        ),
        (
            ([True, False, True, False, True], None),
            ([1, 2, 3, 4, 5], {D0: (-2, 3)}),
            ([6, 7, 8, 9, 10], {D0: (1, 6)}),
            ([3, 6, 5, 8], {D0: (0, 4)}),
        ),
        (
            ([True, False, True, False, True], None),
            ([1, 2, 3, 4, 5], {D0: (-2, 3)}),
            ([6, 7, 8, 9, 10], {D0: (2, 7)}),
            None,
        ),
        (
            # empty result domain
            ([True, False, True, False, True], None),
            ([1, 2, 3, 4, 5], {D0: (-5, 0)}),
            ([6, 7, 8, 9, 10], {D0: (5, 10)}),
            ([], {D0: (0, 0)}),
        ),
        (
            ([True, False, True, False, True], None),
            ([1, 2, 3, 4, 5], {D0: (-4, 1)}),
            ([6, 7, 8, 9, 10], {D0: (5, 10)}),
            ([5], {D0: (0, 1)}),
        ),
        (
            # broadcasting true_field
            ([True, False, True, False, True], {D0: 5}),
            ([1, 2, 3, 4, 5], {D0: 5}),
            ([[6, 11], [7, 12], [8, 13], [9, 14], [10, 15]], {D0: 5, D1: 2}),
            ([[1, 1], [7, 12], [3, 3], [9, 14], [5, 5]], {D0: 5, D1: 2}),
        ),
        (
            ([True, False, True, False, True], None),
            (42, None),
            ([6, 7, 8, 9, 10], None),
            ([42, 7, 42, 9, 42], None),
        ),
        (
            # parts of mask_ranges are concatenated
            ([True, True, False, False], None),
            ([1, 2], {D0: (1, 3)}),
            ([3, 4], {D0: (1, 3)}),
            ([1, 4], {D0: (1, 3)}),
        ),
        (
            # parts of mask_ranges are concatenated and yield non-contiguous domain
            ([True, False, True, False], None),
            ([1, 2], {D0: (0, 2)}),
            ([3, 4], {D0: (2, 4)}),
            None,
        ),
    ],
)
def test_concat_where(
    nd_array_implementation,
    mask_data: tuple[list[bool], Optional[common.DomainLike]],
    true_data: tuple[list[int], Optional[common.DomainLike]],
    false_data: tuple[list[int], Optional[common.DomainLike]],
    expected: Optional[tuple[list[int], Optional[common.DomainLike]]],
):
    mask_lst, mask_domain = mask_data
    true_lst, true_domain = true_data
    false_lst, false_domain = false_data

    mask_field = _make_field_or_scalar(
        mask_lst,
        nd_array_implementation=nd_array_implementation,
        domain=common.domain(mask_domain) if mask_domain is not None else None,
        dtype=bool,
    )
    true_field = _make_field_or_scalar(
        true_lst,
        nd_array_implementation=nd_array_implementation,
        domain=common.domain(true_domain) if true_domain is not None else None,
        dtype=np.int32,
    )
    false_field = _make_field_or_scalar(
        false_lst,
        nd_array_implementation=nd_array_implementation,
        domain=common.domain(false_domain) if false_domain is not None else None,
        dtype=np.int32,
    )

    if expected is None:
        with pytest.raises(embedded_exceptions.NonContiguousDomain):
            nd_array_field._concat_where(mask_field, true_field, false_field)
    else:
        expected_lst, expected_domain_like = expected
        expected_array = np.asarray(expected_lst)
        expected_domain = (
            common.domain(expected_domain_like)
            if expected_domain_like is not None
            else _make_default_domain(expected_array.shape)
        )

        result = nd_array_field._concat_where(mask_field, true_field, false_field)

        assert expected_domain == result.domain
        np.testing.assert_allclose(result.asnumpy(), expected_array)
