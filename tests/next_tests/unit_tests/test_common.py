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
from typing import Optional, Pattern
import numpy as np
import pytest

from gt4py.next import common
from gt4py.next.common import Dimension, DimensionKind, Domain, Infinity, UnitRange, promote_dims


IDim = Dimension("IDim")
ECDim = Dimension("ECDim")
JDim = Dimension("JDim")
KDim = Dimension("KDim", kind=DimensionKind.VERTICAL)


@pytest.fixture
def domain():
    range1 = UnitRange(0, 10)
    range2 = UnitRange(5, 15)
    range3 = UnitRange(20, 30)

    dimensions = (IDim, JDim, KDim)
    ranges = (range1, range2, range3)

    return Domain(dimensions, ranges)


def test_empty_range():
    expected = UnitRange(0, 0)
    assert UnitRange(1, 1) == expected
    assert UnitRange(1, -1) == expected


@pytest.fixture
def rng():
    return UnitRange(-5, 5)


def test_unit_range_length(rng):
    assert rng.start == -5
    assert rng.stop == 5
    assert len(rng) == 10


def test_unit_range_repr(rng):
    assert repr(rng) == "UnitRange(-5, 5)"


def test_unit_range_iter(rng):
    actual = []
    expected = list(range(-5, 5))

    for elt in rng:
        actual.append(elt)

    assert actual == expected


def test_unit_range_get_item(rng):
    assert rng[-1] == 4
    assert rng[0] == -5
    assert rng[0:4] == UnitRange(-5, -1)
    assert rng[-4:] == UnitRange(1, 5)


def test_unit_range_index_error(rng):
    with pytest.raises(IndexError):
        rng[10]


def test_unit_range_slice_error(rng):
    with pytest.raises(ValueError):
        rng[1:2:5]


def test_unit_range_set_intersection(rng):
    with pytest.raises(
        NotImplementedError, match="Can only find the intersection between UnitRange instances."
    ):
        rng & {1, 5}


@pytest.mark.parametrize(
    "rng1, rng2, expected",
    [
        (UnitRange(0, 5), UnitRange(10, 15), UnitRange(0, 0)),
        (UnitRange(0, 5), UnitRange(5, 10), UnitRange(5, 5)),
        (UnitRange(0, 5), UnitRange(3, 7), UnitRange(3, 5)),
        (UnitRange(0, 5), UnitRange(1, 6), UnitRange(1, 5)),
        (UnitRange(0, 5), UnitRange(-5, 5), UnitRange(0, 5)),
        (UnitRange(0, 0), UnitRange(0, 5), UnitRange(0, 0)),
        (UnitRange(0, 0), UnitRange(0, 0), UnitRange(0, 0)),
    ],
)
def test_unit_range_intersection(rng1, rng2, expected):
    result = rng1 & rng2
    assert result == expected


@pytest.mark.parametrize(
    "rng1, rng2, expected",
    [
        (UnitRange(20, Infinity.positive()), UnitRange(10, 15), UnitRange(0, 0)),
        (UnitRange(Infinity.negative(), 0), UnitRange(5, 10), UnitRange(0, 0)),
        (UnitRange(Infinity.negative(), 0), UnitRange(-10, 0), UnitRange(-10, 0)),
        (UnitRange(0, Infinity.positive()), UnitRange(Infinity.negative(), 5), UnitRange(0, 5)),
        (
            UnitRange(Infinity.negative(), 0),
            UnitRange(Infinity.negative(), 5),
            UnitRange(Infinity.negative(), 0),
        ),
        (
            UnitRange(Infinity.negative(), Infinity.positive()),
            UnitRange(Infinity.negative(), Infinity.positive()),
            UnitRange(Infinity.negative(), Infinity.positive()),
        ),
    ],
)
def test_unit_range_infinite_intersection(rng1, rng2, expected):
    result = rng1 & rng2
    assert result == expected


def test_positive_infinity_range():
    pos_inf_range = UnitRange(Infinity.positive(), Infinity.positive())
    assert len(pos_inf_range) == 0


def test_mixed_infinity_range():
    mixed_inf_range = UnitRange(Infinity.negative(), Infinity.positive())
    assert len(mixed_inf_range) == Infinity.positive()


def test_domain_length(domain):
    assert len(domain) == 3


def test_domain_iteration(domain):
    iterated_values = [val for val in domain]
    assert iterated_values == list(zip(domain.dims, domain.ranges))


def test_domain_contains_named_range(domain):
    assert (IDim, UnitRange(0, 10)) in domain
    assert (IDim, UnitRange(-5, 5)) not in domain


@pytest.mark.parametrize(
    "second_domain, expected",
    [
        (
            Domain((IDim, JDim), (UnitRange(2, 12), UnitRange(7, 17))),
            Domain((IDim, JDim, KDim), (UnitRange(2, 10), UnitRange(7, 15), UnitRange(20, 30))),
        ),
        (
            Domain((IDim, KDim), (UnitRange(2, 12), UnitRange(7, 27))),
            Domain((IDim, JDim, KDim), (UnitRange(2, 10), UnitRange(5, 15), UnitRange(20, 27))),
        ),
        (
            Domain((JDim, KDim), (UnitRange(2, 12), UnitRange(4, 27))),
            Domain((IDim, JDim, KDim), (UnitRange(0, 10), UnitRange(5, 12), UnitRange(20, 27))),
        ),
    ],
)
def test_domain_intersection_different_dimensions(domain, second_domain, expected):
    result_domain = domain & second_domain
    print(result_domain)

    assert result_domain == expected


def test_domain_intersection_reversed_dimensions(domain):
    dimensions = (JDim, IDim)
    ranges = (UnitRange(2, 12), UnitRange(7, 17))
    domain2 = Domain(dimensions, ranges)

    with pytest.raises(
        ValueError,
        match="Dimensions can not be promoted. The following dimensions appear in contradicting order: IDim, JDim.",
    ):
        domain & domain2


@pytest.mark.parametrize(
    "index, expected",
    [
        (0, (IDim, UnitRange(0, 10))),
        (1, (JDim, UnitRange(5, 15))),
        (2, (KDim, UnitRange(20, 30))),
        (-1, (KDim, UnitRange(20, 30))),
        (-2, (JDim, UnitRange(5, 15))),
    ],
)
def test_domain_integer_indexing(domain, index, expected):
    result = domain[index]
    assert result == expected


@pytest.mark.parametrize(
    "slice_obj, expected",
    [
        (slice(0, 2), ((IDim, UnitRange(0, 10)), (JDim, UnitRange(5, 15)))),
        (slice(1, None), ((JDim, UnitRange(5, 15)), (KDim, UnitRange(20, 30)))),
    ],
)
def test_domain_slice_indexing(domain, slice_obj, expected):
    result = domain[slice_obj]
    assert isinstance(result, Domain)
    assert len(result) == len(expected)
    assert all(res == exp for res, exp in zip(result, expected))


@pytest.mark.parametrize(
    "index, expected_result",
    [
        (JDim, (JDim, UnitRange(5, 15))),
        (KDim, (KDim, UnitRange(20, 30))),
    ],
)
def test_domain_dimension_indexing(domain, index, expected_result):
    result = domain[index]
    assert result == expected_result


def test_domain_indexing_dimension_missing(domain):
    with pytest.raises(KeyError, match=r"No Dimension of type .* is present in the Domain."):
        domain[ECDim]


def test_domain_indexing_invalid_type(domain):
    with pytest.raises(
        KeyError, match="Invalid index type, must be either int, slice, or Dimension."
    ):
        domain["foo"]


def test_domain_repeat_dims():
    dims = (IDim, JDim, IDim)
    ranges = (UnitRange(0, 5), UnitRange(0, 8), UnitRange(0, 3))
    with pytest.raises(NotImplementedError, match=r"Domain dimensions must be unique, not .*"):
        Domain(dims, ranges)


def test_domain_dims_ranges_length_mismatch():
    with pytest.raises(
        ValueError,
        match=r"Number of provided dimensions \(\d+\) does not match number of provided ranges \(\d+\)",
    ):
        dims = [Dimension("X"), Dimension("Y"), Dimension("Z")]
        ranges = [UnitRange(0, 1), UnitRange(0, 1)]
        Domain(dims=dims, ranges=ranges)


def dimension_promotion_cases() -> (
    list[tuple[list[list[Dimension]], list[Dimension] | None, None | Pattern]]
):
    raw_list = [
        # list of list of dimensions, expected result, expected error message
        ([["I", "J"], ["I"]], ["I", "J"], None),
        ([["I", "J"], ["J"]], ["I", "J"], None),
        ([["I", "J"], ["J", "K"]], ["I", "J", "K"], None),
        (
            [["I", "J"], ["J", "I"]],
            None,
            r"The following dimensions appear in contradicting order: I, J.",
        ),
        (
            [["I", "K"], ["J", "K"]],
            None,
            r"Could not determine order of the following dimensions: I, J",
        ),
    ]
    # transform dimension names into Dimension objects
    return [
        (
            [[Dimension(el) for el in arg] for arg in args],
            [Dimension(el) for el in result] if result else result,
            msg,
        )
        for args, result, msg in raw_list
    ]


@pytest.mark.parametrize("dim_list,expected_result,expected_error_msg", dimension_promotion_cases())
def test_dimension_promotion(
    dim_list: list[list[Dimension]],
    expected_result: Optional[list[Dimension]],
    expected_error_msg: Optional[str],
):
    if expected_result:
        assert promote_dims(*dim_list) == expected_result
    else:
        with pytest.raises(Exception) as exc_info:
            promote_dims(*dim_list)

        assert exc_info.match(expected_error_msg)


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
def test_binary_operations(op_func, expected_result):
    cf1 = common.ConstantField(10)
    cf2 = common.ConstantField(20)
    result = op_func(cf1, cf2)
    assert result.value == expected_result


@pytest.mark.parametrize(
    "cf1,cf2,expected",
    [
        (common.ConstantField(10.0), common.ConstantField(20), 30.0),
        (common.ConstantField(10.0), 10, 20.0),
    ],
)
def test_constant_field_incompatible_value_type(cf1, cf2, expected):
    res = cf1 + cf2
    assert res.value == expected
    assert res.dtype == float


def test_constant_field_getitem_domain():
    cf = common.ConstantField(10)
    domain = common.Domain(dims=(IDim,), ranges=(UnitRange(0, 10),))
    result = cf[domain]
    assert isinstance(result.domain, Domain)


def test_constant_field_getitem_named_range():
    cf = common.ConstantField(10)
    nr = ((IDim, UnitRange(0, 10)),)
    result = cf[nr]
    assert isinstance(result.domain, Domain)


def test_constant_field_array():
    cf = common.ConstantField(10)
    nr = ((IDim, UnitRange(0, 5)),(JDim, UnitRange(-3, 13)))
    result = cf[nr]
    assert result.ndarray.shape == (5, 16)
    assert np.all(result.ndarray == 10)
