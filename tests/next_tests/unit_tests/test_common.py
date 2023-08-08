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
from typing import Optional, Pattern

import pytest

from gt4py.next.common import UnitRange, Dimension, Domain, DimensionKind, Infinity, promote_dims

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
    with pytest.raises(NotImplementedError, match="Can only find the intersection between UnitRange instances."):
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


def test_domain_intersection_different_dimensions(domain):
    dimensions = (IDim, JDim)
    ranges = (UnitRange(2, 12), UnitRange(7, 17))
    domain2 = Domain(dimensions, ranges)
    result_domain = domain & domain2

    assert len(result_domain) == 3
    assert all(isinstance(r, UnitRange) for r in result_domain.ranges)
    assert result_domain.dims == [IDim, JDim, KDim]

    assert result_domain.ranges[0] == UnitRange(
        2, 10
    )  # Intersection of domain.range1 and domain2.range1
    assert result_domain.ranges[1] == UnitRange(
        7, 15
    )  # Intersection of domain.range2 and domain2.range2
    assert result_domain.ranges[2] == UnitRange(20, 30)  # Broadcasting on missing dimension


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
