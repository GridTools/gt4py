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

import pytest

from gt4py.next.common import (
    Dimension,
    DimensionKind,
    Domain,
    Infinity,
    NamedRange,
    UnitRange,
    domain,
    named_range,
    promote_dims,
    unit_range,
)


IDim = Dimension("IDim")
ECDim = Dimension("ECDim")
JDim = Dimension("JDim")
KDim = Dimension("KDim", kind=DimensionKind.VERTICAL)


@pytest.fixture
def a_domain():
    return Domain((IDim, UnitRange(0, 10)), (JDim, UnitRange(5, 15)), (KDim, UnitRange(20, 30)))


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


@pytest.mark.parametrize("rng_like", [(2, 4), range(2, 4), UnitRange(2, 4)])
def test_unit_range_like(rng_like):
    assert unit_range(rng_like) == UnitRange(2, 4)


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


@pytest.mark.parametrize(
    "op, rng1, rng2, expected",
    [
        (operator.le, UnitRange(-1, 2), UnitRange(-2, 3), True),
        (operator.le, UnitRange(-1, 2), {-1, 0, 1}, True),
        (operator.le, UnitRange(-1, 2), {-1, 0}, False),
        (operator.le, UnitRange(-1, 2), {-2, -1, 0, 1, 2}, True),
        (operator.le, UnitRange(Infinity.negative(), 2), UnitRange(Infinity.negative(), 3), True),
        (operator.le, UnitRange(Infinity.negative(), 2), {1, 2, 3}, False),
    ],
)
def test_range_comparison(op, rng1, rng2, expected):
    assert op(rng1, rng2) == expected


@pytest.mark.parametrize(
    "named_rng_like",
    [
        (IDim, (2, 4)),
        (IDim, range(2, 4)),
        (IDim, UnitRange(2, 4)),
    ],
)
def test_named_range_like(named_rng_like):
    assert named_range(named_rng_like) == (IDim, UnitRange(2, 4))


def test_domain_length(a_domain):
    assert len(a_domain) == 3


@pytest.mark.parametrize(
    "domain_like",
    [
        (Domain(dims=(IDim, JDim), ranges=(UnitRange(2, 4), UnitRange(3, 5)))),
        ((IDim, (2, 4)), (JDim, (3, 5))),
        ({IDim: (2, 4), JDim: (3, 5)}),
    ],
)
def test_domain_like(domain_like):
    assert domain(domain_like) == Domain(
        dims=(IDim, JDim), ranges=(UnitRange(2, 4), UnitRange(3, 5))
    )


def test_domain_iteration(a_domain):
    iterated_values = [val for val in a_domain]
    assert iterated_values == list(zip(a_domain.dims, a_domain.ranges))


def test_domain_contains_named_range(a_domain):
    assert (IDim, UnitRange(0, 10)) in a_domain
    assert (IDim, UnitRange(-5, 5)) not in a_domain


@pytest.mark.parametrize(
    "second_domain, expected",
    [
        (
            Domain(dims=(IDim, JDim), ranges=(UnitRange(2, 12), UnitRange(7, 17))),
            Domain(
                dims=(IDim, JDim, KDim),
                ranges=(UnitRange(2, 10), UnitRange(7, 15), UnitRange(20, 30)),
            ),
        ),
        (
            Domain(dims=(IDim, KDim), ranges=(UnitRange(2, 12), UnitRange(7, 27))),
            Domain(
                dims=(IDim, JDim, KDim),
                ranges=(UnitRange(2, 10), UnitRange(5, 15), UnitRange(20, 27)),
            ),
        ),
        (
            Domain(dims=(JDim, KDim), ranges=(UnitRange(2, 12), UnitRange(4, 27))),
            Domain(
                dims=(IDim, JDim, KDim),
                ranges=(UnitRange(0, 10), UnitRange(5, 12), UnitRange(20, 27)),
            ),
        ),
    ],
)
def test_domain_intersection_different_dimensions(a_domain, second_domain, expected):
    result_domain = a_domain & second_domain
    print(result_domain)

    assert result_domain == expected


def test_domain_intersection_reversed_dimensions(a_domain):
    domain2 = Domain(dims=(JDim, IDim), ranges=(UnitRange(2, 12), UnitRange(7, 17)))

    with pytest.raises(
        ValueError,
        match="Dimensions can not be promoted. The following dimensions appear in contradicting order: IDim, JDim.",
    ):
        a_domain & domain2


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
def test_domain_integer_indexing(a_domain, index, expected):
    result = a_domain[index]
    assert result == expected


@pytest.mark.parametrize(
    "slice_obj, expected",
    [
        (slice(0, 2), ((IDim, UnitRange(0, 10)), (JDim, UnitRange(5, 15)))),
        (slice(1, None), ((JDim, UnitRange(5, 15)), (KDim, UnitRange(20, 30)))),
    ],
)
def test_domain_slice_indexing(a_domain, slice_obj, expected):
    result = a_domain[slice_obj]
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
def test_domain_dimension_indexing(a_domain, index, expected_result):
    result = a_domain[index]
    assert result == expected_result


def test_domain_indexing_dimension_missing(a_domain):
    with pytest.raises(KeyError, match=r"No Dimension of type .* is present in the Domain."):
        a_domain[ECDim]


def test_domain_indexing_invalid_type(a_domain):
    with pytest.raises(
        KeyError, match="Invalid index type, must be either int, slice, or Dimension."
    ):
        a_domain["foo"]


def test_domain_repeat_dims():
    dims = (IDim, JDim, IDim)
    ranges = (UnitRange(0, 5), UnitRange(0, 8), UnitRange(0, 3))
    with pytest.raises(NotImplementedError, match=r"Domain dimensions must be unique, not .*"):
        Domain(dims=dims, ranges=ranges)


def test_domain_dims_ranges_length_mismatch():
    with pytest.raises(
        ValueError,
        match=r"Number of provided dimensions \(\d+\) does not match number of provided ranges \(\d+\)",
    ):
        dims = [Dimension("X"), Dimension("Y"), Dimension("Z")]
        ranges = [UnitRange(0, 1), UnitRange(0, 1)]
        Domain(dims=dims, ranges=ranges)


def test_domain_dim_index():
    dims = [Dimension("X"), Dimension("Y"), Dimension("Z")]
    ranges = [UnitRange(0, 1), UnitRange(0, 1), UnitRange(0, 1)]
    domain = Domain(dims=dims, ranges=ranges)

    domain.dim_index(Dimension("Y")) == 1

    domain.dim_index(Dimension("Foo")) == None


def test_domain_pop():
    dims = [Dimension("X"), Dimension("Y"), Dimension("Z")]
    ranges = [UnitRange(0, 1), UnitRange(0, 1), UnitRange(0, 1)]
    domain = Domain(dims=dims, ranges=ranges)

    domain.pop(Dimension("X")) == Domain(dims=dims[1:], ranges=ranges[1:])

    domain.pop(0) == Domain(dims=dims[1:], ranges=ranges[1:])

    domain.pop(-1) == Domain(dims=dims[:-1], ranges=ranges[:-1])


@pytest.mark.parametrize(
    "index, named_ranges, domain, expected",
    [
        # Valid index and named ranges
        (
            0,
            [(Dimension("X"), UnitRange(100, 110))],
            Domain(
                (Dimension("I"), UnitRange(0, 10)),
                (Dimension("J"), UnitRange(0, 10)),
                (Dimension("K"), UnitRange(0, 10)),
            ),
            Domain(
                (Dimension("X"), UnitRange(100, 110)),
                (Dimension("J"), UnitRange(0, 10)),
                (Dimension("K"), UnitRange(0, 10)),
            ),
        ),
        (
            1,
            [(Dimension("X"), UnitRange(100, 110))],
            Domain(
                (Dimension("I"), UnitRange(0, 10)),
                (Dimension("J"), UnitRange(0, 10)),
                (Dimension("K"), UnitRange(0, 10)),
            ),
            Domain(
                (Dimension("I"), UnitRange(0, 10)),
                (Dimension("X"), UnitRange(100, 110)),
                (Dimension("K"), UnitRange(0, 10)),
            ),
        ),
        (
            -1,
            [(Dimension("X"), UnitRange(100, 110))],
            Domain(
                (Dimension("I"), UnitRange(0, 10)),
                (Dimension("J"), UnitRange(0, 10)),
                (Dimension("K"), UnitRange(0, 10)),
            ),
            Domain(
                (Dimension("I"), UnitRange(0, 10)),
                (Dimension("J"), UnitRange(0, 10)),
                (Dimension("X"), UnitRange(100, 110)),
            ),
        ),
        (
            Dimension("J"),
            [(Dimension("X"), UnitRange(100, 110)), (Dimension("Z"), UnitRange(100, 110))],
            Domain(
                (Dimension("I"), UnitRange(0, 10)),
                (Dimension("J"), UnitRange(0, 10)),
                (Dimension("K"), UnitRange(0, 10)),
            ),
            Domain(
                (Dimension("I"), UnitRange(0, 10)),
                (Dimension("X"), UnitRange(100, 110)),
                (Dimension("Z"), UnitRange(100, 110)),
                (Dimension("K"), UnitRange(0, 10)),
            ),
        ),
        # Invalid indices
        (
            3,
            [(Dimension("X"), UnitRange(100, 110))],
            Domain(
                (Dimension("I"), UnitRange(0, 10)),
                (Dimension("J"), UnitRange(0, 10)),
                (Dimension("K"), UnitRange(0, 10)),
            ),
            IndexError,
        ),
        (
            -4,
            [(Dimension("X"), UnitRange(100, 110))],
            Domain(
                (Dimension("I"), UnitRange(0, 10)),
                (Dimension("J"), UnitRange(0, 10)),
                (Dimension("K"), UnitRange(0, 10)),
            ),
            IndexError,
        ),
        (
            Dimension("Foo"),
            [(Dimension("X"), UnitRange(100, 110))],
            Domain(
                (Dimension("I"), UnitRange(0, 10)),
                (Dimension("J"), UnitRange(0, 10)),
                (Dimension("K"), UnitRange(0, 10)),
            ),
            ValueError,
        ),
    ],
)
def test_domain_replace(index, named_ranges, domain, expected):
    if expected is ValueError:
        with pytest.raises(ValueError):
            domain.replace(index, *named_ranges)
    elif expected is IndexError:
        with pytest.raises(IndexError):
            domain.replace(index, *named_ranges)
    else:
        new_domain = domain.replace(index, *named_ranges)
        assert new_domain == expected


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
