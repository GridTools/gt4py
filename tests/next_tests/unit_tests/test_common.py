# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import operator
from typing import Optional, Pattern

import pytest
import re

from gt4py import next as gtx
import gt4py.next.common as common
from gt4py.next.common import (
    Dimension,
    DimensionKind,
    Domain,
    Infinity,
    UnitRange,
    domain,
    named_range,
    NamedRange,
    promote_dims,
    unit_range,
)

C2E = Dimension("C2E", kind=DimensionKind.LOCAL)
V2E = Dimension("V2E", kind=DimensionKind.LOCAL)
E2V = Dimension("E2V", kind=DimensionKind.LOCAL)
E2C = Dimension("E2C", kind=DimensionKind.LOCAL)
E2C2V = Dimension("E2C2V", kind=DimensionKind.LOCAL)
ECDim = Dimension("ECDim")
IDim = Dimension("IDim")
JDim = Dimension("JDim")
KDim = Dimension("KDim", kind=DimensionKind.VERTICAL)


@pytest.fixture
def a_domain():
    return Domain(
        NamedRange(IDim, UnitRange(0, 10)),
        NamedRange(JDim, UnitRange(5, 15)),
        NamedRange(KDim, UnitRange(20, 30)),
    )


@pytest.fixture(params=[Infinity.POSITIVE, Infinity.NEGATIVE])
def unbounded(request):
    yield request.param


def test_unbounded_add_sub(unbounded):
    assert unbounded + 1 == unbounded
    assert unbounded - 1 == unbounded


@pytest.mark.parametrize("value", [-1, 0, 1])
@pytest.mark.parametrize("op", [operator.le, operator.lt])
def test_unbounded_comparison_less(value, op):
    assert not op(Infinity.POSITIVE, value)
    assert op(value, Infinity.POSITIVE)

    assert op(Infinity.NEGATIVE, value)
    assert not op(value, Infinity.NEGATIVE)

    assert op(Infinity.NEGATIVE, Infinity.POSITIVE)


@pytest.mark.parametrize("value", [-1, 0, 1])
@pytest.mark.parametrize("op", [operator.ge, operator.gt])
def test_unbounded_comparison_greater(value, op):
    assert op(Infinity.POSITIVE, value)
    assert not op(value, Infinity.POSITIVE)

    assert not op(Infinity.NEGATIVE, value)
    assert op(value, Infinity.NEGATIVE)

    assert not op(Infinity.NEGATIVE, Infinity.POSITIVE)


def test_unbounded_eq(unbounded):
    assert unbounded == unbounded
    assert unbounded <= unbounded
    assert unbounded >= unbounded
    assert not unbounded < unbounded
    assert not unbounded > unbounded


@pytest.mark.parametrize("value", [-1, 0, 1])
def test_unbounded_max_min(value):
    assert max(Infinity.POSITIVE, value) == Infinity.POSITIVE
    assert min(Infinity.POSITIVE, value) == value
    assert max(Infinity.NEGATIVE, value) == value
    assert min(Infinity.NEGATIVE, value) == Infinity.NEGATIVE


@pytest.mark.parametrize("empty_range", [UnitRange(1, 0), UnitRange(1, -1)])
def test_empty_range(empty_range):
    expected = UnitRange(0, 0)

    assert empty_range == expected
    assert empty_range.is_empty()


@pytest.fixture
def rng():
    return UnitRange(-5, 5)


def test_unit_range_length(rng):
    assert rng.start == -5
    assert rng.stop == 5
    assert len(rng) == 10


@pytest.mark.parametrize(
    "rng_like, expected",
    [
        ((2, 4), UnitRange(2, 4)),
        (range(2, 4), UnitRange(2, 4)),
        (UnitRange(2, 4), UnitRange(2, 4)),
        ((None, None), UnitRange(Infinity.NEGATIVE, Infinity.POSITIVE)),
        ((2, None), UnitRange(2, Infinity.POSITIVE)),
        ((None, 4), UnitRange(Infinity.NEGATIVE, 4)),
        (None, UnitRange(Infinity.NEGATIVE, Infinity.POSITIVE)),
    ],
)
def test_unit_range_like(rng_like, expected):
    assert unit_range(rng_like) == expected


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
        (UnitRange(20, Infinity.POSITIVE), UnitRange(10, 15), UnitRange(0, 0)),
        (UnitRange(Infinity.NEGATIVE, 0), UnitRange(5, 10), UnitRange(0, 0)),
        (UnitRange(Infinity.NEGATIVE, 0), UnitRange(-10, 0), UnitRange(-10, 0)),
        (UnitRange(0, Infinity.POSITIVE), UnitRange(Infinity.NEGATIVE, 5), UnitRange(0, 5)),
        (
            UnitRange(Infinity.NEGATIVE, 0),
            UnitRange(Infinity.NEGATIVE, 5),
            UnitRange(Infinity.NEGATIVE, 0),
        ),
        (
            UnitRange(Infinity.NEGATIVE, Infinity.POSITIVE),
            UnitRange(Infinity.NEGATIVE, Infinity.POSITIVE),
            UnitRange(Infinity.NEGATIVE, Infinity.POSITIVE),
        ),
    ],
)
def test_unit_range_unbounded_intersection(rng1, rng2, expected):
    result = rng1 & rng2
    assert result == expected


@pytest.mark.parametrize(
    "rng",
    [
        UnitRange(Infinity.NEGATIVE, 0),
        UnitRange(0, Infinity.POSITIVE),
        UnitRange(Infinity.NEGATIVE, Infinity.POSITIVE),
    ],
)
def test_positive_infinite_range_len(rng):
    with pytest.raises(ValueError, match=r".*open.*"):
        len(rng)


def test_range_contains():
    assert 1 in UnitRange(0, 2)
    assert 1 not in UnitRange(0, 1)
    assert 1 in UnitRange(0, Infinity.POSITIVE)
    assert 1 in UnitRange(Infinity.NEGATIVE, 2)
    assert 1 in UnitRange(Infinity.NEGATIVE, Infinity.POSITIVE)
    assert "s" not in UnitRange(Infinity.NEGATIVE, Infinity.POSITIVE)


@pytest.mark.parametrize(
    "op, rng1, rng2, expected",
    [
        (operator.le, UnitRange(-1, 2), UnitRange(-2, 3), True),
        (operator.le, UnitRange(Infinity.NEGATIVE, 2), UnitRange(Infinity.NEGATIVE, 3), True),
        (operator.ge, UnitRange(-2, 3), UnitRange(-1, 2), True),
        (operator.ge, UnitRange(Infinity.NEGATIVE, 3), UnitRange(Infinity.NEGATIVE, 2), True),
        (operator.lt, UnitRange(-1, 2), UnitRange(-2, 2), True),
        (operator.lt, UnitRange(-2, 1), UnitRange(-2, 2), True),
        (operator.lt, UnitRange(Infinity.NEGATIVE, 2), UnitRange(Infinity.NEGATIVE, 3), True),
        (operator.gt, UnitRange(-2, 2), UnitRange(-1, 2), True),
        (operator.gt, UnitRange(-2, 2), UnitRange(-2, 1), True),
        (operator.gt, UnitRange(Infinity.NEGATIVE, 3), UnitRange(Infinity.NEGATIVE, 2), True),
        (operator.eq, UnitRange(Infinity.NEGATIVE, 2), UnitRange(Infinity.NEGATIVE, 2), True),
        (operator.ne, UnitRange(Infinity.NEGATIVE, 2), UnitRange(Infinity.NEGATIVE, 3), True),
        (operator.ne, UnitRange(Infinity.NEGATIVE, 2), UnitRange(Infinity.NEGATIVE, 2), False),
    ],
)
def test_range_comparison(op, rng1, rng2, expected):
    assert op(rng1, rng2) == expected


@pytest.mark.parametrize(
    "named_rng_like", [(IDim, (2, 4)), (IDim, range(2, 4)), (IDim, UnitRange(2, 4))]
)
def test_named_range_like(named_rng_like):
    assert named_range(named_rng_like) == (IDim, UnitRange(2, 4))


def test_domain_length(a_domain):
    assert len(a_domain) == 3


@pytest.mark.parametrize(
    "empty_domain, expected",
    [
        (Domain(), False),
        (Domain(NamedRange(IDim, UnitRange(0, 10))), False),
        (Domain(NamedRange(IDim, UnitRange(0, 0))), True),
        (Domain(NamedRange(IDim, UnitRange(0, 0)), NamedRange(JDim, UnitRange(0, 1))), True),
        (Domain(NamedRange(IDim, UnitRange(0, 1)), NamedRange(JDim, UnitRange(0, 0))), True),
    ],
)
def test_empty_domain(empty_domain, expected):
    assert empty_domain.is_empty() == expected


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
    "index, expected_result", [(JDim, (JDim, UnitRange(5, 15))), (KDim, (KDim, UnitRange(20, 30)))]
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


def test_domain_slice_at():
    # Create a sample domain
    domain = Domain(
        NamedRange(IDim, UnitRange(0, 10)),
        NamedRange(JDim, UnitRange(5, 15)),
        NamedRange(KDim, UnitRange(20, 30)),
    )

    # Test indexing with slices
    result = domain.slice_at[slice(2, 5), slice(5, 7), slice(7, 10)]
    expected_result = Domain(
        NamedRange(IDim, UnitRange(2, 5)),
        NamedRange(JDim, UnitRange(10, 12)),
        NamedRange(KDim, UnitRange(27, 30)),
    )
    assert result == expected_result

    # Test indexing with out-of-range slices
    result = domain.slice_at[slice(2, 15), slice(5, 7), slice(7, 10)]
    expected_result = Domain(
        NamedRange(IDim, UnitRange(2, 10)),
        NamedRange(JDim, UnitRange(10, 12)),
        NamedRange(KDim, UnitRange(27, 30)),
    )
    assert result == expected_result

    # Test indexing with incorrect types
    with pytest.raises(TypeError):
        domain.slice_at["a", 7, 25]

    # Test indexing with incorrect number of indices
    with pytest.raises(ValueError, match="not match the number of dimensions"):
        domain.slice_at[slice(2, 5), slice(7, 10)]


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
            [NamedRange(Dimension("X"), UnitRange(100, 110))],
            Domain(
                NamedRange(Dimension("I"), UnitRange(0, 10)),
                NamedRange(Dimension("J"), UnitRange(0, 10)),
                NamedRange(Dimension("K"), UnitRange(0, 10)),
            ),
            Domain(
                NamedRange(Dimension("X"), UnitRange(100, 110)),
                NamedRange(Dimension("J"), UnitRange(0, 10)),
                NamedRange(Dimension("K"), UnitRange(0, 10)),
            ),
        ),
        (
            1,
            [NamedRange(Dimension("X"), UnitRange(100, 110))],
            Domain(
                NamedRange(Dimension("I"), UnitRange(0, 10)),
                NamedRange(Dimension("J"), UnitRange(0, 10)),
                NamedRange(Dimension("K"), UnitRange(0, 10)),
            ),
            Domain(
                NamedRange(Dimension("I"), UnitRange(0, 10)),
                NamedRange(Dimension("X"), UnitRange(100, 110)),
                NamedRange(Dimension("K"), UnitRange(0, 10)),
            ),
        ),
        (
            -1,
            [NamedRange(Dimension("X"), UnitRange(100, 110))],
            Domain(
                NamedRange(Dimension("I"), UnitRange(0, 10)),
                NamedRange(Dimension("J"), UnitRange(0, 10)),
                NamedRange(Dimension("K"), UnitRange(0, 10)),
            ),
            Domain(
                NamedRange(Dimension("I"), UnitRange(0, 10)),
                NamedRange(Dimension("J"), UnitRange(0, 10)),
                NamedRange(Dimension("X"), UnitRange(100, 110)),
            ),
        ),
        (
            Dimension("J"),
            [
                NamedRange(Dimension("X"), UnitRange(100, 110)),
                NamedRange(Dimension("Z"), UnitRange(100, 110)),
            ],
            Domain(
                NamedRange(Dimension("I"), UnitRange(0, 10)),
                NamedRange(Dimension("J"), UnitRange(0, 10)),
                NamedRange(Dimension("K"), UnitRange(0, 10)),
            ),
            Domain(
                NamedRange(Dimension("I"), UnitRange(0, 10)),
                NamedRange(Dimension("X"), UnitRange(100, 110)),
                NamedRange(Dimension("Z"), UnitRange(100, 110)),
                NamedRange(Dimension("K"), UnitRange(0, 10)),
            ),
        ),
        # Invalid indices
        (
            3,
            [NamedRange(Dimension("X"), UnitRange(100, 110))],
            Domain(
                NamedRange(Dimension("I"), UnitRange(0, 10)),
                NamedRange(Dimension("J"), UnitRange(0, 10)),
                NamedRange(Dimension("K"), UnitRange(0, 10)),
            ),
            IndexError,
        ),
        (
            -4,
            [NamedRange(Dimension("X"), UnitRange(100, 110))],
            Domain(
                NamedRange(Dimension("I"), UnitRange(0, 10)),
                NamedRange(Dimension("J"), UnitRange(0, 10)),
                NamedRange(Dimension("K"), UnitRange(0, 10)),
            ),
            IndexError,
        ),
        (
            Dimension("Foo"),
            [NamedRange(Dimension("X"), UnitRange(100, 110))],
            Domain(
                NamedRange(Dimension("I"), UnitRange(0, 10)),
                NamedRange(Dimension("J"), UnitRange(0, 10)),
                NamedRange(Dimension("K"), UnitRange(0, 10)),
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
        ([[IDim, JDim], [IDim]], [IDim, JDim], None),
        ([[JDim], [IDim, JDim]], [IDim, JDim], None),
        ([[JDim, KDim], [IDim, JDim]], [IDim, JDim, KDim], None),
        (
            [[IDim, JDim], [JDim, IDim]],
            None,
            "Dimensions 'JDim[horizontal], IDim[horizontal]' are not ordered correctly, expected 'IDim[horizontal], JDim[horizontal]'.",
        ),
        ([[JDim, KDim], [IDim, KDim]], [IDim, JDim, KDim], None),
        (
            [[KDim, JDim], [IDim, KDim]],
            None,
            "Dimensions 'KDim[vertical], JDim[horizontal]' are not ordered correctly, expected 'JDim[horizontal], KDim[vertical]'.",
        ),
        (
            [[JDim, V2E], [IDim, E2C2V, KDim]],
            None,
            "There are more than one dimension with DimensionKind 'LOCAL'.",
        ),
        ([[JDim, V2E], [IDim, KDim]], [IDim, JDim, V2E, KDim], None),
    ]
    return [
        ([[el for el in arg] for arg in args], [el for el in result] if result else result, msg)
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

        assert exc_info.match(re.escape(expected_error_msg))


class TestCartesianConnectivity:
    def test_for_translation(self):
        offset = 5
        I = common.Dimension("I")

        result = common.CartesianConnectivity.for_translation(I, offset)
        assert isinstance(result, common.CartesianConnectivity)
        assert result.domain_dim == I
        assert result.codomain == I
        assert result.offset == offset

    def test_for_relocation(self):
        I = common.Dimension("I")
        I_half = common.Dimension("I_half")

        result = common.CartesianConnectivity.for_relocation(I, I_half)
        assert isinstance(result, common.CartesianConnectivity)
        assert result.domain_dim == I_half
        assert result.codomain == I
        assert result.offset == 0
