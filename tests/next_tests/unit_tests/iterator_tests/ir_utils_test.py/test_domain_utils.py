# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import domain_utils, ir_makers as im

I = common.Dimension("I")
J = common.Dimension("J")

a_range = domain_utils.SymbolicRange(0, 10)
another_range = domain_utils.SymbolicRange(5, 15)
infinity_range = domain_utils.SymbolicRange(
    itir.InfinityLiteral.NEGATIVE, itir.InfinityLiteral.POSITIVE
)
# the two next ranges are complement of each other (which is used in tests)
right_infinity_range = domain_utils.SymbolicRange(0, itir.InfinityLiteral.POSITIVE)
left_infinity_range = domain_utils.SymbolicRange(itir.InfinityLiteral.NEGATIVE, 0)


def _make_domain(i: int):
    return domain_utils.SymbolicDomain(
        grid_type=common.GridType.CARTESIAN,
        ranges={I: domain_utils.SymbolicRange(im.ref(f"start{i}"), im.ref(f"end{i}"))},
    )


def test_symbolic_range():
    with pytest.raises(AssertionError):
        domain_utils.SymbolicRange(itir.InfinityLiteral.POSITIVE, 0)
    with pytest.raises(AssertionError):
        domain_utils.SymbolicRange(0, itir.InfinityLiteral.NEGATIVE)


def test_domain_union():
    domain0 = _make_domain(0)
    domain1 = _make_domain(1)
    domain2 = _make_domain(2)

    expected = domain_utils.SymbolicDomain(
        grid_type=common.GridType.CARTESIAN,
        ranges={
            I: domain_utils.SymbolicRange(
                im.minimum(im.minimum(im.ref("start0"), im.ref("start1")), im.ref("start2")),
                im.maximum(im.maximum(im.ref("end0"), im.ref("end1")), im.ref("end2")),
            )
        },
    )
    assert expected == domain_utils.domain_union(domain0, domain1, domain2)


def test_domain_intersection():
    domain0 = _make_domain(0)
    domain1 = _make_domain(1)
    domain2 = _make_domain(2)

    expected = domain_utils.SymbolicDomain(
        grid_type=common.GridType.CARTESIAN,
        ranges={
            I: domain_utils.SymbolicRange(
                im.maximum(im.maximum(im.ref("start0"), im.ref("start1")), im.ref("start2")),
                im.minimum(im.minimum(im.ref("end0"), im.ref("end1")), im.ref("end2")),
            )
        },
    )
    assert expected == domain_utils.domain_intersection(domain0, domain1, domain2)


@pytest.mark.parametrize(
    "ranges, expected",
    [
        ({I: a_range}, None),
        ({I: infinity_range}, None),
        ({I: a_range, J: right_infinity_range}, None),
        (
            {I: right_infinity_range, J: left_infinity_range},
            {I: left_infinity_range, J: right_infinity_range},
        ),
    ],
)
def test_domain_complement(ranges, expected):
    if expected is None:
        with pytest.raises(AssertionError):
            domain_utils.domain_complement(
                domain_utils.SymbolicDomain(grid_type=common.GridType.CARTESIAN, ranges=ranges)
            )
    else:
        assert domain_utils.domain_complement(
            domain_utils.SymbolicDomain(grid_type=common.GridType.CARTESIAN, ranges=ranges)
        ) == domain_utils.SymbolicDomain(grid_type=common.GridType.CARTESIAN, ranges=expected)


@pytest.mark.parametrize(
    "testee_ranges, dimensions, expected_ranges",
    [
        ({I: a_range}, [I, J], {I: a_range, J: infinity_range}),
        ({I: a_range}, [J, I], {I: a_range, J: infinity_range}),
        ({I: a_range}, [I], {I: a_range}),
        ({I: a_range}, [J], None),
        ({I: a_range, J: another_range}, [J], None),
    ],
)
def test_promote_domain(testee_ranges, dimensions, expected_ranges):
    testee = domain_utils.SymbolicDomain(grid_type=common.GridType.CARTESIAN, ranges=testee_ranges)
    if expected_ranges is None:
        with pytest.raises(AssertionError):
            domain_utils.promote_domain(testee, dimensions)
    else:
        expected = domain_utils.SymbolicDomain(
            grid_type=common.GridType.CARTESIAN, ranges=expected_ranges
        )
        promoted = domain_utils.promote_domain(testee, dimensions)
        assert promoted == expected


def test_is_finite_symbolic_range():
    assert not domain_utils.is_finite(infinity_range)
    assert not domain_utils.is_finite(left_infinity_range)
    assert not domain_utils.is_finite(right_infinity_range)
    assert domain_utils.is_finite(a_range)


@pytest.mark.parametrize(
    "ranges, expected",
    [
        ({I: a_range, J: a_range}, True),
        ({I: a_range, J: another_range}, True),
        ({I: right_infinity_range, J: a_range}, False),
        ({I: a_range, J: right_infinity_range}, False),
    ],
)
def test_is_finite_symbolic_domain(ranges, expected):
    assert (
        domain_utils.is_finite(
            domain_utils.SymbolicDomain(grid_type=common.GridType.CARTESIAN, ranges=ranges)
        )
        == expected
    )
