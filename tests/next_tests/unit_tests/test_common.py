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

import pytest

from gt4py.next.common import UnitRange, Dimension, Domain, DimensionKind

IDim = Dimension("IDim")
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


@pytest.mark.parametrize("rng1, rng2, expected", [
    (UnitRange(0, 5), UnitRange(10, 15), UnitRange(0, 0)),
    (UnitRange(0, 5), UnitRange(5, 10), UnitRange(5, 5)),
    (UnitRange(0, 5), UnitRange(3, 7), UnitRange(3, 5)),
    (UnitRange(0, 5), UnitRange(1, 6), UnitRange(1, 5)),
    (UnitRange(0, 5), UnitRange(-5, 5), UnitRange(0, 5)),
    (UnitRange(0, 0), UnitRange(0, 5), UnitRange(0, 0)),
    (UnitRange(0, 0), UnitRange(0, 0), UnitRange(0, 0)),
])
def test_unit_range_intersection(rng1, rng2, expected):
    result = rng1 & rng2
    assert result == expected


def test_domain_length(domain):
    assert len(domain) == 3


def test_domain_contains_range(domain):
    assert UnitRange(0, 10) in domain
    assert UnitRange(5, 15) in domain
    assert UnitRange(20, 30) in domain
    assert UnitRange(25, 28) in domain


def test_domain_contains_element(domain):
    assert 0 in domain
    assert 5 in domain
    assert 20 in domain
    assert 9 in domain
    assert 14 in domain


def test_domain_not_contains_range(domain):
    assert UnitRange(25, 35) not in domain
    assert UnitRange(10, 16) not in domain


def test_domain_not_contains_element(domain):
    assert 100 not in domain
    assert -5 not in domain
    assert 16 not in domain


def test_domain_iter_method(domain):
    iterated_values = [val for val in domain]
    assert iterated_values == list(zip(domain.dims, domain.ranges))


def test_domain_and_operation_different_dimensions(domain):
    dimensions = (IDim, JDim)
    ranges = (UnitRange(2, 12), UnitRange(7, 17))
    domain2 = Domain(dimensions, ranges)
    result_domain = domain & domain2

    assert len(result_domain) == 3
    assert all(isinstance(r, UnitRange) for r in result_domain.ranges)
    assert result_domain.dims == (IDim, JDim, KDim)

    assert result_domain.ranges[0] == UnitRange(2, 10)  # Intersection of domain.range1 and domain2.range1
    assert result_domain.ranges[1] == UnitRange(7, 15)  # Intersection of domain.range2 and domain2.range2
    assert result_domain.ranges[2] == UnitRange(20, 30)  # Broadcasting on missing dimension


def test_domain_and_operation_different_dimensions_reversed(domain):
    dimensions = (JDim, IDim)
    ranges = (UnitRange(2, 12), UnitRange(7, 17))
    domain2 = Domain(dimensions, ranges)

    with pytest.raises(ValueError,
                       match="Dimensions can not be promoted. The following dimensions appear in contradicting order: IDim, JDim."):
        domain & domain2
