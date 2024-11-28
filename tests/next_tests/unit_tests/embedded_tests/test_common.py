# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Sequence

import pytest

from gt4py.next import common
from gt4py.next.common import UnitRange, NamedIndex, NamedRange
from gt4py.next.embedded import exceptions as embedded_exceptions
from gt4py.next.embedded.common import (
    _slice_range,
    canonicalize_any_index_sequence,
    iterate_domain,
    sub_domain,
    restrict_to_intersection,
    domain_intersection,
)


@pytest.mark.parametrize(
    "rng, slce, expected",
    [
        (UnitRange(2, 10), slice(2, -2), UnitRange(4, 8)),
        (UnitRange(2, 10), slice(2, None), UnitRange(4, 10)),
        (UnitRange(2, 10), slice(None, -2), UnitRange(2, 8)),
        (UnitRange(2, 10), slice(None), UnitRange(2, 10)),
    ],
)
def test_slice_range(rng, slce, expected):
    result = _slice_range(rng, slce)
    assert result == expected


I = common.Dimension("I")
J = common.Dimension("J")
K = common.Dimension("K")


@pytest.mark.parametrize(
    "domain, index, expected",
    [
        ([(I, (2, 5))], 1, []),
        ([(I, (2, 5))], slice(1, 2), [(I, (3, 4))]),
        ([(I, (2, 5))], NamedIndex(I, 2), []),
        ([(I, (2, 5))], NamedRange(I, UnitRange(2, 3)), [(I, (2, 3))]),
        ([(I, (-2, 3))], 1, []),
        ([(I, (-2, 3))], slice(1, 2), [(I, (-1, 0))]),
        ([(I, (-2, 3))], NamedIndex(I, 1), []),
        ([(I, (-2, 3))], NamedRange(I, UnitRange(2, 3)), [(I, (2, 3))]),
        ([(I, (-2, 3))], -5, []),
        ([(I, (-2, 3))], -6, IndexError),
        ([(I, (-2, 3))], slice(-7, -6), IndexError),
        ([(I, (-2, 3))], slice(-6, -7), IndexError),
        ([(I, (-2, 3))], 4, []),
        ([(I, (-2, 3))], 5, IndexError),
        ([(I, (-2, 3))], slice(4, 5), [(I, (2, 3))]),
        ([(I, (-2, 3))], slice(5, 6), IndexError),
        ([(I, (-2, 3))], NamedIndex(I, -3), IndexError),
        ([(I, (-2, 3))], NamedRange(I, UnitRange(-3, -2)), IndexError),
        ([(I, (-2, 3))], NamedIndex(I, 3), IndexError),
        ([(I, (-2, 3))], NamedRange(I, UnitRange(3, 4)), IndexError),
        ([(I, (2, 5)), (J, (3, 6)), (K, (4, 7))], 2, [(J, (3, 6)), (K, (4, 7))]),
        (
            [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
            slice(2, 3),
            [(I, (4, 5)), (J, (3, 6)), (K, (4, 7))],
        ),
        ([(I, (2, 5)), (J, (3, 6)), (K, (4, 7))], NamedIndex(I, 2), [(J, (3, 6)), (K, (4, 7))]),
        (
            [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
            NamedRange(I, UnitRange(2, 3)),
            [(I, (2, 3)), (J, (3, 6)), (K, (4, 7))],
        ),
        ([(I, (2, 5)), (J, (3, 6)), (K, (4, 7))], NamedIndex(J, 3), [(I, (2, 5)), (K, (4, 7))]),
        (
            [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
            NamedRange(J, UnitRange(4, 5)),
            [(I, (2, 5)), (J, (4, 5)), (K, (4, 7))],
        ),
        (
            [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
            (NamedIndex(J, 3), NamedIndex(I, 2)),
            [(K, (4, 7))],
        ),
        (
            [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
            (NamedRange(J, UnitRange(4, 5)), NamedIndex(I, 2)),
            [(J, (4, 5)), (K, (4, 7))],
        ),
        (
            [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
            (slice(1, 2), slice(2, 3)),
            [(I, (3, 4)), (J, (5, 6)), (K, (4, 7))],
        ),
        (
            [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
            (Ellipsis, slice(2, 3)),
            [(I, (2, 5)), (J, (3, 6)), (K, (6, 7))],
        ),
        (
            [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
            (slice(1, 2), Ellipsis, slice(2, 3)),
            [(I, (3, 4)), (J, (3, 6)), (K, (6, 7))],
        ),
        (
            [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
            (slice(1, 2), slice(1, 2), Ellipsis),
            [(I, (3, 4)), (J, (4, 5)), (K, (4, 7))],
        ),
    ],
)
def test_sub_domain(domain, index, expected):
    domain = common.domain(domain)
    if expected is IndexError:
        with pytest.raises(embedded_exceptions.IndexOutOfBounds):
            sub_domain(domain, index)
    else:
        expected = common.domain(expected)
        result = sub_domain(domain, index)
        assert result == expected


def test_iterate_domain():
    domain = common.domain({I: 2, J: 3})
    ref = []
    for i in domain[I].unit_range:
        for j in domain[J].unit_range:
            ref.append(((I, i), (J, j)))

    testee = list(iterate_domain(domain))

    assert testee == ref


@pytest.mark.parametrize(
    "slices, expected",
    [
        [slice(I(3), I(4)), (NamedRange(I, common.UnitRange(3, 4)),)],
        [
            (slice(J(3), J(6)), slice(I(3), I(5))),
            (NamedRange(J, common.UnitRange(3, 6)), NamedRange(I, common.UnitRange(3, 5))),
        ],
        [slice(I(1), J(7)), IndexError],
        [slice(I(1), None), IndexError],
        [slice(None, K(8)), IndexError],
    ],
)
def test_slicing(slices, expected):
    if expected is IndexError:
        with pytest.raises(IndexError):
            canonicalize_any_index_sequence(slices)
    else:
        testee = canonicalize_any_index_sequence(slices)
        assert testee == expected


def test_domain_intersection():
    # see also tests in unit_tests/test_common.py for tests with 2 domains: `dom0 & dom1`
    testee = (common.domain({I: (0, 5)}), common.domain({I: (1, 3)}), common.domain({I: (0, 3)}))

    result = domain_intersection(*testee)

    expected = testee[0] & testee[1] & testee[2]
    assert result == expected


def test_domain_intersection_empty():
    result = domain_intersection()
    assert result == common.Domain()


def test_intersect_domains():
    testee = (common.domain({I: (0, 5), J: (1, 2)}), common.domain({I: (1, 3), J: (1, 3)}))
    result = restrict_to_intersection(*testee, ignore_dims=J)

    expected = (common.domain({I: (1, 3), J: (1, 2)}), common.domain({I: (1, 3), J: (1, 3)}))
    assert result == expected


def test_intersect_domains_ignore_dims_none():
    testee = (common.domain({I: (0, 5), J: (1, 2)}), common.domain({I: (1, 3), J: (1, 3)}))
    result = restrict_to_intersection(*testee)

    expected = (domain_intersection(*testee),) * 2
    assert result == expected


def test_intersect_domains_ignore_all_dims():
    testee = (common.domain({I: (0, 5), J: (1, 2)}), common.domain({I: (1, 3), J: (1, 3)}))
    result = restrict_to_intersection(*testee, ignore_dims=(I, J))

    expected = testee
    assert result == expected
