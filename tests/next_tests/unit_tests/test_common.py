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

from gt4py.next.common import UnitRange


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
def test_intersection(rng1, rng2, expected):
    result = rng1 & rng2
    assert result == expected
