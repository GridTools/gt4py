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

from typing import Sequence

import pytest

from gt4py.next import common
from gt4py.next.common import UnitRange
from gt4py.next.embedded.common import _slice_range, sub_domain


def _d(*dom: tuple[common.Dimension, tuple[int, int]]):
    dims = []
    rngs = []
    for dim, (start, stop) in dom:
        dims.append(dim)
        rngs.append(common.UnitRange(start, stop))
    return common.Domain(tuple(dims), tuple(rngs))


def test_slice_range():
    input_range = UnitRange(2, 10)
    slice_obj = slice(2, -2)
    expected = UnitRange(4, 8)

    result = _slice_range(input_range, slice_obj)
    assert result == expected


I = common.Dimension("I")
J = common.Dimension("J")
K = common.Dimension("K")


@pytest.mark.parametrize(
    "domain, index, expected",
    [
        (_d((I, (2, 5))), 1, _d()),
        (_d((I, (2, 5))), slice(1, 2), _d((I, (3, 4)))),
        (_d((I, (2, 5))), (I, 2), _d()),
        (_d((I, (2, 5))), (I, UnitRange(2, 3)), _d((I, (2, 3)))),
        (_d((I, (-2, 3))), 1, _d()),
        (_d((I, (-2, 3))), slice(1, 2), _d((I, (-1, 0)))),
        (_d((I, (-2, 3))), (I, 1), _d()),
        (_d((I, (-2, 3))), (I, UnitRange(2, 3)), _d((I, (2, 3)))),
        (_d((I, (-2, 3))), -5, _d()),
        # (_d((I, (-2, 3))), -6, IndexError),
        # (_d((I, (-2, 3))), slice(-6, -7), IndexError),
        (_d((I, (-2, 3))), 4, _d()),
        # (_d((I, (-2, 3))), 5, IndexError),
        # (_d((I, (-2, 3))), slice(4, 5), IndexError),
        # (_d((I, (-2, 3))), (I, -3), IndexError),
        # (_d((I, (-2, 3))), (I, UnitRange(-3, -2)), IndexError),
        # (_d((I, (-2, 3))), (I, 3), IndexError),
        # (_d((I, (-2, 3))), (I, UnitRange(3, 4)), IndexError),
        (
            _d((I, (2, 5)), (J, (3, 6)), (K, (4, 7))),
            2,
            _d((J, (3, 6)), (K, (4, 7))),
        ),
        (
            _d((I, (2, 5)), (J, (3, 6)), (K, (4, 7))),
            slice(2, 3),
            _d((I, (4, 5)), (J, (3, 6)), (K, (4, 7))),
        ),
        (
            _d((I, (2, 5)), (J, (3, 6)), (K, (4, 7))),
            (I, 2),
            _d((J, (3, 6)), (K, (4, 7))),
        ),
        (
            _d((I, (2, 5)), (J, (3, 6)), (K, (4, 7))),
            (I, UnitRange(2, 3)),
            _d((I, (2, 3)), (J, (3, 6)), (K, (4, 7))),
        ),
        (
            _d((I, (2, 5)), (J, (3, 6)), (K, (4, 7))),
            (J, 3),
            _d((I, (2, 5)), (K, (4, 7))),
        ),
        (
            _d((I, (2, 5)), (J, (3, 6)), (K, (4, 7))),
            (J, UnitRange(4, 5)),
            _d((I, (2, 5)), (J, (4, 5)), (K, (4, 7))),
        ),
        (
            _d((I, (2, 5)), (J, (3, 6)), (K, (4, 7))),
            ((J, 3), (I, 2)),
            _d((K, (4, 7))),
        ),
        (
            _d((I, (2, 5)), (J, (3, 6)), (K, (4, 7))),
            ((J, UnitRange(4, 5)), (I, 2)),
            _d((J, (4, 5)), (K, (4, 7))),
        ),
        (
            _d((I, (2, 5)), (J, (3, 6)), (K, (4, 7))),
            (slice(1, 2), slice(2, 3)),
            _d((I, (3, 4)), (J, (5, 6)), (K, (4, 7))),
        ),
        (
            _d((I, (2, 5)), (J, (3, 6)), (K, (4, 7))),
            (Ellipsis, slice(2, 3)),
            _d((I, (2, 5)), (J, (3, 6)), (K, (6, 7))),
        ),
        (
            _d((I, (2, 5)), (J, (3, 6)), (K, (4, 7))),
            (slice(1, 2), Ellipsis, slice(2, 3)),
            _d((I, (3, 4)), (J, (3, 6)), (K, (6, 7))),
        ),
    ],
)
def test_sub_domain(domain, index, expected):
    if expected is IndexError:
        with pytest.raises(IndexError):
            print(sub_domain(domain, index))
    else:
        result = sub_domain(domain, index)
        assert result == expected
