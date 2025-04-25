# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.dace import daceir as dcir

# Because "dace tests" filter by `requires_dace`, we still need to add the marker.
# This global variable add the marker to all test functions in this module.
pytestmark = pytest.mark.requires_dace


def test_DomainInterval_intersection() -> None:
    I_0_4 = dcir.DomainInterval(
        start=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.START, offset=0),
        end=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.START, offset=4),
    )
    I_2_10 = dcir.DomainInterval(
        start=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.START, offset=2),
        end=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.START, offset=10),
    )
    I_2_5 = dcir.DomainInterval(
        start=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.START, offset=2),
        end=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.START, offset=5),
    )
    I_8_15 = dcir.DomainInterval(
        start=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.START, offset=8),
        end=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.START, offset=15),
    )

    # expected results
    I_2_4 = dcir.DomainInterval(
        start=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.START, offset=2),
        end=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.START, offset=4),
    )
    I_8_10 = dcir.DomainInterval(
        start=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.START, offset=8),
        end=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.START, offset=10),
    )

    assert (
        dcir.DomainInterval.intersection(dcir.Axis.I, I_0_4, I_2_10) == I_2_4
    ), "intersection left"
    assert (
        dcir.DomainInterval.intersection(dcir.Axis.I, I_2_10, I_8_15) == I_8_10
    ), "intersection right"

    assert (
        dcir.DomainInterval.intersection(dcir.Axis.I, I_2_5, I_2_10) == I_2_5
    ), "first contained in second"
    assert (
        dcir.DomainInterval.intersection(dcir.Axis.I, I_2_10, I_2_5) == I_2_5
    ), "second contained in first"

    with pytest.raises(ValueError, match=r"^No intersection found for intervals *"):
        dcir.DomainInterval.intersection(dcir.Axis.I, I_0_4, I_8_15)
