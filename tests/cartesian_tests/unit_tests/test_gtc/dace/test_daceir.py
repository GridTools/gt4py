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


def test_DomainInterval() -> None:
    I_start = dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.START)
    I_end = dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.END)
    interval = dcir.DomainInterval(start=I_start, end=I_end)

    assert interval.start == I_start
    assert interval.end == I_end

    with pytest.raises(ValueError, match=r"^Axis need to match for start and end bounds. Got *"):
        dcir.DomainInterval(
            start=I_start,
            end=dcir.AxisBound(axis=dcir.Axis.J, level=common.LevelMarker.END),
        )


def test_DomainInterval_intersection() -> None:
    I_0_4 = dcir.DomainInterval(
        start=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.START),
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
    I_full = dcir.DomainInterval(
        start=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.START),
        end=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.END),
    )
    I_end_m3 = dcir.DomainInterval(
        start=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.END, offset=-3),
        end=dcir.AxisBound(axis=dcir.Axis.I, level=common.LevelMarker.END),
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
    assert (
        dcir.DomainInterval.intersection(dcir.Axis.I, I_8_15, I_full) == I_8_15,
        "full interval overlaps with start level",
    )
    assert (
        dcir.DomainInterval.intersection(dcir.Axis.I, I_end_m3, I_full) == I_end_m3,
        "full interval overlaps with end level",
    )

    with pytest.raises(ValueError, match=r"^No intersection found for intervals *"):
        dcir.DomainInterval.intersection(dcir.Axis.I, I_0_4, I_8_15)

    with pytest.raises(ValueError, match=r"^Axis need to match: *"):
        dcir.DomainInterval.intersection(
            dcir.Axis.I,
            dcir.DomainInterval(
                start=dcir.AxisBound(axis=dcir.Axis.J, level=common.LevelMarker.START),
                end=dcir.AxisBound(axis=dcir.Axis.J, level=common.LevelMarker.END),
            ),
            I_full,
        )

    with pytest.raises(ValueError, match=r"^Axis need to match: *"):
        dcir.DomainInterval.intersection(
            dcir.Axis.I,
            I_full,
            dcir.DomainInterval(
                start=dcir.AxisBound(axis=dcir.Axis.J, level=common.LevelMarker.START),
                end=dcir.AxisBound(axis=dcir.Axis.J, level=common.LevelMarker.END),
            ),
        )
