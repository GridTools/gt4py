# -*- coding: utf-8 -*-
from gt4py.analysis.passes import overlap_with_extent
from gt4py.ir.nodes import AxisBound, AxisInterval, LevelMarker


def test_intervalinfo_overlap():
    overlap = overlap_with_extent(
        AxisInterval(
            start=AxisBound(level=LevelMarker.START, offset=2),
            end=AxisBound(level=LevelMarker.START, offset=4),
        ),
        (0, 2),
    )
    assert overlap[0] == -2 and overlap[1] > 100

    overlap = overlap_with_extent(
        AxisInterval(
            start=AxisBound(level=LevelMarker.START, offset=-1),
            end=AxisBound(level=LevelMarker.END, offset=0),
        ),
        (0, 2),
    )
    assert overlap == (0, 2)

    overlap = overlap_with_extent(
        AxisInterval(
            start=AxisBound(level=LevelMarker.START, offset=-3),
            end=AxisBound(level=LevelMarker.START, offset=-1),
        ),
        (0, 0),
    )
    assert overlap is None

    overlap = overlap_with_extent(
        AxisInterval(
            start=AxisBound(level=LevelMarker.END, offset=-3),
            end=AxisBound(level=LevelMarker.END, offset=0),
        ),
        (0, 1),
    )
    assert overlap[0] < -100 and overlap[1] == 1

    overlap = overlap_with_extent(
        AxisInterval(
            start=AxisBound(level=LevelMarker.END, offset=-1),
            end=AxisBound(level=LevelMarker.END, offset=0),
        ),
        (0, 0),
    )
    assert overlap[0] < -100 and overlap[1] == 0

    overlap = overlap_with_extent(
        AxisInterval(
            start=AxisBound(level=LevelMarker.START, offset=2),
            end=AxisBound(level=LevelMarker.START, offset=3),
        ),
        (0, 0),
    )
    assert overlap[0] == -2 and overlap[1] > 100
