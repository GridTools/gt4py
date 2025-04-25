# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Tuple

from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.definitions import Extent


def _overlap_along_axis(
    extent: Tuple[int, int], interval: common.HorizontalInterval
) -> Optional[Tuple[int, int]]:
    """Return a tuple of the distances to the edge of the compute domain, if overlapping."""
    start_diff: Optional[int]
    end_diff: Optional[int]

    if interval.start is None:
        start_diff = 1000
    elif interval.start.level == common.LevelMarker.START:
        start_diff = extent[0] - interval.start.offset
    else:
        start_diff = None

    if interval.end is None:
        end_diff = -1000
    elif interval.end.level == common.LevelMarker.END:
        end_diff = extent[1] - interval.end.offset
    else:
        end_diff = None

    if start_diff is not None and start_diff > 0 and end_diff is None and interval.end is not None:
        if interval.end.offset <= extent[0]:
            return None
    elif (
        end_diff is not None and end_diff < 0 and start_diff is None and interval.start is not None
    ):
        if interval.start.offset > extent[1]:
            return None

    start_diff = min(start_diff, 0) if start_diff is not None else -10000
    end_diff = max(end_diff, 0) if end_diff is not None else 10000
    return (start_diff, end_diff)


def mask_overlap_with_extent(
    mask: common.HorizontalMask, horizontal_extent: Extent
) -> Optional[Extent]:
    """Compute an overlap extent between a mask and horizontal extent."""
    diffs = [
        _overlap_along_axis(ext, interval)
        for ext, interval in zip(horizontal_extent, mask.intervals)
    ]
    return Extent(diffs[0], diffs[1]) if all(d is not None for d in diffs) else None


def _compute_relative_interval(
    extent: Tuple[int, int], interval: common.HorizontalInterval
) -> Optional[Tuple[common.AxisBound, common.AxisBound]]:
    def _offset(
        extent: Tuple[int, int], bound: Optional[common.AxisBound], start: bool = True
    ) -> int:
        if bound:
            if start:
                if bound.level == common.LevelMarker.START:
                    offset = max(0, bound.offset - extent[0])
                else:
                    offset = min(0, bound.offset - extent[1])
            else:
                if bound.level == common.LevelMarker.END:
                    offset = min(0, bound.offset - extent[1])
                else:
                    offset = max(0, bound.offset - extent[0])
        else:
            offset = 0
        return offset

    return (
        (
            common.AxisBound(
                level=interval.start.level if interval.start else common.LevelMarker.START,
                offset=_offset(extent, interval.start, start=True),
            ),
            common.AxisBound(
                level=interval.end.level if interval.end else common.LevelMarker.END,
                offset=_offset(extent, interval.end, start=False),
            ),
        )
        if _overlap_along_axis(extent, interval)
        else None
    )


def compute_relative_mask(
    extent: Extent, mask: common.HorizontalMask
) -> Optional[
    Tuple[Tuple[common.AxisBound, common.AxisBound], Tuple[common.AxisBound, common.AxisBound]]
]:
    """
    Output a HorizontalMask that is relative to and always inside the extent instead of the compute domain.

    This is used in the numpy backend to compute HorizontalMask bounds relative to
    the start/end bounds of the horizontal axes.
    """
    i_interval = _compute_relative_interval(extent[0], mask.i)
    j_interval = _compute_relative_interval(extent[1], mask.j)

    return (i_interval, j_interval) if i_interval and j_interval else None
