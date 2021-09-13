# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

"""Utility functions used across passes."""

from typing import Optional, Tuple

from gt4py.definitions import Extent
from gtc import common


CARTESIAN_PARALLEL_AXES = ("i", "j")


def _overlap_along_axis(
    extent: Tuple[int, int], interval: common.HorizontalInterval
) -> Optional[Tuple[int, int]]:
    """Return a tuple of the distances to the edge of the compute domain, if overlapping."""
    if hasattr(interval.start, "level") and interval.start.level == common.LevelMarker.START:
        start_diff = extent[0] - interval.start.offset
    else:
        start_diff = None

    if hasattr(interval.end, "level") and interval.end.level == common.LevelMarker.END:
        end_diff = extent[1] - interval.end.offset
    else:
        end_diff = None

    if start_diff is not None and start_diff > 0 and end_diff is None:
        if interval.end.offset <= extent[0]:
            return None
    elif end_diff is not None and end_diff < 0 and start_diff is None:
        if interval.start.offset > extent[1]:
            return None

    start_diff = min(start_diff, 0) if start_diff is not None else 0
    end_diff = max(end_diff, 0) if end_diff is not None else 0
    return (start_diff, end_diff)


def compute_extent_difference(extent: Extent, mask: common.HorizontalMask) -> Optional[Extent]:
    """Compute the difference between an compute extent and a common.HorizontalMask.

    This is used to augment the extents on fields for gtir_legacy_extents and removing
    unexecuted regions.
    """
    diffs = [
        _overlap_along_axis(extent[i], interval) for i, interval in enumerate((mask.i, mask.j))
    ]
    if any(d is None for d in diffs):
        return None
    return Extent((diffs[0], diffs[1], (0, 0)))


def _compute_relative_interval(
    extent: Tuple[int, int], interval: common.HorizontalInterval
) -> common.HorizontalInterval:
    def compute_offset(
        extent: Tuple[int, int], bound: Optional[common.AxisBound], start: bool = True
    ) -> Tuple[common.LevelMarker, int]:
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
        common.HorizontalInterval(
            start=common.AxisBound(
                level=interval.start.level if interval.start else common.LevelMarker.START,
                offset=compute_offset(extent, interval.start, start=True),
            ),
            end=common.AxisBound(
                level=interval.end.level if interval.end else common.LevelMarker.END,
                offset=compute_offset(extent, interval.end, start=False),
            ),
        )
        if _overlap_along_axis(extent, interval)
        else None
    )


def compute_relative_mask(
    extent: Extent, mask: common.HorizontalMask
) -> Optional[common.HorizontalMask]:
    """Compute a HorizontalMask relative to the compute extent in `extent`.

    This is used in the numpy backend to compute HorizontalMask bounds relative to
    the start/end bounds of each axis in the HorizontalBlock.
    """
    i_interval = _compute_relative_interval(extent[0], mask.i)
    j_interval = _compute_relative_interval(extent[1], mask.j)

    return common.HorizontalMask(i=i_interval, j=j_interval) if i_interval and j_interval else None


def extent_from_offset(offset: common.CartesianOffset) -> Extent:
    return Extent(
        (
            (min(offset.i, 0), max(offset.i, 0)),
            (min(offset.j, 0), max(offset.j, 0)),
            (min(offset.k, 0), max(offset.k, 0)),
        )
    )
