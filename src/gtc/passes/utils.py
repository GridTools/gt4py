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

from typing import Dict, Optional, Tuple

from gt4py.definitions import CartesianSpace, Extent
from gtc import common


CARTESIAN_PARALLEL_AXES = ("i", "j")


def _overlap_along_axis(
    extent: Tuple[int, int],
    interval: common.HorizontalInterval,
) -> Optional[Tuple[int, int]]:
    """Return a tuple of the distances to the edge of the compute domain, if overlapping."""
    LARGE_NUM = 10000

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

    start_diff = min(start_diff, 0) if start_diff is not None else -LARGE_NUM
    end_diff = max(end_diff, 0) if end_diff is not None else LARGE_NUM
    return (start_diff, end_diff)


def compute_extent_difference(extent: Extent, mask: common.HorizontalMask) -> Optional[Extent]:
    """Compute the difference between an compute extent and a common.HorizontalMask.

    This is used to augment the extents on fields for gtir_legacy_extents and removing
    unexecuted regions.
    """
    diffs = [
        _overlap_along_axis(extent[i], interval) if interval else None
        for i, interval in enumerate((mask.i, mask.j))
    ]
    if any(d is None for d in diffs):
        return None
    return Extent((diffs[0], diffs[1], (0, 0)))


def compute_relative_mask(extent: Extent, mask: common.HorizontalMask) -> common.HorizontalMask:
    """Compute a HorizontalMask relative to the compute extent in `extent`.

    This is used in the numpy backend to compute HorizontalMask bounds relative to
    the start/end bounds of each axis in the HorizontalBlock.
    """

    def compute_level_offset(
        bound: Optional[common.AxisBound], extent: Tuple[int, int], start: bool = True
    ) -> Tuple[common.LevelMarker, int]:
        if bound:
            level = bound.level
            if start:
                if level == common.LevelMarker.START:
                    offset = max(0, bound.offset - extent[0])
                else:
                    offset = min(0, extent[1] - bound.offset)
            else:
                if level == common.LevelMarker.END:
                    offset = min(0, extent[1] - bound.offset)
                else:
                    offset = max(0, bound.offset - extent[0])
        else:
            level = common.LevelMarker.START if start else common.LevelMarker.END
            offset = 0
        return level, offset

    args: Dict[str, common.HorizontalInterval] = {}
    for i, axis in enumerate(CartesianSpace.names[:-1]):
        horizontal_interval = getattr(mask, axis.lower())
        start_level, start_offset = compute_level_offset(
            horizontal_interval.start, extent[i], start=True
        )
        end_level, end_offset = compute_level_offset(
            horizontal_interval.end, extent[i], start=False
        )

        args[axis.lower()] = common.HorizontalInterval(
            start=common.AxisBound(level=start_level, offset=start_offset),
            end=common.AxisBound(level=end_level, offset=end_offset),
        )

    return common.HorizontalMask(**args)


def extent_from_offset(offset: common.CartesianOffset) -> Extent:
    return Extent(
        (
            (min(offset.i, 0), max(offset.i, 0)),
            (min(offset.j, 0), max(offset.j, 0)),
            (min(offset.k, 0), max(offset.k, 0)),
        )
    )
