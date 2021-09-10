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
    extent: Tuple[int, int], interval: common.HorizontalInterval, use_interval_offsets: bool = False
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

    LARGE_NUM = 10000
    start_max = -LARGE_NUM if not use_interval_offsets else interval.start.offset
    end_max = LARGE_NUM if not use_interval_offsets else interval.end.offset

    start_diff = min(start_diff, 0) if start_diff is not None else start_max
    end_diff = max(end_diff, 0) if end_diff is not None else end_max
    return (start_diff, end_diff)


def compute_extent_difference(
    extent: Extent, mask: common.HorizontalMask, use_interval_offsets: bool = False
) -> Optional[Extent]:
    """Compute the difference between an compute extent and a common.HorizontalMask.

    This is used to augment the extents on fields for gtir_legacy_extents and removing
    unexecuted regions.
    """
    diffs = [
        _overlap_along_axis(extent[i], interval, use_interval_offsets=use_interval_offsets)
        if interval
        else None
        for i, interval in enumerate((mask.i, mask.j))
    ]
    if any(d is None for d in diffs):
        return None
    return Extent((diffs[0], diffs[1], (0, 0)))


def compute_relative_mask(
    extent: Extent, mask: common.HorizontalMask
) -> Optional[common.HorizontalMask]:
    """Compute a HorizontalMask relative to the compute extent in `extent`.

    This is used in the numpy backend to compute HorizontalMask bounds relative to
    the start/end bounds of each axis in the HorizontalBlock.
    """
    if rel_ext := compute_extent_difference(extent, mask, use_interval_offsets=True):
        return common.HorizontalMask(
            i=common.HorizontalInterval(
                start=common.AxisBound(level=mask.i.start.level, offset=rel_ext[0][0]),
                end=common.AxisBound(level=mask.i.end.level, offset=rel_ext[0][1]),
            ),
            j=common.HorizontalInterval(
                start=common.AxisBound(level=mask.j.start.level, offset=rel_ext[1][0]),
                end=common.AxisBound(level=mask.j.end.level, offset=rel_ext[1][1]),
            ),
        )
    else:
        return None


def extent_from_offset(offset: common.CartesianOffset) -> Extent:
    return Extent(
        (
            (min(offset.i, 0), max(offset.i, 0)),
            (min(offset.j, 0), max(offset.j, 0)),
            (min(offset.k, 0), max(offset.k, 0)),
        )
    )
