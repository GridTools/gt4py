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

from gtc import common


CARTESIAN_PARALLEL_AXES = ("i", "j")


def _overlap_along_axis(
    extent: Tuple[int, int],
    interval: common.HorizontalInterval,
) -> Optional[Tuple[int, int]]:
    """Return a tuple of the distances to the edge of the compute domain, if overlapping."""
    LARGE_NUM = 10000

    if interval.start.level == common.LevelMarker.START:
        start_diff = extent[0] - interval.start.offset
    else:
        start_diff = None

    if interval.end.level == common.LevelMarker.END:
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


def compute_extent_difference(
    extent: common.IJExtent, mask: common.HorizontalMask
) -> Optional[common.IJExtent]:
    """Compute the difference between an compute extent and a common.HorizontalMask.

    This is used to augment the extents on fields for gtir_legacy_extents and removing
    unexecuted regions.
    """
    diffs: Dict[str, Tuple[int, int]] = {}
    for axis in CARTESIAN_PARALLEL_AXES:
        axis_diff = _overlap_along_axis(getattr(extent, axis), getattr(mask, axis))
        if not axis_diff:
            return None
        else:
            diffs[axis] = axis_diff

    return common.IJExtent(**diffs)
