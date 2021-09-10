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

from typing import Tuple, Optional
import pytest
from gtc import common
from gtc.passes import utils
from gtc.common import HorizontalInterval, AxisBound, LevelMarker


@pytest.mark.parametrize(
    "extent, interval, expected",
    [
        (
            (0, 0),
            HorizontalInterval(
                start=AxisBound(level=LevelMarker.START, offset=2),
                end=AxisBound(level=LevelMarker.END, offset=-2),
            ),
            (-2, 2),
        ),
        (
            (0, 0),
            HorizontalInterval(
                start=AxisBound(level=LevelMarker.START, offset=-2),
                end=AxisBound(level=LevelMarker.END, offset=2),
            ),
            (0, 0),
        ),
        (
            (-2, 0),
            HorizontalInterval(
                start=AxisBound(level=LevelMarker.START, offset=-3),
                end=AxisBound(level=LevelMarker.START, offset=-2),
            ),
            None,
        ),
        (
            (0, 1),
            HorizontalInterval(
                start=AxisBound(level=LevelMarker.END, offset=2),
                end=AxisBound(level=LevelMarker.END, offset=3),
            ),
            None,
        ),
    ],
)
def test_overlap_along_axis(
    extent: Tuple[int, int],
    interval: common.HorizontalInterval,
    expected: Optional[Tuple[int, int]],
):
    assert expected == utils._overlap_along_axis(extent, interval)
