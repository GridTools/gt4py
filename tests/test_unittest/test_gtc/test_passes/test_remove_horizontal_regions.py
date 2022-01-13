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

import pytest

import eve
from gtc import common
from gtc.oir import HorizontalMask
from gtc.passes.oir_optimizations.remove_regions import RemoveUnexecutedRegions
from tests.test_unittest.test_gtc.oir_utils import (
    HorizontalExecutionFactory,
    MaskStmtFactory,
    VerticalLoopSectionFactory,
)


@pytest.fixture
def inaccessible_horizontal_mask():
    interval = common.HorizontalInterval(
        start=common.AxisBound.from_start(-2), end=common.AxisBound.from_start(-1)
    )
    yield HorizontalMask(i=interval, j=interval)


def test_remove_unexecuted(inaccessible_horizontal_mask):
    testee = VerticalLoopSectionFactory(
        horizontal_executions=[
            HorizontalExecutionFactory(body=[MaskStmtFactory(mask=inaccessible_horizontal_mask)]),
        ]
    )

    ctx = RemoveUnexecutedRegions.Context()
    assert RemoveUnexecutedRegions().visit(testee, ctx=ctx) == eve.NOTHING
