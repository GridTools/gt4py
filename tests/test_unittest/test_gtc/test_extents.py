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

from gt4py.definitions import Extent
from gtc.passes.oir_optimizations.utils import compute_extents

from .oir_utils import AssignStmtFactory, HorizontalExecutionFactory, StencilFactory


def test_stencil_extents_simple():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="tmp", right__name="input", right__offset__i=1)]
            ),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="output", right__name="tmp", right__offset__i=1)]
            ),
        ],
        declarations__0__name="tmp",
    )

    field_extents, block_extents = compute_extents(testee)

    assert field_extents["input"] == Extent((1, 2), (0, 0))
    assert field_extents["output"] == Extent((0, 0), (0, 0))

    hexecs = testee.vertical_loops[0].sections[0].horizontal_executions
    assert block_extents[id(hexecs[0])] == Extent((0, 1), (0, 0))
    assert block_extents[id(hexecs[1])] == Extent((0, 0), (0, 0))
