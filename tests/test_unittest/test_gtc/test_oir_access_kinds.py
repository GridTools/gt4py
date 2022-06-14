# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from gt4py.definitions import AccessKind
from gtc.passes.oir_access_kinds import compute_access_kinds

from .oir_utils import AssignStmtFactory, StencilFactory


def test_access_read_and_write():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(left__name="output_field", right__name="input_field")
        ]
    )
    access = compute_access_kinds(testee)

    assert access["input_field"] == AccessKind.READ
    assert access["output_field"] == AccessKind.WRITE


def test_access_readwrite():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(left__name="output_field", right__name="inout_field"),
            AssignStmtFactory(left__name="inout_field", right__name="other_field"),
        ],
    )
    access = compute_access_kinds(testee)

    assert access["output_field"] == AccessKind.WRITE
    assert access["inout_field"] == AccessKind.READ_WRITE


def test_access_write_only():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(left__name="inout_field", right__name="other_field"),
            AssignStmtFactory(left__name="output_field", right__name="inout_field"),
        ],
    )
    access = compute_access_kinds(testee)

    assert access["output_field"] == AccessKind.WRITE
    assert access["inout_field"] == AccessKind.WRITE
