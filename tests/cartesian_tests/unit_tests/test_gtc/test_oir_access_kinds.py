# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.cartesian.definitions import AccessKind
from gt4py.cartesian.gtc.passes.oir_access_kinds import compute_access_kinds

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
        ]
    )
    access = compute_access_kinds(testee)

    assert access["output_field"] == AccessKind.WRITE
    assert access["inout_field"] == AccessKind.READ_WRITE


def test_access_write_only():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(left__name="inout_field", right__name="other_field"),
            AssignStmtFactory(left__name="output_field", right__name="inout_field"),
        ]
    )
    access = compute_access_kinds(testee)

    assert access["output_field"] == AccessKind.WRITE
    assert access["inout_field"] == AccessKind.WRITE
