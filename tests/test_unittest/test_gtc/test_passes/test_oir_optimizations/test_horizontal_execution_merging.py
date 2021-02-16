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

from gtc.passes.oir_optimizations.horizontal_execution_merging import GreedyMerging

from ...oir_utils import AssignStmtFactory, HorizontalExecutionFactory, VerticalLoopSectionFactory


def test_zero_extent_merging():
    testee = VerticalLoopSectionFactory(
        horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="foo", right__name="bar")]
            ),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="baz", right__name="bar")]
            ),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="foo", right__name="foo")]
            ),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="foo", right__name="baz")],
            ),
        ]
    )
    transformed = GreedyMerging().visit(testee)
    assert len(transformed.horizontal_executions) == 1
    assert transformed.horizontal_executions[0].body == sum(
        (he.body for he in testee.horizontal_executions), []
    )


def test_mixed_merging():
    testee = VerticalLoopSectionFactory(
        horizontal_executions=[
            HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="foo")]),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="bar", right__name="foo", right__offset__i=1)]
            ),
            HorizontalExecutionFactory(body=[AssignStmtFactory(right__name="bar")]),
        ]
    )
    transformed = GreedyMerging().visit(testee)
    assert len(transformed.horizontal_executions) == 2
    assert transformed.horizontal_executions[0].body == testee.horizontal_executions[0].body
    assert transformed.horizontal_executions[1].body == sum(
        (he.body for he in testee.horizontal_executions[1:]), []
    )


def test_write_after_read_with_offset():
    testee = VerticalLoopSectionFactory(
        horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(right__name="foo", right__offset__i=1)]
            ),
            HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="foo")]),
        ]
    )
    transformed = GreedyMerging().visit(testee)
    assert transformed == testee


def test_nonzero_extent_merging():
    testee = VerticalLoopSectionFactory(
        horizontal_executions=[
            HorizontalExecutionFactory(body=[AssignStmtFactory(right__name="foo")]),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(right__name="foo", right__offset__j=1)]
            ),
        ]
    )
    transformed = GreedyMerging().visit(testee)
    assert len(transformed.horizontal_executions) == 1
    assert transformed.horizontal_executions[0].body == sum(
        (he.body for he in testee.horizontal_executions), []
    )
