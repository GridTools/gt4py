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

from gtc.passes.oir_optimizations.horizontal_execution_merging import GreedyMerging

from ...oir_utils import (
    AssignStmtBuilder,
    CartesianOffsetBuilder,
    FieldAccessBuilder,
    HorizontalExecutionBuilder,
    VerticalLoopSectionBuilder,
)


@pytest.fixture(params=[GreedyMerging()])
def merger(request):
    return request.param


def test_zero_extent_merging(merger):
    testee = (
        VerticalLoopSectionBuilder()
        .add_horizontal_execution(
            HorizontalExecutionBuilder().add_stmt(AssignStmtBuilder("foo", "bar").build()).build()
        )
        .add_horizontal_execution(
            HorizontalExecutionBuilder().add_stmt(AssignStmtBuilder("baz", "bar").build()).build()
        )
        .add_horizontal_execution(
            HorizontalExecutionBuilder().add_stmt(AssignStmtBuilder("foo", "foo").build()).build()
        )
        .add_horizontal_execution(
            HorizontalExecutionBuilder().add_stmt(AssignStmtBuilder("foo", "baz").build()).build()
        )
        .build()
    )
    transformed = merger.visit(testee)
    assert len(transformed.horizontal_executions) == 1
    assert transformed.horizontal_executions[0].body == sum(
        (he.body for he in testee.horizontal_executions), []
    )


def test_mixed_merging(merger):
    testee = (
        VerticalLoopSectionBuilder()
        .add_horizontal_execution(
            HorizontalExecutionBuilder().add_stmt(AssignStmtBuilder("foo", "bar").build()).build()
        )
        .add_horizontal_execution(
            HorizontalExecutionBuilder()
            .add_stmt(
                AssignStmtBuilder("baz")
                .right(
                    FieldAccessBuilder("foo").offset(CartesianOffsetBuilder(i=1).build()).build()
                )
                .build()
            )
            .build()
        )
        .add_horizontal_execution(
            HorizontalExecutionBuilder().add_stmt(AssignStmtBuilder("bar", "baz").build()).build()
        )
        .build()
    )
    transformed = merger.visit(testee)
    assert len(transformed.horizontal_executions) == 2
    assert transformed.horizontal_executions[0].body == testee.horizontal_executions[0].body
    assert transformed.horizontal_executions[1].body == sum(
        (he.body for he in testee.horizontal_executions[1:]), []
    )


def test_write_after_read_with_offset(merger):
    testee = (
        VerticalLoopSectionBuilder()
        .add_horizontal_execution(
            HorizontalExecutionBuilder()
            .add_stmt(AssignStmtBuilder("foo", "bar", (1, 0, 0)).build())
            .build()
        )
        .add_horizontal_execution(
            HorizontalExecutionBuilder().add_stmt(AssignStmtBuilder("bar", "baz").build()).build()
        )
        .build()
    )
    transformed = merger.visit(testee)
    assert transformed == testee


def test_nonzero_extent_merging(merger):
    testee = (
        VerticalLoopSectionBuilder()
        .add_horizontal_execution(
            HorizontalExecutionBuilder().add_stmt(AssignStmtBuilder("foo", "bar").build()).build()
        )
        .add_horizontal_execution(
            HorizontalExecutionBuilder()
            .add_stmt(
                AssignStmtBuilder("baz")
                .right(
                    FieldAccessBuilder("bar").offset(CartesianOffsetBuilder(i=1).build()).build()
                )
                .build()
            )
            .build()
        )
        .build()
    )
    transformed = merger.visit(testee)
    assert len(transformed.horizontal_executions) == 1
    assert transformed.horizontal_executions[0].body == sum(
        (he.body for he in testee.horizontal_executions), []
    )
