# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest


dace = pytest.importorskip("dace")  # noqa: E402 module level import not at top of file

from gtc import common
from gtc.passes.oir_dace_optimizations.api import optimize_horizontal_executions
from gtc.passes.oir_dace_optimizations.horizontal_execution_merging import (
    GraphMerging,
    graph_merge_horizontal_executions,
)

from ..oir_utils import (
    AssignStmtFactory,
    FieldAccessFactory,
    HorizontalExecutionFactory,
    NativeFuncCallFactory,
    StencilFactory,
    TemporaryFactory,
    VerticalLoopSectionFactory,
)


def test_zero_extent_merging():
    testee = VerticalLoopSectionFactory(
        horizontal_executions=[
            HorizontalExecutionFactory(
                body=[assignment_0 := AssignStmtFactory(left__name="foo", right__name="bar")]
            ),
            HorizontalExecutionFactory(
                body=[assignment_1 := AssignStmtFactory(left__name="baz", right__name="bar")]
            ),
            HorizontalExecutionFactory(
                body=[assignment_2 := AssignStmtFactory(left__name="foo", right__name="foo")]
            ),
            HorizontalExecutionFactory(
                body=[assignment_3 := AssignStmtFactory(left__name="foo", right__name="baz")],
            ),
        ]
    )
    transformed = (
        optimize_horizontal_executions(
            StencilFactory(vertical_loops__0__sections__0=testee),
            GraphMerging,
        )
        .vertical_loops[0]
        .sections[0]
    )
    assert len(transformed.horizontal_executions) == 1
    transformed_order = transformed.horizontal_executions[0].body
    assert transformed_order.index(assignment_0) < transformed_order.index(assignment_2)
    assert transformed_order.index(assignment_1) < transformed_order.index(assignment_3)
    assert transformed_order.index(assignment_2) < transformed_order.index(assignment_3)


def test_mixed_merging():
    testee = VerticalLoopSectionFactory(
        horizontal_executions=[
            HorizontalExecutionFactory(body=[assignment_0 := AssignStmtFactory(left__name="foo")]),
            HorizontalExecutionFactory(
                body=[
                    assignment_1 := AssignStmtFactory(
                        left__name="bar", right__name="foo", right__offset__i=1
                    )
                ]
            ),
            HorizontalExecutionFactory(body=[assignment_2 := AssignStmtFactory(right__name="bar")]),
        ]
    )
    transformed = (
        optimize_horizontal_executions(
            StencilFactory(vertical_loops__0__sections__0=testee),
            GraphMerging,
        )
        .vertical_loops[0]
        .sections[0]
    )
    assert len(transformed.horizontal_executions) == 2
    assert transformed.horizontal_executions[0].body == [assignment_0]
    assert transformed.horizontal_executions[1].body == [assignment_1, assignment_2]


def test_write_after_read_with_offset():
    testee = VerticalLoopSectionFactory(
        horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(right__name="foo", right__offset__i=1)]
            ),
            HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="foo")]),
        ]
    )
    transformed = (
        optimize_horizontal_executions(
            StencilFactory(vertical_loops__0__sections__0=testee),
            GraphMerging,
        )
        .vertical_loops[0]
        .sections[0]
    )
    for result, reference in zip(transformed.horizontal_executions, testee.horizontal_executions):
        assert result.body == reference.body


def test_nonzero_extent_merging():
    testee = VerticalLoopSectionFactory(
        horizontal_executions=[
            HorizontalExecutionFactory(body=[AssignStmtFactory(right__name="foo")]),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(right__name="foo", right__offset__j=1)]
            ),
        ]
    )
    transformed = (
        optimize_horizontal_executions(
            StencilFactory(vertical_loops__0__sections__0=testee),
            GraphMerging,
        )
        .vertical_loops[0]
        .sections[0]
    )
    assert len(transformed.horizontal_executions) == 1
    assert transformed.horizontal_executions[0].body == sum(
        (he.body for he in testee.horizontal_executions), []
    )


def test_different_iteration_spaces_param():
    # need three HE since only a read-write dependency would not be allowed to merge anyways due
    # to the read with offset. The interesting part is to enforce that the first two are not
    # merged.
    testee = StencilFactory(
        vertical_loops__0__sections__0=VerticalLoopSectionFactory(
            horizontal_executions=[
                HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="api1")]),
                HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="api2")]),
                HorizontalExecutionFactory(
                    body__0=AssignStmtFactory(
                        right=NativeFuncCallFactory(
                            func=common.NativeFunction.MIN,
                            args=[
                                FieldAccessFactory(name="api1", offset__i=1),
                                FieldAccessFactory(name="api2", offset__j=1),
                            ],
                        )
                    )
                ),
            ]
        )
    )

    transformed = graph_merge_horizontal_executions(testee)
    transformed_hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(transformed_hexecs) == 3


def test_different_iteration_spaces_temporary():
    # need three HE since only a read-write dependency would not be allowed to merge anyways due
    # to the read with offset. The interesting part is to enforce that the first two are not
    # merged.
    testee = StencilFactory(
        vertical_loops__0__sections__0=VerticalLoopSectionFactory(
            horizontal_executions=[
                HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="tmp1")]),
                HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="tmp2")]),
                HorizontalExecutionFactory(
                    body__0=AssignStmtFactory(
                        right=NativeFuncCallFactory(
                            func=common.NativeFunction.MIN,
                            args=[
                                FieldAccessFactory(name="tmp1", offset__i=1),
                                FieldAccessFactory(name="tmp2", offset__j=1),
                            ],
                        )
                    )
                ),
            ]
        ),
        declarations=[TemporaryFactory(name="tmp1"), TemporaryFactory(name="tmp2")],
    )

    transformed = graph_merge_horizontal_executions(testee)
    transformed_hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(transformed_hexecs) == 2
