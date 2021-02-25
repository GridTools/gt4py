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

from gtc import common, oir
from gtc.passes.oir_optimizations.horizontal_execution_merging import GreedyMerging, OnTheFlyMerging

from ...oir_utils import (
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


def test_on_the_fly_merging_basic():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="tmp")]),
            HorizontalExecutionFactory(body=[AssignStmtFactory(right__name="tmp")]),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = OnTheFlyMerging().visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 1
    assert len(hexecs[0].declarations) == 1
    assert isinstance(hexecs[0].declarations[0], oir.LocalScalar)
    assert not transformed.declarations


def test_on_the_fly_merging_with_offsets():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="tmp", right__name="foo")]
            ),
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(right__name="tmp", right__offset__i=1),
                    AssignStmtFactory(right__name="tmp", right__offset__j=1),
                ]
            ),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = OnTheFlyMerging().visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 1
    assert len(hexecs[0].declarations) == 2
    assert all(isinstance(d, oir.LocalScalar) for d in hexecs[0].declarations)
    assert not transformed.declarations
    assert transformed.iter_tree().if_isinstance(oir.FieldAccess).filter(
        lambda x: x.name == "foo"
    ).getattr("offset").map(lambda o: (o.i, o.j, o.k)).to_set() == {(1, 0, 0), (0, 1, 0)}


def test_on_the_fly_merging_with_mask():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="tmp")],
                mask=FieldAccessFactory(name="mask", dtype=common.DataType.BOOL),
            ),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(right__name="tmp")],
            ),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = OnTheFlyMerging().visit(testee)
    assert len(transformed.vertical_loops[0].sections[0].horizontal_executions) == 2
    assert len(transformed.declarations) == 1


def test_on_the_fly_merging_with_expensive_function():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(
                        left__name="tmp",
                        right=NativeFuncCallFactory(func=common.NativeFunction.SIN),
                    )
                ]
            ),
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(right__name="tmp", right__offset__i=1),
                    AssignStmtFactory(right__name="tmp", right__offset__j=1),
                ]
            ),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = OnTheFlyMerging(allow_expensive_function_duplication=False).visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 2


def test_on_the_fly_merging_body_size_limit():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="tmp", right__name="foo")]
            ),
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(right__name="tmp", right__offset__i=1),
                    AssignStmtFactory(right__name="tmp", right__offset__j=1),
                ]
            ),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = OnTheFlyMerging(max_horizontal_execution_body_size=0).visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 2
