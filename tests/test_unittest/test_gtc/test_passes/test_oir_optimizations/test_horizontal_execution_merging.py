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
from gtc.passes.oir_optimizations.horizontal_execution_merging import OnTheFlyMerging

from ...oir_utils import (
    AssignStmtFactory,
    BinaryOpFactory,
    HorizontalExecutionFactory,
    LocalScalarFactory,
    NativeFuncCallFactory,
    ScalarAccessFactory,
    StencilFactory,
    TemporaryFactory,
    VerticalLoopFactory,
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


def test_on_the_fly_merging_api_field():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body__0=AssignStmtFactory(left__name="mid", right__name="inp")
            ),
            HorizontalExecutionFactory(
                body__0=AssignStmtFactory(left__name="outp", right__name="mid")
            ),
        ]
    )
    transformed = OnTheFlyMerging().visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 2


def test_on_the_fly_merging_field_read_later():
    testee = StencilFactory(
        vertical_loops=[
            VerticalLoopFactory(
                sections__0__horizontal_executions=[
                    HorizontalExecutionFactory(
                        body=[AssignStmtFactory(left__name="mid", right__name="inp")]
                    ),
                    HorizontalExecutionFactory(
                        body=[AssignStmtFactory(left__name="outp1", right__name="mid")]
                    ),
                ]
            ),
            VerticalLoopFactory(
                sections__0__horizontal_executions__0=HorizontalExecutionFactory(
                    body=[AssignStmtFactory(left__name="outp2", right__name="mid")]
                )
            ),
        ]
    )
    transformed = OnTheFlyMerging().visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 2


def test_on_the_fly_merging_repeated():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="tmp")]),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="out1", right__name="tmp")]
            ),
            HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="tmp")]),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="out2", right__name="tmp")]
            ),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = OnTheFlyMerging().visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 2


def test_on_the_fly_merging_nested():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="tmp")]),
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(
                        left__name="tmp_offset", right__name="tmp", right__offset__i=1
                    )
                ]
            ),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="out", right__name="tmp_offset")]
            ),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="out2", right__name="tmp_offset")]
            ),
        ],
        declarations=[TemporaryFactory(name="tmp"), TemporaryFactory(name="tmp_offset")],
    )
    transformed = OnTheFlyMerging().visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 2


def test_on_the_fly_merging_localscalars():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(
                        left=ScalarAccessFactory(name="scalar_tmp"),
                        right__name="in",
                    ),
                    AssignStmtFactory(
                        left__name="tmp", right=ScalarAccessFactory(name="scalar_tmp")
                    ),
                ],
                declarations=[LocalScalarFactory(name="scalar_tmp")],
            ),
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(
                        left=ScalarAccessFactory(name="scalar_tmp"),
                        right__name="in",
                        right__offset__i=1,
                    ),
                    AssignStmtFactory(
                        left__name="out",
                        right=BinaryOpFactory(
                            left=ScalarAccessFactory(name="scalar_tmp"), right__name="tmp"
                        ),
                    ),
                ],
                declarations=[LocalScalarFactory(name="scalar_tmp")],
            ),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = OnTheFlyMerging().visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 1
    assert len(hexecs[0].declarations) == 3
