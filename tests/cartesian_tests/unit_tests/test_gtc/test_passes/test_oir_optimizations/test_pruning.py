# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from gt4py.cartesian.gtc.common import HorizontalInterval, HorizontalMask, LevelMarker
from gt4py.cartesian.gtc.passes.oir_optimizations.pruning import (
    NoFieldAccessPruning,
    UnreachableStmtPruning,
)

from ...oir_utils import (
    AssignStmtFactory,
    FieldAccessFactory,
    HorizontalExecutionFactory,
    HorizontalRestrictionFactory,
    LiteralFactory,
    LocalScalarFactory,
    ScalarAccessFactory,
    StencilFactory,
    VerticalLoopFactory,
)


def test_no_field_access_pruning():
    testee = StencilFactory(
        vertical_loops=[
            VerticalLoopFactory(
                sections__0__horizontal_executions=[
                    HorizontalExecutionFactory(body=[AssignStmtFactory()]),
                    HorizontalExecutionFactory(
                        body=[
                            AssignStmtFactory(
                                left=ScalarAccessFactory(name="foo"), right=LiteralFactory()
                            )
                        ],
                        declarations=[LocalScalarFactory(name="foo")],
                    ),
                ]
            ),
            VerticalLoopFactory(
                sections__0__horizontal_executions=[
                    HorizontalExecutionFactory(
                        body=[
                            AssignStmtFactory(
                                left=ScalarAccessFactory(name="bar"), right=LiteralFactory()
                            )
                        ],
                        declarations=[LocalScalarFactory(name="bar")],
                    )
                ]
            ),
        ]
    )
    transformed = NoFieldAccessPruning().visit(testee)
    assert len(transformed.vertical_loops) == 1
    assert len(transformed.vertical_loops[0].sections[0].horizontal_executions) == 1


def test_no_field_write_access_pruning():
    testee = StencilFactory(
        vertical_loops=[
            VerticalLoopFactory(
                sections__0__horizontal_executions=[
                    HorizontalExecutionFactory(
                        body=[
                            AssignStmtFactory(
                                left=FieldAccessFactory(name="foo"), right=LiteralFactory()
                            )
                        ]
                    )
                ]
            ),
            VerticalLoopFactory(
                sections__0__horizontal_executions=[
                    HorizontalExecutionFactory(
                        body=[
                            AssignStmtFactory(
                                left=ScalarAccessFactory(name="bar"),
                                right=FieldAccessFactory(name="foo"),
                            )
                        ],
                        declarations=[LocalScalarFactory(name="bar")],
                    )
                ]
            ),
        ]
    )
    transformed = NoFieldAccessPruning().visit(testee)
    assert len(transformed.vertical_loops) == 1
    assert len(transformed.vertical_loops[0].sections[0].horizontal_executions) == 1


def test_unreachable_stmt_pruning():
    out_name = "out_field"
    in_name = "in_field"
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name=out_name, right__name=in_name)]
            ),
            HorizontalExecutionFactory(
                body=[
                    HorizontalRestrictionFactory(
                        mask=HorizontalMask(
                            i=HorizontalInterval.at_endpt(LevelMarker.START, 0),
                            j=HorizontalInterval.full(),
                        ),
                        body=[AssignStmtFactory(left__name=out_name, right=LiteralFactory())],
                    ),
                    HorizontalRestrictionFactory(
                        mask=HorizontalMask(
                            i=HorizontalInterval.at_endpt(LevelMarker.END, 1),
                            j=HorizontalInterval.full(),
                        ),
                        body=[AssignStmtFactory(left__name=out_name, right=LiteralFactory())],
                    ),
                    HorizontalRestrictionFactory(
                        mask=HorizontalMask(
                            i=HorizontalInterval.full(),
                            j=HorizontalInterval.at_endpt(LevelMarker.START, -1),
                        ),
                        body=[AssignStmtFactory(left__name=out_name, right=LiteralFactory())],
                    ),
                    HorizontalRestrictionFactory(
                        mask=HorizontalMask(
                            i=HorizontalInterval.full(),
                            j=HorizontalInterval.at_endpt(LevelMarker.END, 0),
                        ),
                        body=[AssignStmtFactory(left__name=out_name, right=LiteralFactory())],
                    ),
                ]
            ),
        ]
    )

    stencil = UnreachableStmtPruning().visit(testee)
    assert len(stencil.vertical_loops[0].sections[0].horizontal_executions[1].body) == 2
