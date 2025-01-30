# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
