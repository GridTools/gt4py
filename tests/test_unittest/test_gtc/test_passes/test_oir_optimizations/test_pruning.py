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

from gtc.common import HorizontalInterval, HorizontalMask, LevelMarker
from gtc.passes.oir_optimizations.pruning import (
    NoFieldAccessPruning,
    UnreachableStmtPruning,
    prune_unused_parameters,
)

from ...oir_utils import (
    AssignStmtFactory,
    FieldDeclFactory,
    HorizontalExecutionFactory,
    HorizontalRestrictionFactory,
    LiteralFactory,
    LocalScalarFactory,
    ScalarAccessFactory,
    ScalarDeclFactory,
    StencilFactory,
    VerticalLoopFactory,
)


def test_all_parameters_used():
    field_param = FieldDeclFactory()
    scalar_param = ScalarDeclFactory()
    testee = StencilFactory(
        params=[field_param, scalar_param],
        vertical_loops__0__sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(left__name=field_param.name, right__name=scalar_param.name)
        ],
    )
    expected_params = [field_param, scalar_param]

    result = prune_unused_parameters(testee)

    assert expected_params == result.params


def test_unused_are_removed():
    field_param = FieldDeclFactory()
    unused_field_param = FieldDeclFactory()
    scalar_param = ScalarDeclFactory()
    unused_scalar_param = ScalarDeclFactory()
    testee = StencilFactory(
        params=[field_param, unused_field_param, scalar_param, unused_scalar_param],
        vertical_loops__0__sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(left__name=field_param.name, right__name=scalar_param.name)
        ],
    )
    expected_params = [field_param, scalar_param]

    result = prune_unused_parameters(testee)

    assert expected_params == result.params


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
                    ),
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
                body=[
                    AssignStmtFactory(left__name=out_name, right__name=in_name),
                ]
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
