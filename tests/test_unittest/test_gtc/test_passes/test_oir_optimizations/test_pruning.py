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

from gtc.passes.oir_optimizations.pruning import NoFieldAccessPruning

from ...oir_utils import (
    AssignStmtFactory,
    HorizontalExecutionFactory,
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
                    ),
                ]
            ),
        ]
    )
    transformed = NoFieldAccessPruning().visit(testee)
    assert len(transformed.vertical_loops) == 1
    assert len(transformed.vertical_loops[0].sections[0].horizontal_executions) == 1
