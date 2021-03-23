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

from gtc import oir
from gtc.passes.oir_optimizations.temporaries import TemporariesToScalars

from ...oir_utils import (
    AssignStmtFactory,
    HorizontalExecutionFactory,
    StencilFactory,
    TemporaryFactory,
)


def test_temporaries_to_scalars_basic():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(left__name="tmp"),
            AssignStmtFactory(right__name="tmp"),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = TemporariesToScalars().visit(testee)
    hexec = transformed.vertical_loops[0].sections[0].horizontal_executions[0]
    assert isinstance(hexec.body[0].left, oir.ScalarAccess)
    assert isinstance(hexec.body[1].right, oir.ScalarAccess)
    assert not transformed.declarations
    assert len(hexec.declarations) == 1


def test_temporaries_to_scalars_multiexec():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(left__name="tmp"),
                    AssignStmtFactory(right__name="tmp"),
                ]
            ),
            HorizontalExecutionFactory(body=[AssignStmtFactory(right__name="tmp")]),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = TemporariesToScalars().visit(testee)
    assert "tmp" in {d.name for d in transformed.declarations}
    assert not transformed.iter_tree().if_isinstance(oir.ScalarAccess).to_list()
