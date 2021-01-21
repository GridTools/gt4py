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
    AssignStmtBuilder,
    HorizontalExecutionBuilder,
    TemporaryBuilder,
    VerticalLoopBuilder,
)


def test_temporaries_to_scalars_basic():
    testee = (
        VerticalLoopBuilder()
        .add_horizontal_execution(
            HorizontalExecutionBuilder()
            .add_stmt(AssignStmtBuilder("tmp", "foo").build())
            .add_stmt(AssignStmtBuilder("bar", "tmp").build())
            .build()
        )
        .add_declaration(TemporaryBuilder(name="tmp").build())
        .build()
    )
    transformed = TemporariesToScalars().visit(testee)
    hexec = transformed.horizontal_executions[0]
    assert isinstance(hexec.body[0].left, oir.ScalarAccess)
    assert isinstance(hexec.body[1].right, oir.ScalarAccess)
    assert not transformed.declarations
    assert hexec.declarations[0].name == "tmp"
