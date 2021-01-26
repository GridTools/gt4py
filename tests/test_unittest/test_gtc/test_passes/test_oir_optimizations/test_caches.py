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

from gtc.passes.oir_optimizations.caches import IJCacheDetection

from ...oir_utils import (
    AssignStmtBuilder,
    HorizontalExecutionBuilder,
    IJCacheBuilder,
    TemporaryBuilder,
    VerticalLoopBuilder,
)


def test_ij_cache_detection():
    testee = (
        VerticalLoopBuilder()
        .add_horizontal_execution(
            HorizontalExecutionBuilder()
            .add_stmt(AssignStmtBuilder("tmp1", "bar", (1, 0, 0)).build())
            .build()
        )
        .add_horizontal_execution(
            HorizontalExecutionBuilder()
            .add_stmt(AssignStmtBuilder("tmp2", "tmp1", (0, 1, 0)).build())
            .build()
        )
        .add_horizontal_execution(
            HorizontalExecutionBuilder()
            .add_stmt(AssignStmtBuilder("baz", "tmp2", (0, 0, 1)).build())
            .add_stmt(AssignStmtBuilder("tmp3", "baz").build())
            .add_stmt(AssignStmtBuilder("foo", "tmp3").build())
            .build()
        )
        .add_declaration(TemporaryBuilder(name="tmp1").build())
        .add_declaration(TemporaryBuilder(name="tmp2").build())
        .add_declaration(TemporaryBuilder(name="tmp3").build())
        .add_cache(IJCacheBuilder("tmp3").build())
        .build()
    )
    transformed = IJCacheDetection().visit(testee)
    assert len(transformed.caches) == 2
    assert {cache.name for cache in transformed.caches} == {"tmp1", "tmp3"}
