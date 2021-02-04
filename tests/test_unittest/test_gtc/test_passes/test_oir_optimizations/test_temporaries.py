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
from gtc.passes.oir_optimizations.temporaries import (
    LocalTemporariesToScalars,
    WriteBeforeReadTemporariesToScalars,
)

from ...oir_utils import (
    AssignStmtBuilder,
    FieldDeclBuilder,
    HorizontalExecutionBuilder,
    StencilBuilder,
    TemporaryBuilder,
    VerticalLoopBuilder,
    VerticalLoopSectionBuilder,
)


def test_local_temporaries_to_scalars_basic():
    testee = (
        StencilBuilder()
        .add_param(FieldDeclBuilder("foo").build())
        .add_param(FieldDeclBuilder("bar").build())
        .add_vertical_loop(
            VerticalLoopBuilder()
            .add_section(
                VerticalLoopSectionBuilder()
                .add_horizontal_execution(
                    HorizontalExecutionBuilder()
                    .add_stmt(AssignStmtBuilder("tmp", "foo").build())
                    .add_stmt(AssignStmtBuilder("bar", "tmp").build())
                    .build()
                )
                .build()
            )
            .add_declaration(TemporaryBuilder(name="tmp").build())
            .build()
        )
        .build()
    )
    transformed = LocalTemporariesToScalars().visit(testee)
    hexec = transformed.vertical_loops[0].sections[0].horizontal_executions[0]
    assert isinstance(hexec.body[0].left, oir.ScalarAccess)
    assert isinstance(hexec.body[1].right, oir.ScalarAccess)
    assert not transformed.vertical_loops[0].declarations
    assert len(hexec.declarations) == 1


def test_local_temporaries_to_scalars_multiexec():
    testee = (
        StencilBuilder()
        .add_param(FieldDeclBuilder("foo").build())
        .add_param(FieldDeclBuilder("bar").build())
        .add_param(FieldDeclBuilder("baz").build())
        .add_vertical_loop(
            VerticalLoopBuilder()
            .add_section(
                VerticalLoopSectionBuilder()
                .add_horizontal_execution(
                    HorizontalExecutionBuilder()
                    .add_stmt(AssignStmtBuilder("tmp", "foo").build())
                    .add_stmt(AssignStmtBuilder("bar", "tmp").build())
                    .build()
                )
                .add_horizontal_execution(
                    HorizontalExecutionBuilder()
                    .add_stmt(AssignStmtBuilder("baz", "tmp").build())
                    .build()
                )
                .build()
            )
            .add_declaration(TemporaryBuilder(name="tmp").build())
            .build()
        )
        .build()
    )
    transformed = LocalTemporariesToScalars().visit(testee)
    assert "tmp" in {d.name for d in transformed.vertical_loops[0].declarations}
    assert not transformed.iter_tree().if_isinstance(oir.ScalarAccess).to_list()


def test_write_before_read_temporaries_to_scalars():
    testee = (
        StencilBuilder()
        .add_param(FieldDeclBuilder("foo").build())
        .add_param(FieldDeclBuilder("bar").build())
        .add_vertical_loop(
            VerticalLoopBuilder()
            .add_section(
                VerticalLoopSectionBuilder()
                .add_horizontal_execution(
                    HorizontalExecutionBuilder()
                    .add_stmt(AssignStmtBuilder("tmp1", "foo").build())
                    .add_stmt(AssignStmtBuilder("tmp2", "tmp1").build())
                    .add_stmt(AssignStmtBuilder("tmp3", "tmp2").build())
                    .build()
                )
                .add_horizontal_execution(
                    HorizontalExecutionBuilder()
                    .add_stmt(AssignStmtBuilder("bar", "tmp2").build())
                    .add_stmt(AssignStmtBuilder("tmp3", "bar").build())
                    .add_stmt(AssignStmtBuilder("foo", "tmp3").build())
                    .build()
                )
                .build()
            )
            .add_declaration(TemporaryBuilder(name="tmp1").build())
            .add_declaration(TemporaryBuilder(name="tmp2").build())
            .add_declaration(TemporaryBuilder(name="tmp3").build())
            .build()
        )
        .build()
    )
    transformed = WriteBeforeReadTemporariesToScalars().visit(testee)
    hexec0 = transformed.vertical_loops[0].sections[0].horizontal_executions[0]
    hexec1 = transformed.vertical_loops[0].sections[0].horizontal_executions[1]
    assert len(hexec0.declarations) == 2
    assert len(hexec1.declarations) == 1
    assert len(transformed.vertical_loops[0].declarations) == 1
    assert isinstance(hexec0.body[0].left, oir.ScalarAccess)
    assert not isinstance(hexec0.body[1].left, oir.ScalarAccess)
    assert isinstance(hexec0.body[1].right, oir.ScalarAccess)
    assert isinstance(hexec0.body[2].left, oir.ScalarAccess)
    assert not isinstance(hexec0.body[2].right, oir.ScalarAccess)
    assert not isinstance(hexec1.body[0].right, oir.ScalarAccess)
    assert isinstance(hexec1.body[1].left, oir.ScalarAccess)
    assert isinstance(hexec1.body[2].right, oir.ScalarAccess)
