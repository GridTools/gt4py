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

import pytest
from pydantic.error_wrappers import ValidationError

from eve import SourceLocation
from gtc.common import ArithmeticOperator, BuiltInLiteral, DataType, LevelMarker, LoopOrder
from gtc.gtir import (
    AxisBound,
    BinaryOp,
    CartesianOffset,
    Decl,
    Expr,
    FieldAccess,
    FieldDecl,
    Interval,
    ParAssignStmt,
    Stencil,
    Stmt,
    VerticalLoop,
)

from .gtir_utils import (
    DummyExpr,
    FieldAccessBuilder,
    FieldIfStmtBuilder,
    ParAssignStmtBuilder,
    StencilBuilder,
    VerticalLoopBuilder,
    make_Literal,
)


ARITHMETIC_TYPE = DataType.FLOAT32
ANOTHER_ARITHMETIC_TYPE = DataType.INT32
A_ARITHMETIC_OPERATOR = ArithmeticOperator.ADD


@pytest.fixture
def copy_assign():
    yield ParAssignStmt(
        loc=SourceLocation(line=3, column=2, source="copy_gtir"),
        left=FieldAccess.centered(
            name="a", loc=SourceLocation(line=3, column=1, source="copy_gtir")
        ),
        right=FieldAccess.centered(
            name="b", loc=SourceLocation(line=3, column=3, source="copy_gtir")
        ),
    )


@pytest.fixture
def interval(copy_assign):
    yield Interval(
        loc=SourceLocation(line=2, column=11, source="copy_gtir"),
        start=AxisBound(level=LevelMarker.START, offset=0),
        end=AxisBound(level=LevelMarker.END, offset=0),
    )


@pytest.fixture
def copy_v_loop(copy_assign, interval):
    yield VerticalLoop(
        loc=SourceLocation(line=2, column=1, source="copy_gtir"),
        loop_order=LoopOrder.FORWARD,
        interval=interval,
        body=[copy_assign],
        temporaries=[],
    )


@pytest.fixture
def copy_computation(copy_v_loop):
    yield Stencil(
        name="copy_gtir",
        loc=SourceLocation(line=1, column=1, source="copy_gtir"),
        params=[
            FieldDecl(name="a", dtype=DataType.FLOAT32),
            FieldDecl(name="b", dtype=DataType.FLOAT32),
        ],
        vertical_loops=[copy_v_loop],
    )


def test_copy(copy_computation):
    assert copy_computation
    assert copy_computation.param_names == ["a", "b"]


@pytest.mark.parametrize(
    "invalid_node",
    [Decl, Expr, Stmt],
)
def test_abstract_classes_not_instantiatable(invalid_node):
    with pytest.raises(TypeError):
        invalid_node()


def test_can_have_vertical_offset():
    ParAssignStmt(
        left=FieldAccessBuilder("foo").offset(CartesianOffset(i=0, j=0, k=1)).build(),
        right=DummyExpr(),
    )


@pytest.mark.parametrize(
    "assign_stmt_with_offset",
    [
        lambda: ParAssignStmt(
            left=FieldAccessBuilder("foo").offset(CartesianOffset(i=1, j=0, k=0)).build(),
            right=DummyExpr(),
        ),
        lambda: ParAssignStmt(
            left=FieldAccessBuilder("foo").offset(CartesianOffset(i=0, j=1, k=0)).build(),
            right=DummyExpr(),
        ),
    ],
)
def test_no_horizontal_offset_allowed(assign_stmt_with_offset):
    with pytest.raises(ValidationError, match=r"must not have .*horizontal offset"):
        assign_stmt_with_offset()


def test_symbolref_without_decl():
    with pytest.raises(ValidationError, match=r"Symbols.*not found"):
        StencilBuilder().add_par_assign_stmt(
            ParAssignStmtBuilder("out_field", "in_field").build()
        ).build()


@pytest.mark.parametrize(
    "write_and_read_with_horizontal_offset",
    [
        lambda: VerticalLoopBuilder()
        .add_stmt(
            ParAssignStmtBuilder("b")
            .right(FieldAccessBuilder("a").offset(CartesianOffset(i=1, j=0, k=0)).build())
            .build()
        )
        .add_stmt(
            ParAssignStmtBuilder("a").right(make_Literal("1.0", dtype=ARITHMETIC_TYPE)).build()
        )
        .build(),
        # nested rhs
        lambda: VerticalLoopBuilder()
        .add_stmt(
            ParAssignStmtBuilder("b")
            .right(
                BinaryOp(
                    op=A_ARITHMETIC_OPERATOR,
                    left=FieldAccessBuilder("a").build(),
                    right=FieldAccessBuilder("a").offset(CartesianOffset(i=1, j=0, k=0)).build(),
                )
            )
            .build()
        )
        .add_stmt(
            ParAssignStmtBuilder("a").right(make_Literal("1.0", dtype=ARITHMETIC_TYPE)).build()
        )
        .build(),
        # offset access in condition
        lambda: VerticalLoopBuilder()
        .add_stmt(
            FieldIfStmtBuilder()
            .cond(
                FieldAccessBuilder("a")
                .dtype(DataType.BOOL)
                .offset(CartesianOffset(i=1, j=0, k=0))
                .build()
            )
            .true_branch(
                [
                    ParAssignStmtBuilder("irrelevant")
                    .right(make_Literal("1.0", dtype=ARITHMETIC_TYPE))
                    .build()
                ]
            )
            .build()
        )
        .add_stmt(
            ParAssignStmtBuilder("a")
            .right(make_Literal(BuiltInLiteral.TRUE, dtype=DataType.BOOL))
            .build()
        )
        .build(),
    ],
)
def test_write_and_read_with_offset_violation(write_and_read_with_horizontal_offset):
    with pytest.raises(ValidationError, match=r"Illegal write.*read with.*offset"):
        write_and_read_with_horizontal_offset()


def test_temporary_write_and_read_with_offset_is_allowed():
    (
        VerticalLoopBuilder()
        .add_temporary("a", ARITHMETIC_TYPE)
        .add_stmt(
            ParAssignStmtBuilder("b")
            .right(FieldAccessBuilder("a").offset(CartesianOffset(i=1, j=0, k=0)).build())
            .build()
        )
        .add_stmt(
            ParAssignStmtBuilder("a").right(make_Literal("1.0", dtype=ARITHMETIC_TYPE)).build()
        )
        .build()
    )


def test_illegal_self_assignment_with_offset():
    with pytest.raises(ValidationError, match=r"Self-assignment"):
        ParAssignStmtBuilder("a").right(
            FieldAccessBuilder("a").offset(CartesianOffset(i=1, j=0, k=0)).build()
        ).build()
