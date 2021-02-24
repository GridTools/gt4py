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
from gtc.common import ArithmeticOperator, DataType, LevelMarker, LoopOrder
from gtc.gtir import (
    AxisBound,
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
    BinaryOpFactory,
    FieldDeclFactory,
    FieldIfStmtFactory,
    ParAssignStmtFactory,
    StencilFactory,
    VerticalLoopFactory,
)


ARITHMETIC_TYPE = DataType.FLOAT32
ANOTHER_ARITHMETIC_TYPE = DataType.INT32
A_ARITHMETIC_OPERATOR = ArithmeticOperator.ADD


@pytest.fixture
def copy_assign():
    yield ParAssignStmt(
        loc=SourceLocation(line=3, column=2, source="copy_gtir"),
        left=FieldAccess.centered(
            name="foo", loc=SourceLocation(line=3, column=1, source="copy_gtir")
        ),
        right=FieldAccess.centered(
            name="bar", loc=SourceLocation(line=3, column=3, source="copy_gtir")
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
            FieldDecl(name="foo", dtype=DataType.FLOAT32),
            FieldDecl(name="bar", dtype=DataType.FLOAT32),
        ],
        vertical_loops=[copy_v_loop],
    )


def test_copy(copy_computation):
    assert copy_computation
    assert copy_computation.param_names == ["foo", "bar"]


@pytest.mark.parametrize(
    "invalid_node",
    [Decl, Expr, Stmt],
)
def test_abstract_classes_not_instantiatable(invalid_node):
    with pytest.raises(TypeError):
        invalid_node()


def test_can_have_vertical_offset():
    ParAssignStmtFactory(left__offset__k=1)


@pytest.mark.parametrize(
    "assign_stmt_with_offset",
    [
        lambda: ParAssignStmtFactory(left__offset__i=1),
        lambda: ParAssignStmtFactory(left__offset__j=1),
    ],
)
def test_no_horizontal_offset_allowed(assign_stmt_with_offset):
    with pytest.raises(ValidationError, match=r"must not have .*horizontal offset"):
        assign_stmt_with_offset()


def test_symbolref_without_decl():
    with pytest.raises(ValidationError, match=r"Symbols.*not found"):
        StencilFactory(
            params=[],
            vertical_loops__0__body__0=ParAssignStmtFactory(
                left__name="out_field", right__name="in_field"
            ),
        )


@pytest.mark.parametrize(
    "write_and_read_with_horizontal_offset",
    [
        lambda: VerticalLoopFactory(
            body=[
                ParAssignStmtFactory(right__name="foo", right__offset__i=1),
                ParAssignStmtFactory(left__name="foo"),
            ]
        ),
        # nested rhs
        lambda: VerticalLoopFactory(
            body=[
                ParAssignStmtFactory(
                    right=BinaryOpFactory(
                        left__name="foo",
                        right__name="foo",
                        right__offset__i=1,
                    )
                ),
                ParAssignStmtFactory(left__name="foo"),
            ]
        ),
        # offset access in condition
        lambda: VerticalLoopFactory(
            body=[
                FieldIfStmtFactory(
                    cond__name="foo",
                    cond__offset__i=1,
                ),
                ParAssignStmtFactory(left__name="foo"),
            ]
        ),
    ],
)
def test_write_and_read_with_offset_violation(write_and_read_with_horizontal_offset):
    with pytest.raises(ValidationError, match=r"Illegal write.*read with.*offset"):
        write_and_read_with_horizontal_offset()


def test_temporary_write_and_read_with_offset_is_allowed():
    VerticalLoopFactory(
        body=[
            ParAssignStmtFactory(right__name="foo", right__offset__i=1),
            ParAssignStmtFactory(left__name="foo"),
        ],
        temporaries=[FieldDeclFactory(name="foo")],
    )


def test_illegal_self_assignment_with_offset():
    with pytest.raises(ValidationError, match=r"Self-assignment"):
        ParAssignStmtFactory(left__name="foo", right__name="foo", right__offset__i=1)
