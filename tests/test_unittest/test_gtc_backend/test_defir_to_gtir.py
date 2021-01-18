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
from tests.definition_setup import ijk_domain  # noqa: F401
from tests.definition_setup import (
    BlockStmt,
    IterationOrder,
    TAssign,
    TComputationBlock,
    TDefinition,
    TFieldRef,
)

from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gt4py.ir.nodes import (
    AxisBound,
    AxisInterval,
    BinaryOperator,
    BinOpExpr,
    DataType,
    FieldDecl,
    LevelMarker,
    ScalarLiteral,
)
from gtc import common, gtir


@pytest.fixture
def defir_to_gtir():
    yield DefIRToGTIR()


def test_stencil_definition(
    defir_to_gtir, ijk_domain  # noqa: F811 [redefinition, reason: fixture]
):
    stencil_definition = (
        TDefinition(name="definition", domain=ijk_domain, fields=["a", "b"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("a", "b", (0, 0, 0)),
            )
        )
        .build()
    )
    gtir1 = defir_to_gtir.apply(stencil_definition)
    gtir2 = defir_to_gtir.visit_StencilDefinition(stencil_definition)
    assert gtir1 is not gtir2
    assert isinstance(gtir1, gtir.Stencil)
    assert isinstance(gtir2, gtir.Stencil)


def test_computation_block(defir_to_gtir):
    block = (
        TComputationBlock(order=IterationOrder.FORWARD)
        .add_statements(TAssign("a", "b", (0, 0, 0)))
        .build()
    )
    vertical_loop = defir_to_gtir.visit_ComputationBlock(block)
    assert isinstance(vertical_loop, gtir.VerticalLoop)


def test_block_stmt(defir_to_gtir):
    block_stmt = BlockStmt(stmts=[TAssign("a", "b", (0, 0, 0)).build()])
    statements = defir_to_gtir.visit_BlockStmt(block_stmt)
    assert isinstance(statements[0], gtir.Stmt)


def test_assign(defir_to_gtir):
    assign = TAssign("a", "b", (0, 0, 0)).build()
    assign_stmt = defir_to_gtir.visit_Assign(assign)
    assert isinstance(assign_stmt, gtir.ParAssignStmt)


def test_scalar_literal(defir_to_gtir):
    scalar_literal = ScalarLiteral(value=5, data_type=DataType.AUTO)
    literal = defir_to_gtir.visit_ScalarLiteral(scalar_literal)
    assert isinstance(literal, gtir.Literal)
    assert literal.value == "5"
    assert literal.dtype == common.DataType.AUTO


def test_bin_op_expr(defir_to_gtir):
    bin_op_expr = BinOpExpr(
        op=BinaryOperator.ADD, lhs=TFieldRef(name="a").build(), rhs=TFieldRef(name="b").build()
    )
    bin_op = defir_to_gtir.visit_BinOpExpr(bin_op_expr)
    assert isinstance(bin_op, gtir.BinaryOp)


def test_field_ref(defir_to_gtir):
    field_ref = TFieldRef(name="a", offset=(-1, 3, 0)).build()
    field_access = defir_to_gtir.visit_FieldRef(field_ref)
    assert isinstance(field_access, gtir.FieldAccess)
    assert field_access.name == "a"
    assert field_access.offset.i == -1
    assert field_access.offset.j == 3
    assert field_access.offset.k == 0


def test_axis_interval(defir_to_gtir):
    axis_interval = AxisInterval(
        start=AxisBound(level=LevelMarker.START, offset=0),
        end=AxisBound(level=LevelMarker.END, offset=1),
    )
    axis_start, axis_end = defir_to_gtir.visit_AxisInterval(axis_interval)
    assert isinstance(axis_start, gtir.AxisBound)
    assert isinstance(axis_end, gtir.AxisBound)


def test_axis_bound(defir_to_gtir):
    axis_bound = AxisBound(level=LevelMarker.START, offset=-51)
    gtir_bound = defir_to_gtir.visit_AxisBound(axis_bound)
    assert isinstance(gtir_bound, gtir.AxisBound)
    assert gtir_bound.level == common.LevelMarker.START
    assert gtir_bound.offset == -51


def test_field_decl(defir_to_gtir):
    field_decl = FieldDecl(name="a", data_type=DataType.BOOL, axes=["I", "J", "K"], is_api=True)
    gtir_decl = defir_to_gtir.visit_FieldDecl(field_decl)
    assert isinstance(gtir_decl, gtir.FieldDecl)
    assert gtir_decl.name == "a"
    assert gtir_decl.dtype == common.DataType.BOOL
