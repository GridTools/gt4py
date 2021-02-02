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

from typing import Type

from eve import Node
from gtc import gtir, gtir_to_oir, oir
from gtc.common import BlockStmt, DataType, ExprKind
from gtc.gtir import ScalarIfStmt
from gtc.gtir_to_oir import GTIRToOIR

from . import gtir_utils
from .gtir_utils import FieldAccessBuilder, FieldIfStmtBuilder


A_ARITHMETIC_TYPE = DataType.FLOAT32


class DummyExpr(oir.Expr):
    dtype = A_ARITHMETIC_TYPE
    kind = ExprKind.FIELD


def isinstance_and_return(node: Node, expected_type: Type[Node]):
    assert isinstance(node, expected_type)
    return node


def test_visit_ParAssignStmt():
    out_name = "out"
    in_name = "in"
    testee = gtir.ParAssignStmt(
        left=FieldAccessBuilder(out_name).build(), right=FieldAccessBuilder(in_name).build()
    )

    ctx = GTIRToOIR.Context()
    GTIRToOIR().visit(testee, ctx=ctx)
    result_decls = ctx.decls
    result_horizontal_executions = ctx.horizontal_executions

    assert len(result_decls) == 1
    assert isinstance(result_decls[0], oir.Temporary)
    tmp_name = result_decls[0].name

    assert len(result_horizontal_executions) == 2
    first_assign = isinstance_and_return(result_horizontal_executions[0].body[0], oir.AssignStmt)
    second_assign = isinstance_and_return(result_horizontal_executions[1].body[0], oir.AssignStmt)

    first_left = isinstance_and_return(first_assign.left, oir.FieldAccess)
    first_right = isinstance_and_return(first_assign.right, oir.FieldAccess)
    assert first_left.name == tmp_name
    assert first_right.name == in_name

    second_left = isinstance_and_return(second_assign.left, oir.FieldAccess)
    second_right = isinstance_and_return(second_assign.right, oir.FieldAccess)
    assert second_left.name == out_name
    assert second_right.name == tmp_name


def test_create_mask():
    mask_name = "mask"
    cond = DummyExpr(dtype=DataType.BOOL)
    ctx = GTIRToOIR.Context()
    result_decl = gtir_to_oir._create_mask(ctx, mask_name, cond)
    result_assign = ctx.horizontal_executions[0]

    assert isinstance(result_decl, oir.Temporary)
    assert result_decl.name == mask_name

    horizontal_exec = isinstance_and_return(result_assign, oir.HorizontalExecution)
    assign = isinstance_and_return(horizontal_exec.body[0], oir.AssignStmt)

    left = isinstance_and_return(assign.left, oir.FieldAccess)
    right = isinstance_and_return(assign.right, DummyExpr)

    assert left.name == mask_name
    assert right == cond


def test_visit_FieldIfStmt():
    testee = (
        FieldIfStmtBuilder()
        .cond(FieldAccessBuilder("cond").dtype(DataType.BOOL).build())
        .false_branch([])
        .build()
    )
    GTIRToOIR().visit(testee, ctx=GTIRToOIR.Context())


def test_visit_FieldIfStmt_no_else():
    testee = (
        FieldIfStmtBuilder().cond(FieldAccessBuilder("cond").dtype(DataType.BOOL).build()).build()
    )
    GTIRToOIR().visit(testee, ctx=GTIRToOIR.Context())


def test_visit_FieldIfStmt_nesting():
    testee = (
        FieldIfStmtBuilder()
        .cond(FieldAccessBuilder("cond").dtype(DataType.BOOL).build())
        .add_true_stmt(
            FieldIfStmtBuilder()
            .cond(FieldAccessBuilder("cond2").dtype(DataType.BOOL).build())
            .build()
        )
        .build()
    )
    GTIRToOIR().visit(testee, ctx=GTIRToOIR.Context())


def test_visit_ScalarIfStmt():
    testee = ScalarIfStmt(
        cond=gtir_utils.DummyExpr(dtype=DataType.BOOL, kind=ExprKind.SCALAR),
        true_branch=BlockStmt(body=[]),
    )
    GTIRToOIR().visit(testee, ctx=GTIRToOIR.Context())
