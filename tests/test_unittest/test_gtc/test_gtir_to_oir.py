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
from gtc.common import DataType
from gtc.gtir_to_oir import GTIRToOIR

from . import oir_utils
from .gtir_utils import (
    BlockStmtFactory,
    FieldAccessFactory,
    FieldIfStmtFactory,
    ScalarIfStmtFactory,
)


def isinstance_and_return(node: Node, expected_type: Type[Node]):
    assert isinstance(node, expected_type)
    return node


def test_visit_ParAssignStmt():
    out_name = "out"
    in_name = "in"
    testee = gtir.ParAssignStmt(
        left=FieldAccessFactory(name=out_name), right=FieldAccessFactory(name=in_name)
    )

    ctx = GTIRToOIR.Context()
    GTIRToOIR().visit(testee, ctx=ctx)
    result_horizontal_executions = ctx.horizontal_executions

    assert len(result_horizontal_executions) == 1
    assign = isinstance_and_return(result_horizontal_executions[0].body[0], oir.AssignStmt)

    left = isinstance_and_return(assign.left, oir.FieldAccess)
    right = isinstance_and_return(assign.right, oir.FieldAccess)
    assert left.name == out_name
    assert right.name == in_name


def test_create_mask():
    mask_name = "mask"
    cond = oir_utils.FieldAccessFactory(dtype=DataType.BOOL)
    ctx = GTIRToOIR.Context()
    result_decl = gtir_to_oir._create_mask(ctx, mask_name, cond)
    result_assign = ctx.horizontal_executions[0]

    assert isinstance(result_decl, oir.Temporary)
    assert result_decl.name == mask_name

    horizontal_exec = isinstance_and_return(result_assign, oir.HorizontalExecution)
    assign = isinstance_and_return(horizontal_exec.body[0], oir.AssignStmt)

    left = isinstance_and_return(assign.left, oir.FieldAccess)
    right = isinstance_and_return(assign.right, oir.FieldAccess)

    assert left.name == mask_name
    assert right == cond


def test_visit_FieldIfStmt():
    testee = FieldIfStmtFactory(false_branch=BlockStmtFactory())
    GTIRToOIR().visit(testee, ctx=GTIRToOIR.Context())


def test_visit_FieldIfStmt_no_else():
    testee = FieldIfStmtFactory(false_branch=None)
    GTIRToOIR().visit(testee, ctx=GTIRToOIR.Context())


def test_visit_FieldIfStmt_nesting():
    testee = FieldIfStmtFactory(true_branch=BlockStmtFactory(body=[FieldIfStmtFactory()]))
    GTIRToOIR().visit(testee, ctx=GTIRToOIR.Context())


def test_visit_ScalarIfStmt():
    testee = ScalarIfStmtFactory()
    GTIRToOIR().visit(testee, ctx=GTIRToOIR.Context())
