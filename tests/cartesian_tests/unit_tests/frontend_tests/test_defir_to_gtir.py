# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import numpy as np
import pytest

from gt4py.cartesian.frontend.defir_to_gtir import DefIRToGTIR, _make_literal
from gt4py.cartesian.frontend.nodes import (
    AxisBound,
    AxisInterval,
    BinaryOperator,
    BinOpExpr,
    DataType,
    FieldDecl,
    LevelMarker,
    ScalarLiteral,
)
from gt4py.cartesian.gtc import common, gtir

from cartesian_tests.unit_tests.frontend_tests.defir_to_gtir_definition_setup import (  # noqa: F401 [unused-import]
    BlockStmt,
    IterationOrder,
    TAssign,
    TComputationBlock,
    TDefinition,
    TFieldRef,
    ijk_domain,
)


@pytest.fixture
def defir_to_gtir():
    yield DefIRToGTIR()


def test_stencil_definition(
    defir_to_gtir,
    ijk_domain,  # [redefinition, reason: fixture]
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

    bin_op_expr = BinOpExpr(
        op=BinaryOperator.POW, lhs=TFieldRef(name="a").build(), rhs=TFieldRef(name="b").build()
    )
    native_func_call = defir_to_gtir.visit_BinOpExpr(bin_op_expr)
    assert isinstance(native_func_call, gtir.NativeFuncCall)
    assert native_func_call.func == common.NativeFunction.POW
    assert len(native_func_call.args) == 2
    assert native_func_call.args[0].name == "a"
    assert native_func_call.args[1].name == "b"


def test_field_ref(defir_to_gtir):
    field_ref = TFieldRef(name="a", offset=(-1, 3, 0)).build()
    field_access = defir_to_gtir.visit_FieldRef(field_ref)
    assert isinstance(field_access, gtir.FieldAccess)
    assert field_access.name == "a"
    assert field_access.offset.i == -1
    assert field_access.offset.j == 3
    assert field_access.offset.k == 0

    field_ref = TFieldRef(name="a", offset=(0, 0, TFieldRef(name="index").build())).build()
    field_access = defir_to_gtir.visit_FieldRef(field_ref)
    assert isinstance(field_access, gtir.FieldAccess)
    assert field_access.name == "a"

    assert isinstance(field_access.offset, gtir.VariableKOffset)
    assert isinstance(field_access.offset.k, gtir.FieldAccess)


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


@pytest.mark.parametrize(
    ["axes", "expected_mask"],
    [
        (["I", "J", "K"], (True, True, True)),
        (["I", "J"], (True, True, False)),
        (["I", "K"], (True, False, True)),
        (["J", "K"], (False, True, True)),
        (["I"], (True, False, False)),
        (["J"], (False, True, False)),
        (["K"], (False, False, True)),
    ],
)
def test_field_decl_dims(defir_to_gtir, axes, expected_mask):
    field_decl = FieldDecl(name="a", data_type=DataType.INT64, axes=axes, is_api=True)
    gtir_decl = defir_to_gtir.visit_FieldDecl(field_decl)
    assert gtir_decl.dimensions == expected_mask


def test_make_literal():
    # All of those are o.k.
    gtir_lit = _make_literal(10.10)
    assert gtir_lit.dtype == common.DataType.FLOAT64
    gtir_lit = _make_literal(np.float64(10.10))
    assert gtir_lit.dtype == common.DataType.FLOAT64
    gtir_lit = _make_literal(np.float32(10.10))
    assert gtir_lit.dtype == common.DataType.FLOAT32
    gtir_lit = _make_literal(10)
    assert gtir_lit.dtype == common.DataType.INT64
    gtir_lit = _make_literal(np.int64(10))
    assert gtir_lit.dtype == common.DataType.INT64
    gtir_lit = _make_literal(np.int32(10))
    assert gtir_lit.dtype == common.DataType.INT32
    gtir_lit = _make_literal(np.int16(10))
    assert gtir_lit.dtype == common.DataType.INT16
    gtir_lit = _make_literal(np.int8(10))
    assert gtir_lit.dtype == common.DataType.INT8
    gtir_lit = _make_literal(True)
    assert gtir_lit.dtype == common.DataType.BOOL
    gtir_lit = _make_literal(np.bool_(True))
    assert gtir_lit.dtype == common.DataType.BOOL
    # Not allowed
    with pytest.raises(TypeError):
        gtir_lit = _make_literal("a")
