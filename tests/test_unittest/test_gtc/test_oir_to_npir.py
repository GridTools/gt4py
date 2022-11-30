# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

import pytest

from gtc import common, oir
from gtc.definitions import Extent
from gtc.numpy import npir
from gtc.numpy.oir_to_npir import OirToNpir

from .oir_utils import (
    AssignStmtFactory,
    BinaryOpFactory,
    FieldAccessFactory,
    FieldDeclFactory,
    HorizontalExecutionFactory,
    LocalScalarFactory,
    MaskStmtFactory,
    NativeFuncCallFactory,
    ScalarAccessFactory,
    StencilFactory,
    VerticalLoopFactory,
    VerticalLoopSectionFactory,
)


def test_stencil_to_computation() -> None:
    stencil = StencilFactory(
        name="stencil",
        params=[
            FieldDeclFactory(
                name="a",
                dtype=common.DataType.FLOAT64,
            ),
            oir.ScalarDecl(
                name="b",
                dtype=common.DataType.INT32,
            ),
        ],
        vertical_loops__0__sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(
                left=FieldAccessFactory(name="a"), right=ScalarAccessFactory(name="b")
            )
        ],
    )
    computation = OirToNpir().visit(stencil)

    assert set(d.name for d in computation.api_field_decls) == {
        "a",
    }
    assert set(computation.arguments) == {"a", "b"}
    assert len(computation.vertical_passes) == 1


def test_vertical_loop_to_vertical_passes() -> None:
    vertical_loop = VerticalLoopFactory(sections__0__horizontal_executions=[])
    vertical_passes = OirToNpir().visit(vertical_loop)

    assert vertical_passes[0].body == []


def test_vertical_loop_section_to_vertical_pass() -> None:
    vertical_loop_section = VerticalLoopSectionFactory(horizontal_executions=[])
    vertical_pass = OirToNpir().visit(vertical_loop_section, loop_order=common.LoopOrder.PARALLEL)

    assert vertical_pass.body == []


def test_horizontal_execution_to_vector_assigns() -> None:
    horizontal_execution = HorizontalExecutionFactory(body=[])
    horizontal_block = OirToNpir().visit(horizontal_execution)
    assert horizontal_block.body == []


def test_mask_stmt_to_assigns() -> None:
    mask_stmt = MaskStmtFactory(body=[AssignStmtFactory()])
    assign_stmts = OirToNpir().visit(mask_stmt, extent=Extent.zeros(ndims=2))
    assert isinstance(assign_stmts[0].right.cond, npir.FieldSlice)
    assert len(assign_stmts) == 1


def test_mask_propagation() -> None:
    mask_stmt = MaskStmtFactory()
    assign_stmts = OirToNpir().visit(mask_stmt, extent=Extent.zeros(ndims=2))
    assert assign_stmts[0].right.cond == OirToNpir().visit(mask_stmt.mask)


def make_block_and_transform(**kwargs) -> npir.HorizontalBlock:
    oir_stencil = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[HorizontalExecutionFactory(**kwargs)]
    )

    return OirToNpir().visit(oir_stencil).vertical_passes[0].body[0]


def test_assign_stmt_broadcast() -> None:
    block = make_block_and_transform(
        body=[
            AssignStmtFactory(
                left=FieldAccessFactory(name="a"), right=ScalarAccessFactory(name="param")
            )
        ]
    )

    v_assign = block.body[0]
    assert isinstance(v_assign, npir.VectorAssign)
    assert isinstance(v_assign.left, npir.FieldSlice)
    assert v_assign.left.name == "a"
    assert isinstance(v_assign.right, npir.Broadcast)


def test_temp_assign() -> None:
    block = make_block_and_transform(
        body=[
            AssignStmtFactory(left=FieldAccessFactory(name="a"), right=FieldAccessFactory(name="b"))
        ],
        declarations=[LocalScalarFactory(name="a")],
    )
    assert set(d.name for d in block.declarations) == {"a"}


def test_field_access_to_field_slice_cartesian() -> None:
    field_access = FieldAccessFactory(offset__i=-1, offset__j=2, offset__k=0)
    field_slice = OirToNpir().visit(field_access)
    assert (field_slice.i_offset, field_slice.j_offset, field_slice.k_offset) == (-1, 2, 0)


def test_field_access_to_field_slice_variablek() -> None:
    field_access = FieldAccessFactory(
        offset=oir.VariableKOffset(k=oir.Literal(value="1", dtype=common.DataType.INT32))
    )
    field_slice = OirToNpir().visit(field_access)
    assert (field_slice.i_offset, field_slice.j_offset) == (0, 0)
    assert field_slice.k_offset.k.value == "1"


@pytest.mark.parametrize(
    "oir_node,npir_type",
    (
        (BinaryOpFactory(op=common.ArithmeticOperator.ADD), npir.VectorArithmetic),
        (
            BinaryOpFactory(
                op=common.LogicalOperator.AND,
                left__dtype=common.DataType.BOOL,
                right__dtype=common.DataType.BOOL,
            ),
            npir.VectorLogic,
        ),
    ),
)
def test_binary_op_to_npir(oir_node: oir.Expr, npir_type: Type[npir.Expr]) -> None:
    assert isinstance(OirToNpir().visit(oir_node), npir_type)


def test_literal_broadcast() -> None:
    result = OirToNpir().visit(
        AssignStmtFactory(
            left__dtype=common.DataType.FLOAT32,
            right=oir.Literal(value="42", dtype=common.DataType.FLOAT32),
        ),
        local_assigns={},
    )
    assert isinstance(result.right, npir.Broadcast)
    assert isinstance(result.right.expr, npir.ScalarLiteral)
    assert (result.right.expr.value, result.right.expr.dtype) == ("42", common.DataType.FLOAT32)


@pytest.mark.parametrize(
    "oir_int_expr, npir_type",
    (
        (oir.Literal(value="42", dtype=common.DataType.INT32), npir.ScalarCast),
        (FieldAccessFactory(dtype=common.DataType.INT32), npir.VectorCast),
    ),
)
def test_cast(oir_int_expr, npir_type) -> None:
    assert isinstance(
        OirToNpir().visit(oir.Cast(dtype=common.DataType.FLOAT64, expr=oir_int_expr)), npir_type
    )


def test_native_func_call() -> None:
    assert isinstance(
        OirToNpir().visit(NativeFuncCallFactory(args__0=FieldAccessFactory())), npir.NativeFuncCall
    )
