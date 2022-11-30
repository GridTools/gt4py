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

import pytest

from gtc.common import ArithmeticOperator, ComparisonOperator, DataType
from gtc.gtir import Decl, Expr, Stmt

from .gtir_utils import (
    BinaryOpFactory,
    FieldAccessFactory,
    FieldDeclFactory,
    FieldIfStmtFactory,
    HorizontalMaskFactory,
    ParAssignStmtFactory,
    ScalarAccessFactory,
    ScalarIfStmtFactory,
    StencilFactory,
    VariableKOffsetFactory,
    VerticalLoopFactory,
    WhileFactory,
)


ARITHMETIC_TYPE = DataType.FLOAT32
ANOTHER_ARITHMETIC_TYPE = DataType.INT32
A_ARITHMETIC_OPERATOR = ArithmeticOperator.ADD


def test_copy():
    copy_computation = StencilFactory(
        name="copy_gtir",
        vertical_loops__0=VerticalLoopFactory(
            body__0__left__name="foo", body__0__right__name="bar"
        ),
    )
    assert copy_computation
    assert set(copy_computation.param_names) == {"foo", "bar"}


@pytest.mark.parametrize("invalid_node", [Decl, Expr, Stmt])
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
    with pytest.raises(ValueError, match=r"must not have .*horizontal offset"):
        assign_stmt_with_offset()


def test_symbolref_without_decl():
    with pytest.raises(ValueError, match=r"Symbol.*not found"):
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
    with pytest.raises(ValueError, match=r"Illegal write.*read with.*offset"):
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
    with pytest.raises(ValueError, match=r"Self-assignment"):
        ParAssignStmtFactory(left__name="foo", right__name="foo", right__offset__i=1)


def test_indirect_address_data_dims():
    # Integer expressions are OK
    FieldAccessFactory(data_index=[ScalarAccessFactory(dtype=DataType.INT32)])

    # ... but others are not
    with pytest.raises(ValueError, match="must be integer expressions"):
        FieldAccessFactory(data_index=[ScalarAccessFactory(dtype=DataType.FLOAT32)])


def test_while_without_boolean_condition():
    with pytest.raises(ValueError, match=r"Condition in.*must be boolean."):
        WhileFactory(
            cond=BinaryOpFactory(left__name="foo", right__name="bar", op=ArithmeticOperator.ADD)
        )


def test_while_with_accumulated_extents():
    with pytest.raises(
        ValueError, match=r"Illegal write and read with horizontal offset detected for.*"
    ):
        WhileFactory(
            cond=BinaryOpFactory(
                left__name="a",
                right__name="b",
                op=ComparisonOperator.LT,
                dtype=DataType.BOOL,
            ),
            body=[
                ParAssignStmtFactory(left__name="a", right__name="b", right__offset__i=1),
                ParAssignStmtFactory(left__name="b", right__name="a"),
            ],
        )


def test_variable_k_offset_in_access():
    # Integer expressions are OK
    FieldAccessFactory(offset=VariableKOffsetFactory())

    # ... but others are not
    with pytest.raises(ValueError, match="must be an integer expression"):
        FieldAccessFactory(
            offset=VariableKOffsetFactory(k=FieldAccessFactory(dtype=DataType.FLOAT32))
        )


def test_visit_ScalarIf_HorizontalMask_fail():
    with pytest.raises(Exception):
        ScalarIfStmtFactory(cond=HorizontalMaskFactory())
