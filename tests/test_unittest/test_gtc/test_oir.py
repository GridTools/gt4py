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

from gtc.common import AxisBound, CartesianOffset, DataType, ExprKind, LoopOrder
from gtc.oir import (
    AssignStmt,
    Expr,
    FieldAccess,
    FieldDecl,
    HorizontalExecution,
    Interval,
    Stencil,
    Temporary,
    VerticalLoop,
)


A_ARITHMETIC_TYPE = DataType.INT32


class DummyExpr(Expr):
    """Fake expression for cases where a concrete expression is not needed."""

    kind: ExprKind = ExprKind.FIELD


def test_no_horizontal_offset_allowed():
    with pytest.raises(ValidationError, match=r"must not have .*horizontal offset"):
        AssignStmt(
            left=FieldAccess(
                name="foo", dtype=A_ARITHMETIC_TYPE, offset=CartesianOffset(i=1, j=0, k=0)
            ),
            right=DummyExpr(dtype=A_ARITHMETIC_TYPE),
        ),


def test_mask_must_be_bool():
    with pytest.raises(ValidationError, match=r".*must be.* bool.*"):
        HorizontalExecution(body=[], mask=DummyExpr(dtype=A_ARITHMETIC_TYPE)),


def test_temporary_default_3d():
    temp = Temporary(name="a", dtype=DataType.INT64)
    assert temp.dimensions == (True, True, True)

    temp1d = Temporary(name="b", dtype=DataType.INT64, dimensions=(True, False, False))
    assert temp1d.dimensions == (True, False, False)


def test_assign_to_ik_fwd():
    out_name = "ik_field"
    in_name = "other_ik_field"
    with pytest.raises(ValidationError, match=r"Not allowed to assign to ik-field"):
        Stencil(
            name="assign_to_ik_fwd",
            params=[
                FieldDecl(name=out_name, dtype=DataType.FLOAT32, dimensions=(True, False, True)),
                FieldDecl(name=in_name, dtype=DataType.FLOAT32, dimensions=(True, False, True)),
            ],
            vertical_loops=[
                VerticalLoop(
                    interval=Interval(start=AxisBound.start(), end=AxisBound.end()),
                    loop_order=LoopOrder.FORWARD,
                    declarations=[],
                    horizontal_executions=[
                        HorizontalExecution(
                            body=[
                                AssignStmt(
                                    left=FieldAccess(
                                        name=out_name,
                                        dtype=DataType.FLOAT32,
                                        offset=CartesianOffset.zero(),
                                    ),
                                    right=FieldAccess(
                                        name=in_name,
                                        dtype=DataType.FLOAT32,
                                        offset=CartesianOffset.zero(),
                                    ),
                                )
                            ],
                            declarations=[],
                        ),
                    ],
                ),
            ],
        )
