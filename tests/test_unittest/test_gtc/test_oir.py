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

from gtc.common import CartesianOffset, DataType, ExprKind
from gtc.oir import AssignStmt, Expr, FieldAccess, HorizontalExecution


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
