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

from gtc.common import DataType, LoopOrder

from .oir_utils import (
    AssignStmtFactory,
    FieldAccessFactory,
    FieldDeclFactory,
    HorizontalExecutionFactory,
    MaskStmtFactory,
    StencilFactory,
    VerticalLoopFactory,
    VerticalLoopSectionFactory,
)


def test_no_horizontal_offset_allowed():
    with pytest.raises(ValidationError, match=r"must not have .*horizontal offset"):
        AssignStmtFactory(left__offset__i=1)


def test_mask_must_be_bool():
    with pytest.raises(ValidationError, match=r".*must be.* bool.*"):
        MaskStmtFactory(mask=FieldAccessFactory(dtype=DataType.INT32))


def test_assign_to_ik_fwd():
    out_name = "ik_field"
    in_name = "other_ik_field"
    with pytest.raises(ValidationError, match=r"Not allowed to assign to ik-field"):
        StencilFactory(
            params=[
                FieldDeclFactory(
                    name=out_name, dtype=DataType.FLOAT32, dimensions=(True, False, True)
                ),
                FieldDeclFactory(
                    name=in_name, dtype=DataType.FLOAT32, dimensions=(True, False, True)
                ),
            ],
            vertical_loops__0=VerticalLoopFactory(
                loop_order=LoopOrder.FORWARD,
                sections=[
                    VerticalLoopSectionFactory(
                        horizontal_executions=[
                            HorizontalExecutionFactory(
                                body=[AssignStmtFactory(left__name=out_name, right__name=in_name)]
                            )
                        ]
                    )
                ],
            ),
        )
