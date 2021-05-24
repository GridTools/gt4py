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

from gtc.common import ComparisonOperator, DataType, LoopOrder
from gtc.oir import BinaryOp, FieldAccess
from gtc.passes.oir_optimizations.inlining import MaskInlining

from ...oir_utils import (
    AssignStmtFactory,
    FieldAccessFactory,
    FieldDeclFactory,
    HorizontalExecutionFactory,
    LiteralFactory,
    MaskStmtFactory,
    StencilFactory,
    TemporaryFactory,
    VerticalLoopFactory,
    VerticalLoopSectionFactory,
)


def test_mask_inlining():
    out_name = "out_f"
    in_name = "in_f"
    cond_name = "cond_f"
    mask_name = "mask_f"

    pre_oir = StencilFactory(
        params=[
            FieldDeclFactory(name=out_name),
            FieldDeclFactory(name=in_name),
            FieldDeclFactory(name=cond_name),
        ],
        vertical_loops__0=VerticalLoopFactory(
            loop_order=LoopOrder.PARALLEL,
            sections=[
                VerticalLoopSectionFactory(
                    horizontal_executions=[
                        HorizontalExecutionFactory(
                            body=[
                                AssignStmtFactory(
                                    left__name=mask_name,
                                    left__dtype=DataType.BOOL,
                                    right=BinaryOp(
                                        op=ComparisonOperator.EQ,
                                        left=FieldAccessFactory(name=cond_name),
                                        right=LiteralFactory(value="0"),
                                    ),
                                )
                            ]
                        ),
                        HorizontalExecutionFactory(
                            body=[
                                MaskStmtFactory(
                                    mask=FieldAccessFactory(name=mask_name, dtype=DataType.BOOL),
                                    body=[
                                        AssignStmtFactory(left__name=out_name, right__name=in_name)
                                    ],
                                )
                            ]
                        ),
                    ]
                )
            ],
        ),
        declarations=[
            TemporaryFactory(name=mask_name, dtype=DataType.BOOL),
        ],
    )

    pre_section = pre_oir.vertical_loops[0].sections[0]
    assert pre_section.horizontal_executions[0].body
    pre_mask = pre_section.horizontal_executions[1].body[0].mask
    assert isinstance(pre_mask, FieldAccess)

    post_oir = MaskInlining().visit(pre_oir)
    post_section = post_oir.vertical_loops[0].sections[0]
    assert not post_section.horizontal_executions[0].body
    post_mask = post_section.horizontal_executions[1].body[0].mask
    assert isinstance(post_mask, BinaryOp)
