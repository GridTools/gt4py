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

from gtc.common import ComparisonOperator, DataType, LoopOrder
from gtc.oir import BinaryOp, FieldAccess
from gtc.passes.oir_optimizations.inlining import MaskInlining

from ...oir_utils import (
    AssignStmtFactory,
    FieldAccessFactory,
    HorizontalExecutionFactory,
    IJCacheFactory,
    LiteralFactory,
    MaskStmtFactory,
    StencilFactory,
    TemporaryFactory,
    VerticalLoopFactory,
    VerticalLoopSectionFactory,
)


@pytest.fixture
def mask_cond() -> BinaryOp:
    return BinaryOp(
        op=ComparisonOperator.EQ,
        left=FieldAccessFactory(name="cond_f", offset__i=1),
        right=LiteralFactory(value="0"),
    )


@pytest.fixture
def mask_assign(mask_cond) -> AssignStmtFactory:
    return AssignStmtFactory(
        left__name="mask_f",
        left__dtype=DataType.BOOL,
        right=mask_cond,
    )


def test_mask_inlining(mask_assign):
    mask_name = mask_assign.left.name
    pre_oir = StencilFactory(
        vertical_loops__0=VerticalLoopFactory(
            loop_order=LoopOrder.PARALLEL,
            sections=[
                VerticalLoopSectionFactory(
                    horizontal_executions=[
                        HorizontalExecutionFactory(body=[mask_assign]),
                        HorizontalExecutionFactory(
                            body=[
                                MaskStmtFactory(
                                    mask=FieldAccessFactory(name=mask_name, dtype=DataType.BOOL),
                                    body=[
                                        AssignStmtFactory(left__name="out_f", right__name="in_f")
                                    ],
                                )
                            ]
                        ),
                    ]
                )
            ],
            caches=[IJCacheFactory(name=mask_name)],
        ),
        declarations=[
            TemporaryFactory(name=mask_name, dtype=DataType.BOOL),
        ],
    )

    pre_section = pre_oir.vertical_loops[0].sections[0]
    pre_mask = pre_section.horizontal_executions[1].body[0].mask
    assert isinstance(pre_mask, FieldAccess)
    assert pre_oir.declarations
    assert pre_oir.vertical_loops[0].caches

    post_oir = MaskInlining().visit(pre_oir)

    # FieldAccess becomes Expr (BinaryOp in this case)
    post_section = post_oir.vertical_loops[0].sections[0]
    post_mask = post_section.horizontal_executions[1].body[0].mask
    assert isinstance(post_mask, BinaryOp)

    # Cache is removed
    assert not post_oir.vertical_loops[0].caches

    # Temporary is removed
    assert not post_oir.declarations


def test_mask_no_inlining(mask_assign, mask_cond):
    cond_name = mask_cond.left.name
    mask_name = mask_assign.left.name
    pre_oir = StencilFactory(
        vertical_loops__0=VerticalLoopFactory(
            loop_order=LoopOrder.PARALLEL,
            sections=[
                VerticalLoopSectionFactory(
                    horizontal_executions=[
                        HorizontalExecutionFactory(body=[mask_assign]),
                        HorizontalExecutionFactory(
                            body=[
                                MaskStmtFactory(
                                    mask=FieldAccessFactory(name=mask_name, dtype=DataType.BOOL),
                                    body=[
                                        AssignStmtFactory(
                                            left__name=cond_name,
                                            right=LiteralFactory(),
                                        )
                                    ],
                                )
                            ]
                        ),
                    ]
                )
            ],
            caches=[IJCacheFactory(name=mask_name)],
        ),
        declarations=[
            TemporaryFactory(name=mask_name, dtype=DataType.BOOL),
        ],
    )

    pre_section = pre_oir.vertical_loops[0].sections[0]
    assert pre_section.horizontal_executions[0].body
    pre_mask = pre_section.horizontal_executions[1].body[0].mask
    assert isinstance(pre_mask, FieldAccess)
    assert pre_oir.declarations
    assert pre_oir.vertical_loops[0].caches

    post_oir = MaskInlining().visit(pre_oir)
    post_section = post_oir.vertical_loops[0].sections[0]
    assert post_section.horizontal_executions[0].body
    post_mask = post_section.horizontal_executions[1].body[0].mask
    assert isinstance(post_mask, FieldAccess)

    # Cache is not removed
    assert post_oir.vertical_loops[0].caches

    # Temporary is not removed
    assert post_oir.declarations
