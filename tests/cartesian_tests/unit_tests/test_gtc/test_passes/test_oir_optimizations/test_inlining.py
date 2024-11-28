# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.cartesian.gtc.common import ComparisonOperator, DataType, LoopOrder
from gt4py.cartesian.gtc.oir import BinaryOp, FieldAccess
from gt4py.cartesian.gtc.passes.oir_optimizations.inlining import MaskInlining

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
    return AssignStmtFactory(left__name="mask_f", left__dtype=DataType.BOOL, right=mask_cond)


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
        declarations=[TemporaryFactory(name=mask_name, dtype=DataType.BOOL)],
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
                                            left__name=cond_name, right=LiteralFactory()
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
        declarations=[TemporaryFactory(name=mask_name, dtype=DataType.BOOL)],
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
