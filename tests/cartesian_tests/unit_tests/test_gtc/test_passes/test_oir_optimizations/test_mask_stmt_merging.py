# GT4Py - GridTools Framework
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

from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.passes.oir_optimizations.mask_stmt_merging import MaskStmtMerging

from ...oir_utils import (
    AssignStmtFactory,
    FieldAccessFactory,
    HorizontalExecutionFactory,
    MaskStmtFactory,
)


def test_basic_merging():
    testee = HorizontalExecutionFactory(
        body=[
            MaskStmtFactory(mask=FieldAccessFactory(name="mask", dtype=common.DataType.BOOL)),
            MaskStmtFactory(mask=FieldAccessFactory(name="mask", dtype=common.DataType.BOOL)),
        ]
    )
    transformed = MaskStmtMerging().visit(testee)
    assert len(transformed.body) == 1
    assert transformed.body[0].mask == testee.body[0].mask
    assert transformed.body[0].body == testee.body[0].body + testee.body[1].body


def test_recursive_merging():
    testee = HorizontalExecutionFactory(
        body=[
            MaskStmtFactory(
                mask=FieldAccessFactory(name="mask", dtype=common.DataType.BOOL),
                body=[
                    MaskStmtFactory(
                        mask=FieldAccessFactory(name="mask2", dtype=common.DataType.BOOL)
                    ),
                    MaskStmtFactory(
                        mask=FieldAccessFactory(name="mask2", dtype=common.DataType.BOOL)
                    ),
                ],
            ),
            MaskStmtFactory(mask=FieldAccessFactory(name="mask", dtype=common.DataType.BOOL)),
        ]
    )
    transformed = MaskStmtMerging().visit(testee)
    assert len(transformed.body) == 1
    assert transformed.body[0].mask == testee.body[0].mask
    assert len(transformed.body[0].body) == 2
    assert transformed.body[0].body[0].mask == testee.body[0].body[0].mask


def test_not_merging_different_masks():
    testee = HorizontalExecutionFactory(
        body=[
            MaskStmtFactory(mask=FieldAccessFactory(name="mask", dtype=common.DataType.BOOL)),
            MaskStmtFactory(mask=FieldAccessFactory(name="mask2", dtype=common.DataType.BOOL)),
        ]
    )
    transformed = MaskStmtMerging().visit(testee)
    assert transformed == testee


def test_not_merging_rewritten_mask():
    testee = HorizontalExecutionFactory(
        body=[
            MaskStmtFactory(
                mask=FieldAccessFactory(name="mask", dtype=common.DataType.BOOL),
                body=[
                    AssignStmtFactory(
                        left__name="mask",
                        left__dtype=common.DataType.BOOL,
                        right__dtype=common.DataType.BOOL,
                    )
                ],
            ),
            MaskStmtFactory(mask=FieldAccessFactory(name="mask", dtype=common.DataType.BOOL)),
        ]
    )
    transformed = MaskStmtMerging().visit(testee)
    assert transformed == testee
