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

from gtc.common import DataType, LoopOrder
from gtc.oir import AxisBound, Interval

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
    with pytest.raises(ValueError, match=r"must not have .*horizontal offset"):
        AssignStmtFactory(left__offset__i=1)


def test_mask_must_be_bool():
    with pytest.raises(ValueError, match=r".*must be.* bool.*"):
        MaskStmtFactory(mask=FieldAccessFactory(dtype=DataType.INT32))


EQUAL_AXISBOUNDS = [
    (AxisBound.start(), AxisBound.start()),
    (AxisBound.end(), AxisBound.end()),
    (AxisBound.from_end(-1), AxisBound.from_end(-1)),
]
LESS_AXISBOUNDS = [
    (AxisBound.start(), AxisBound.end()),
    (AxisBound.start(), AxisBound.from_start(1)),
    (AxisBound.from_end(-1), AxisBound.end()),
    (AxisBound.from_start(1), AxisBound.from_end(-1)),
]
GREATER_AXISBOUNDS = [
    (AxisBound.end(), AxisBound.start()),
    (AxisBound.from_start(1), AxisBound.start()),
    (AxisBound.end(), AxisBound.from_end(-1)),
    (AxisBound.from_end(-1), AxisBound.from_start(1)),
]


class TestAxisBoundsComparison:
    @pytest.mark.parametrize(["lhs", "rhs"], EQUAL_AXISBOUNDS)
    def test_eq_true(self, lhs, rhs):
        res1 = lhs == rhs
        assert isinstance(res1, bool)
        assert res1

        res2 = rhs == lhs
        assert isinstance(res2, bool)
        assert res2

    @pytest.mark.parametrize(
        ["lhs", "rhs"],
        LESS_AXISBOUNDS + GREATER_AXISBOUNDS,
    )
    def test_eq_false(self, lhs, rhs):
        res1 = lhs == rhs
        assert isinstance(res1, bool)
        assert not res1

        res2 = rhs == lhs
        assert isinstance(res2, bool)
        assert not res2

    @pytest.mark.parametrize(["lhs", "rhs"], LESS_AXISBOUNDS)
    def test_lt_true(self, lhs, rhs):
        res = lhs < rhs
        assert isinstance(res, bool)
        assert res

    @pytest.mark.parametrize(
        ["lhs", "rhs"],
        GREATER_AXISBOUNDS + EQUAL_AXISBOUNDS,
    )
    def test_lt_false(self, lhs, rhs):
        res = lhs < rhs
        assert isinstance(res, bool)
        assert not res

    @pytest.mark.parametrize(["lhs", "rhs"], GREATER_AXISBOUNDS)
    def test_gt_true(self, lhs, rhs):
        res = lhs > rhs
        assert isinstance(res, bool)
        assert res

    @pytest.mark.parametrize(["lhs", "rhs"], LESS_AXISBOUNDS + EQUAL_AXISBOUNDS)
    def test_gt_false(self, lhs, rhs):
        res = lhs > rhs
        assert isinstance(res, bool)
        assert not res

    @pytest.mark.parametrize(["lhs", "rhs"], LESS_AXISBOUNDS + EQUAL_AXISBOUNDS)
    def test_le_true(self, lhs, rhs):
        res = lhs <= rhs
        assert isinstance(res, bool)
        assert res

    @pytest.mark.parametrize(["lhs", "rhs"], GREATER_AXISBOUNDS)
    def test_le_false(self, lhs, rhs):
        res = lhs <= rhs
        assert isinstance(res, bool)
        assert not res

    @pytest.mark.parametrize(["lhs", "rhs"], GREATER_AXISBOUNDS + EQUAL_AXISBOUNDS)
    def test_ge_true(self, lhs, rhs):
        res = lhs >= rhs
        assert isinstance(res, bool)
        assert res

    @pytest.mark.parametrize(["lhs", "rhs"], LESS_AXISBOUNDS)
    def test_ge_false(self, lhs, rhs):
        res = lhs >= rhs
        assert isinstance(res, bool)
        assert not res


COVER_INTERVALS = [
    (
        Interval(start=AxisBound.start(), end=AxisBound.end()),
        Interval(start=AxisBound.start(), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.start(), end=AxisBound.end()),
        Interval(start=AxisBound.start(), end=AxisBound.from_end(-1)),
    ),
    (
        Interval(start=AxisBound.start(), end=AxisBound.end()),
        Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.start(), end=AxisBound.end()),
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_end(-1)),
    ),
    (
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
    ),
    (
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
        Interval(start=AxisBound.from_start(2), end=AxisBound.from_start(4)),
    ),
    (
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(3)),
    ),
    (
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
        Interval(start=AxisBound.from_start(2), end=AxisBound.from_start(3)),
    ),
]
SUBSET_INTERVALS = [
    (
        Interval(start=AxisBound.start(), end=AxisBound.from_end(-1)),
        Interval(start=AxisBound.start(), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
        Interval(start=AxisBound.start(), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_end(-1)),
        Interval(start=AxisBound.start(), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.from_start(2), end=AxisBound.from_start(4)),
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
    ),
    (
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(3)),
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
    ),
    (
        Interval(start=AxisBound.from_start(2), end=AxisBound.from_start(3)),
        Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(4)),
    ),
]
DISJOINT_INTERVALS = [
    (
        Interval(start=AxisBound.start(), end=AxisBound.from_end(-1)),
        Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
        Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.start(), end=AxisBound.from_start(3)),
        Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.start(), end=AxisBound.from_start(3)),
        Interval(start=AxisBound.from_end(1), end=AxisBound.from_end(2)),
    ),
]
OVERLAP_INTERVALS = [
    (
        Interval(start=AxisBound.start(), end=AxisBound.from_end(-1)),
        Interval(start=AxisBound.from_end(-2), end=AxisBound.end()),
    ),
    (
        Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
        Interval(start=AxisBound.start(), end=AxisBound.from_start(2)),
    ),
]


class TestIntervalOperations:
    @pytest.mark.parametrize(["lhs", "rhs"], COVER_INTERVALS)
    def test_covers_true(self, lhs, rhs):
        res = lhs.covers(rhs)
        assert isinstance(res, bool)
        assert res

    @pytest.mark.parametrize(
        ["lhs", "rhs"], SUBSET_INTERVALS + OVERLAP_INTERVALS + DISJOINT_INTERVALS
    )
    def test_covers_false(self, lhs, rhs):
        res = lhs.covers(rhs)
        assert isinstance(res, bool)
        assert not res

    @pytest.mark.parametrize(["lhs", "rhs"], COVER_INTERVALS + SUBSET_INTERVALS + OVERLAP_INTERVALS)
    def test_intersects_true(self, lhs, rhs):
        res1 = lhs.intersects(rhs)
        assert isinstance(res1, bool)
        assert res1

        res2 = lhs.intersects(rhs)
        assert isinstance(res2, bool)
        assert res2

    @pytest.mark.parametrize(["lhs", "rhs"], DISJOINT_INTERVALS)
    def test_intersects_false(self, lhs, rhs):
        res1 = lhs.intersects(rhs)
        assert isinstance(res1, bool)
        assert not res1

        res2 = rhs.intersects(lhs)
        assert isinstance(res2, bool)
        assert not res2


def test_assign_to_ik_fwd():
    out_name = "ik_field"
    in_name = "other_ik_field"
    with pytest.raises(ValueError, match=r"Not allowed to assign to ik-field"):
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
