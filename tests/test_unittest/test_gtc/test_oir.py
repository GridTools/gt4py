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
import functools
import itertools
from types import SimpleNamespace

import hypothesis as hyp
import hypothesis.strategies as hyp_st
import pytest
from pydantic.error_wrappers import ValidationError

from gtc.common import DataType, LevelMarker, LoopOrder
from gtc.oir import AxisBound, Interval, IntervalMapping

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


@functools.lru_cache(maxsize=None)
def get_instance(value: int) -> SimpleNamespace:
    return SimpleNamespace(value=value)


@hyp_st.composite
def intervals_strategy(draw):
    length = draw(hyp_st.integers(0, 5))
    intervals = []
    for _ in range(length):
        level1 = draw(hyp_st.sampled_from([LevelMarker.START, LevelMarker.END]))
        offset1 = draw(hyp_st.integers(-5, 5))
        bound1 = AxisBound(level=level1, offset=offset1)

        level2 = draw(hyp_st.sampled_from([LevelMarker.START, LevelMarker.END]))
        offset2 = draw(
            hyp_st.integers(-5, 5).filter(lambda x: x != offset1 if level1 == level2 else True)
        )
        bound2 = AxisBound(level=level2, offset=offset2)

        intervals.append(Interval(start=min(bound1, bound2), end=max(bound1, bound2)))
    return intervals


def test_no_horizontal_offset_allowed():
    with pytest.raises(ValidationError, match=r"must not have .*horizontal offset"):
        AssignStmtFactory(left__offset__i=1)


def test_mask_must_be_bool():
    with pytest.raises(ValidationError, match=r".*must be.* bool.*"):
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


class TestIntervalMapping:
    @staticmethod
    def assert_consistency(imap: IntervalMapping):
        assert len(imap.interval_starts) == len(imap.interval_ends)
        assert len(imap.interval_starts) == len(imap.values)
        for i in range(len(imap.interval_starts) - 1):
            assert imap.interval_starts[i] < imap.interval_starts[i + 1]
            assert imap.interval_ends[i] < imap.interval_ends[i + 1]
            assert imap.interval_ends[i] <= imap.interval_starts[i + 1]
            if imap.interval_ends[i] == imap.interval_starts[i + 1]:
                assert imap.values[i] is not imap.values[i + 1]

        for start, end in zip(imap.interval_starts, imap.interval_ends):
            assert start < end

    @pytest.mark.parametrize(
        ["intervals", "starts", "ends"],
        [
            ([], [], []),
            (
                [Interval(start=AxisBound.start(), end=AxisBound.end())],
                [AxisBound.start()],
                [AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
                ],
                [AxisBound.start()],
                [AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.end()),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                ],
                [AxisBound.start()],
                [AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_end(-1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                ],
                [AxisBound.start()],
                [AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
                ],
                [AxisBound.start()],
                [AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_start(2), end=AxisBound.end()),
                ],
                [AxisBound.start(), AxisBound.from_start(2)],
                [AxisBound.from_start(1), AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(2)),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
                ],
                [AxisBound.start()],
                [AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_end(-1)),
                ],
                [AxisBound.start()],
                [AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(2), end=AxisBound.from_end(-2)),
                ],
                [AxisBound.start(), AxisBound.from_start(2), AxisBound.from_end(-1)],
                [AxisBound.from_start(1), AxisBound.from_end(-2), AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_end(-2)),
                ],
                [AxisBound.start(), AxisBound.from_end(-1)],
                [AxisBound.from_end(-2), AxisBound.end()],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(2), end=AxisBound.from_end(-1)),
                ],
                [AxisBound.start(), AxisBound.from_start(2)],
                [AxisBound.from_start(1), AxisBound.end()],
            ),
        ],
    )
    def test_setitem_same_value(self, intervals, starts, ends):
        # if all values are the same (same instance), the behavior is the same as for a IntervalSet.
        imap = IntervalMapping()
        for interval in intervals:
            imap[interval] = 0
            self.assert_consistency(imap)

        assert len(starts) == len(imap.interval_starts)
        for expected, observed in zip(starts, imap.interval_starts):
            assert observed == expected

        assert len(ends) == len(imap.interval_ends)
        for expected, observed in zip(ends, imap.interval_ends):
            assert observed == expected

    @hyp.given(intervals_strategy())
    def test_setitem_same_value_hypothesis(self, intervals):
        imap = IntervalMapping()
        for interval in intervals:
            imap[interval] = 0
            self.assert_consistency(imap)

        for permutation in itertools.permutations(intervals):
            other_imap = IntervalMapping()
            for interval in permutation:
                other_imap[interval] = 0
                self.assert_consistency(other_imap)
            assert imap.interval_starts == other_imap.interval_starts
            assert imap.interval_ends == other_imap.interval_ends

    @pytest.mark.parametrize(
        ["intervals", "starts", "ends", "values"],
        [
            ([], [], [], []),
            (
                [Interval(start=AxisBound.start(), end=AxisBound.end())],
                [AxisBound.start()],
                [AxisBound.end()],
                [0],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
                ],
                [AxisBound.start(), AxisBound.from_start(1)],
                [AxisBound.from_start(1), AxisBound.end()],
                [0, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.end()),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                ],
                [AxisBound.start(), AxisBound.from_end(-1)],
                [AxisBound.from_end(-1), AxisBound.end()],
                [0, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_end(-1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                ],
                [AxisBound.start(), AxisBound.from_end(-1)],
                [AxisBound.from_end(-1), AxisBound.end()],
                [0, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
                ],
                [AxisBound.start(), AxisBound.from_start(1)],
                [AxisBound.from_start(1), AxisBound.end()],
                [0, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_start(2), end=AxisBound.end()),
                ],
                [AxisBound.start(), AxisBound.from_start(2)],
                [AxisBound.from_start(1), AxisBound.end()],
                [0, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(2)),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.end()),
                ],
                [AxisBound.start(), AxisBound.from_start(1)],
                [AxisBound.from_start(1), AxisBound.end()],
                [0, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_end(-1)),
                ],
                [AxisBound.start(), AxisBound.from_start(1), AxisBound.from_end(-1)],
                [AxisBound.from_start(1), AxisBound.from_end(-1), AxisBound.end()],
                [0, 2, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(2), end=AxisBound.from_end(-2)),
                ],
                [AxisBound.start(), AxisBound.from_start(2), AxisBound.from_end(-1)],
                [AxisBound.from_start(1), AxisBound.from_end(-2), AxisBound.end()],
                [0, 2, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_end(-2)),
                ],
                [AxisBound.start(), AxisBound.from_start(1), AxisBound.from_end(-1)],
                [AxisBound.from_start(1), AxisBound.from_end(-2), AxisBound.end()],
                [0, 2, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    Interval(start=AxisBound.from_end(-1), end=AxisBound.end()),
                    Interval(start=AxisBound.from_start(2), end=AxisBound.from_end(-1)),
                ],
                [AxisBound.start(), AxisBound.from_start(2), AxisBound.from_end(-1)],
                [AxisBound.from_start(1), AxisBound.from_end(-1), AxisBound.end()],
                [0, 2, 1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(3)),
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(2)),
                ],
                [AxisBound.start(), AxisBound.from_start(1), AxisBound.from_start(2)],
                [AxisBound.from_start(1), AxisBound.from_start(2), AxisBound.from_start(3)],
                [0, 1, 0],
            ),
            (
                [
                    Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(2)),
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(3)),
                ],
                [AxisBound.start()],
                [AxisBound.from_start(3)],
                [1],
            ),
            (
                [
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(2)),
                    Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                ],
                [AxisBound.start(), AxisBound.from_start(1)],
                [AxisBound.from_start(1), AxisBound.from_start(2)],
                [1, 0],
            ),
        ],
    )
    def test_setitem_different_value(self, intervals, starts, ends, values):
        imap = IntervalMapping()
        ctr = 0
        for interval in intervals:
            imap[interval] = get_instance(ctr)
            self.assert_consistency(imap)
            ctr = ctr + 1

        assert len(imap.interval_starts) == len(starts)
        assert len(imap.interval_ends) == len(ends)
        assert len(imap.values) == len(values)
        for i, (start, end, value) in enumerate(
            zip(imap.interval_starts, imap.interval_ends, imap.values)
        ):
            assert start == starts[i]
            assert end == ends[i]
            assert value is get_instance(values[i])

    @hyp.given(intervals_strategy())
    def test_setitem_different_value_hypothesis(self, intervals):
        ctr = 0
        imap = IntervalMapping()
        for interval in intervals:
            imap[interval] = get_instance(ctr)
            self.assert_consistency(imap)
            ctr += 1
        for permutation in itertools.permutations(intervals):
            other_imap = IntervalMapping()
            for interval in permutation:
                other_imap[interval] = get_instance(ctr)
                self.assert_consistency(other_imap)
                ctr += 1

            for start, end, value in zip(
                other_imap.interval_starts, other_imap.interval_ends, other_imap.values
            ):
                if start == permutation[-1].start:
                    assert end == permutation[-1].end
                    assert value is get_instance(ctr - 1)
                    break

    @pytest.mark.parametrize(
        ["interval", "values"],
        [
            (Interval(start=AxisBound.from_start(-1), end=AxisBound.from_end(1)), [0, 1]),
            (Interval(start=AxisBound.from_start(-1), end=AxisBound.from_start(3)), [0]),
            (Interval(start=AxisBound.from_start(1), end=AxisBound.from_end(-1)), [0, 1]),
            (Interval(start=AxisBound.from_start(2), end=AxisBound.from_end(-2)), []),
        ],
    )
    def test_getitem_different_value(self, interval, values):
        imap = IntervalMapping()
        imap[Interval(start=AxisBound.start(), end=AxisBound.from_start(2))] = get_instance(0)
        imap[Interval(start=AxisBound.from_end(-2), end=AxisBound.end())] = get_instance(1)
        res = imap[interval]
        assert isinstance(res, list)
        assert len(res) == len(values)
        for observed, expected in zip(res, values):
            assert observed is get_instance(expected)


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
