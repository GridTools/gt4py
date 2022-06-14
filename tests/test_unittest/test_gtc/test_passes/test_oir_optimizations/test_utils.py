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

from gtc import common
from gtc.common import DataType
from gtc.definitions import Extent
from gtc.passes.horizontal_masks import _overlap_along_axis, compute_relative_mask
from gtc.passes.oir_optimizations.utils import (
    AccessCollector,
    GeneralAccess,
    compute_extents,
    compute_horizontal_block_extents,
)

from ...oir_utils import (
    AssignStmtFactory,
    FieldAccessFactory,
    HorizontalExecutionFactory,
    HorizontalRestrictionFactory,
    MaskStmtFactory,
    StencilFactory,
    TemporaryFactory,
)


def test_access_collector():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(left__name="tmp", right__name="foo", right__offset__i=1),
                    AssignStmtFactory(left__name="bar", right__name="tmp"),
                ]
            ),
            HorizontalExecutionFactory(
                body=[
                    MaskStmtFactory(
                        body=[
                            AssignStmtFactory(
                                left__name="baz", right__name="tmp", right__offset__j=1
                            ),
                        ],
                        mask=FieldAccessFactory(
                            name="mask",
                            dtype=DataType.BOOL,
                            offset__i=-1,
                            offset__j=-1,
                            offset__k=1,
                        ),
                    )
                ],
            ),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    read_offsets = {"tmp": {(0, 0, 0), (0, 1, 0)}, "foo": {(1, 0, 0)}, "mask": {(-1, -1, 1)}}
    write_offsets = {"tmp": {(0, 0, 0)}, "bar": {(0, 0, 0)}, "baz": {(0, 0, 0)}}
    offsets = {
        "tmp": {(0, 0, 0), (0, 1, 0)},
        "foo": {(1, 0, 0)},
        "bar": {(0, 0, 0)},
        "baz": {(0, 0, 0)},
        "mask": {(-1, -1, 1)},
    }
    ordered_accesses = [
        GeneralAccess(field="foo", offset=(1, 0, 0), is_write=False),
        GeneralAccess(field="tmp", offset=(0, 0, 0), is_write=True),
        GeneralAccess(field="tmp", offset=(0, 0, 0), is_write=False),
        GeneralAccess(field="bar", offset=(0, 0, 0), is_write=True),
        GeneralAccess(field="mask", offset=(-1, -1, 1), is_write=False),
        GeneralAccess(field="tmp", offset=(0, 1, 0), is_write=False),
        GeneralAccess(field="baz", offset=(0, 0, 0), is_write=True),
    ]

    result = AccessCollector.apply(testee)
    assert result.read_offsets() == read_offsets
    assert result.write_offsets() == write_offsets
    assert result.offsets() == offsets
    assert result.ordered_accesses() == ordered_accesses


def test_stencil_extents_simple():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="tmp", right__name="input", right__offset__i=1)]
            ),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="output", right__name="tmp", right__offset__i=1)]
            ),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )

    field_extents, block_extents = compute_extents(testee)

    assert field_extents["input"] == Extent((1, 2), (0, 0))
    assert field_extents["output"] == Extent((0, 0), (0, 0))

    hexecs = testee.vertical_loops[0].sections[0].horizontal_executions
    assert block_extents[id(hexecs[0])] == Extent((0, 1), (0, 0))
    assert block_extents[id(hexecs[1])] == Extent((0, 0), (0, 0))


def test_access_overlap_along_axis():
    assert _overlap_along_axis((0, 0), common.HorizontalInterval.compute_domain()) == (0, 0)
    assert _overlap_along_axis(
        (0, 0), common.HorizontalInterval.compute_domain(start_offset=-1, end_offset=1)
    ) == (0, 0)

    overlap = _overlap_along_axis(
        (0, 0), common.HorizontalInterval.at_endpt(common.LevelMarker.START, 2)
    )

    assert overlap[0] == -2
    assert overlap[1] > 100

    assert (
        _overlap_along_axis(
            (0, 0), common.HorizontalInterval.at_endpt(common.LevelMarker.START, -4)
        )
        is None
    )

    assert (
        _overlap_along_axis((0, 0), common.HorizontalInterval.at_endpt(common.LevelMarker.END, 4))
        is None
    )

    overlap = _overlap_along_axis(
        (-1, 1),
        common.HorizontalInterval.at_endpt(common.LevelMarker.START, start_offset=-4, end_offset=4),
    )

    assert overlap[0] == 0
    assert overlap[1] > 100

    overlap = _overlap_along_axis(
        (-1, 1),
        common.HorizontalInterval.at_endpt(common.LevelMarker.END, start_offset=-4, end_offset=4),
    )

    assert overlap[0] < -100
    assert overlap[1] == 0


@pytest.mark.parametrize(
    "mask,offset,access_extent",
    (
        (
            common.HorizontalMask(
                i=common.HorizontalInterval.at_endpt(common.LevelMarker.END, 1),
                j=common.HorizontalInterval.full(),
            ),
            1,
            ((0, 2), (0, 0)),
        ),
        (
            common.HorizontalMask(
                i=common.HorizontalInterval.at_endpt(common.LevelMarker.END, 1),
                j=common.HorizontalInterval.full(),
            ),
            -1,
            ((0, 0), (0, 0)),
        ),
        (
            common.HorizontalMask(
                i=common.HorizontalInterval.at_endpt(common.LevelMarker.END, 2),
                j=common.HorizontalInterval.full(),
            ),
            0,
            None,
        ),
        (
            common.HorizontalMask(
                i=common.HorizontalInterval.full(),
                j=common.HorizontalInterval.full(),
            ),
            -1,
            ((-1, 0), (0, 0)),
        ),
    ),
)
def test_stencil_extents_region(mask, offset, access_extent):
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="tmp", right__name="input")]
            ),
            HorizontalExecutionFactory(
                body=[
                    HorizontalRestrictionFactory(
                        mask=mask,
                        body=[
                            AssignStmtFactory(
                                left__name="tmp", right__name="input", right__offset__i=offset
                            )
                        ],
                    ),
                ]
            ),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="output", right__name="tmp", right__offset__i=1)]
            ),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )

    block_extents = compute_horizontal_block_extents(testee)
    hexecs = testee.vertical_loops[0].sections[0].horizontal_executions
    mask_read_accesses = AccessCollector.apply(hexecs[1].body[0])
    input_access = next(
        iter(acc for acc in mask_read_accesses.ordered_accesses() if acc.field == "input")
    )

    block_extent = ((0, 1), (0, 0))
    assert block_extents[id(hexecs[1])] == block_extent
    if access_extent is not None:
        assert input_access.to_extent(Extent(block_extent)) == access_extent
    else:
        assert input_access.to_extent(Extent(block_extent)) is None


def test_compute_relative_mask():
    relative_mask = compute_relative_mask(
        Extent.zeros(ndims=2),
        common.HorizontalMask(
            i=common.HorizontalInterval.compute_domain(start_offset=-1, end_offset=1),
            j=common.HorizontalInterval.full(),
        ),
    )

    assert relative_mask.i == common.HorizontalInterval.compute_domain()
    assert relative_mask.j == common.HorizontalInterval.compute_domain()

    relative_mask = compute_relative_mask(
        Extent.zeros(ndims=2),
        common.HorizontalMask(
            i=common.HorizontalInterval.at_endpt(
                level=common.LevelMarker.START, start_offset=-2, end_offset=3
            ),
            j=common.HorizontalInterval.full(),
        ),
    )

    assert relative_mask.i == common.HorizontalInterval.at_endpt(
        level=common.LevelMarker.START, start_offset=0, end_offset=3
    )
    assert relative_mask.j == common.HorizontalInterval.compute_domain()
