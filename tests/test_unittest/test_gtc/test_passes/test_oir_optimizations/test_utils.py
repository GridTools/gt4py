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

from gt4py.definitions import Extent
from gtc.common import DataType
from gtc.passes.oir_optimizations.utils import AccessCollector, GeneralAccess, compute_extents

from ...oir_utils import (
    AssignStmtFactory,
    FieldAccessFactory,
    HorizontalExecutionFactory,
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
        GeneralAccess(field="foo", offset=(1, 0, 0), in_mask=False, is_write=False),
        GeneralAccess(field="tmp", offset=(0, 0, 0), in_mask=False, is_write=True),
        GeneralAccess(field="tmp", offset=(0, 0, 0), in_mask=False, is_write=False),
        GeneralAccess(field="bar", offset=(0, 0, 0), in_mask=False, is_write=True),
        GeneralAccess(field="mask", offset=(-1, -1, 1), in_mask=False, is_write=False),
        GeneralAccess(field="tmp", offset=(0, 1, 0), in_mask=True, is_write=False),
        GeneralAccess(field="baz", offset=(0, 0, 0), in_mask=True, is_write=True),
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
