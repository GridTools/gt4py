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

from gtc.common import DataType
from gtc.passes.oir_optimizations.utils import Access, AccessCollector

from ...oir_utils import (
    AssignStmtFactory,
    FieldAccessFactory,
    HorizontalExecutionFactory,
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
                    AssignStmtFactory(left__name="baz", right__name="tmp", right__offset__j=1),
                ],
                mask=FieldAccessFactory(
                    name="mask", dtype=DataType.BOOL, offset__i=-1, offset__j=-1, offset__k=1
                ),
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
        Access(field="foo", offset=(1, 0, 0), is_write=False),
        Access(field="tmp", offset=(0, 0, 0), is_write=True),
        Access(field="tmp", offset=(0, 0, 0), is_write=False),
        Access(field="bar", offset=(0, 0, 0), is_write=True),
        Access(field="mask", offset=(-1, -1, 1), is_write=False),
        Access(field="tmp", offset=(0, 1, 0), is_write=False),
        Access(field="baz", offset=(0, 0, 0), is_write=True),
    ]

    result = AccessCollector.apply(testee)
    assert result.read_offsets() == read_offsets
    assert result.write_offsets() == write_offsets
    assert result.offsets() == offsets
    assert result.ordered_accesses() == ordered_accesses
