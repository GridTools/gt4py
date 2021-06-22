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

from gtc.cuir import cuir, extent_analysis

from .cuir_utils import (
    AssignStmtFactory,
    HorizontalExecutionFactory,
    IJCacheAccessFactory,
    IJCacheDeclFactory,
    IJExtentFactory,
    ProgramFactory,
    TemporaryFactory,
    VerticalLoopFactory,
)


def test_compute_extents():
    testee = ProgramFactory(
        kernels__0__vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="tmp")],
                extent=None,
            ),
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(right__name="tmp", right__offset__i=1, right__offset__j=-3)
                ],
                extent=None,
            ),
        ],
        temporaries=[TemporaryFactory(name="tmp")],
    )
    transformed = extent_analysis.ComputeExtents().visit(testee)
    hexecs = transformed.kernels[0].vertical_loops[0].sections[0].horizontal_executions
    assert hexecs[0].extent.i == (0, 1)
    assert hexecs[0].extent.j == (-3, 0)
    assert hexecs[1].extent.i == (0, 0)
    assert hexecs[1].extent.j == (0, 0)


def test_compute_extents_with_multiple_loops():
    testee = ProgramFactory(
        kernels__0__vertical_loops=[
            VerticalLoopFactory(
                sections__0__horizontal_executions=[
                    HorizontalExecutionFactory(
                        body=[AssignStmtFactory(right__name="tmp", right__offset__i=1)],
                        extent=None,
                    ),
                ],
            ),
            VerticalLoopFactory(
                sections__0__horizontal_executions=[
                    HorizontalExecutionFactory(
                        body=[
                            AssignStmtFactory(
                                left__name="tmp",
                            )
                        ],
                        extent=None,
                    ),
                ]
            ),
        ],
        params=[TemporaryFactory(name="tmp")],
    )
    transformed = extent_analysis.ComputeExtents().visit(testee)
    hexecs = transformed.iter_tree().if_isinstance(cuir.HorizontalExecution).to_list()
    assert hexecs[0].extent.i == (0, 0)
    assert hexecs[0].extent.j == (0, 0)
    assert hexecs[1].extent.i == (0, 0)
    assert hexecs[1].extent.j == (0, 0)


def test_cache_extents():
    testee = ProgramFactory(
        kernels__0__vertical_loops__0=VerticalLoopFactory(
            sections__0__horizontal_executions=[
                HorizontalExecutionFactory(
                    body=[AssignStmtFactory(left=IJCacheAccessFactory(name="tmp"))],
                    extent=IJExtentFactory(i=(0, 1), j=(-3, 0)),
                ),
                HorizontalExecutionFactory(
                    body=[
                        AssignStmtFactory(
                            right=IJCacheAccessFactory(name="tmp", offset__i=1, offset__j=-3)
                        )
                    ],
                    extent=IJExtentFactory(i=(0, 0), j=(0, 0)),
                ),
            ],
            ij_caches=[IJCacheDeclFactory(name="tmp", extent=None)],
        )
    )
    transformed = extent_analysis.CacheExtents().visit(testee)
    cache = transformed.kernels[0].vertical_loops[0].ij_caches[0]
    assert cache.extent.i == (0, 1)
    assert cache.extent.j == (-3, 0)
