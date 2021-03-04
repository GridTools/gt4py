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
from pydantic.error_wrappers import ValidationError

from gtc.common import AxisBound, DataType
from gtc.gtir_to_oir import oir_field_boundary_computation, oir_iteration_space_computation

from .oir_utils import (
    AssignStmtFactory,
    FieldAccessFactory,
    HorizontalExecutionFactory,
    IntervalFactory,
    StencilFactory,
    VerticalLoopFactory,
    VerticalLoopSectionFactory,
)


def test_no_horizontal_offset_allowed():
    with pytest.raises(ValidationError, match=r"must not have .*horizontal offset"):
        AssignStmtFactory(left__offset__i=1)


def test_mask_must_be_bool():
    with pytest.raises(ValidationError, match=r".*must be.* bool.*"):
        HorizontalExecutionFactory(mask=FieldAccessFactory(dtype=DataType.INT32))


class TestIterationSpaceAndHalo:
    def test_horizontal_offset_same_section(self):
        oir = StencilFactory(
            vertical_loops__0__sections__0__horizontal_executions=[
                HorizontalExecutionFactory(
                    body__0=AssignStmtFactory(
                        left__name="tmp", right__name="in_field", right__offset__i=1
                    )
                ),
                HorizontalExecutionFactory(
                    body__0=AssignStmtFactory(
                        left__name="out_field", right__name="tmp", right__offset__i=1
                    )
                ),
            ],
        )
        oir = oir_iteration_space_computation(oir)
        he1 = oir.vertical_loops[0].sections[0].horizontal_executions[0]
        he2 = oir.vertical_loops[0].sections[0].horizontal_executions[1]

        assert he1.iteration_space.i_interval.start.offset == 0
        assert he1.iteration_space.i_interval.end.offset == 1
        assert he1.iteration_space.j_interval.start.offset == 0
        assert he1.iteration_space.j_interval.end.offset == 0

        assert he2.iteration_space.i_interval.start.offset == 0
        assert he2.iteration_space.i_interval.end.offset == 0
        assert he2.iteration_space.j_interval.start.offset == 0
        assert he2.iteration_space.j_interval.end.offset == 0
        field_boundaries = oir_field_boundary_computation(oir)
        assert field_boundaries["in_field"].i_interval.start.offset == 0
        assert field_boundaries["in_field"].i_interval.end.offset == 2
        assert field_boundaries["in_field"].j_interval.start.offset == 0
        assert field_boundaries["in_field"].j_interval.end.offset == 0

        assert field_boundaries["tmp"].i_interval.start.offset == 0
        assert field_boundaries["tmp"].i_interval.end.offset == 1
        assert field_boundaries["tmp"].j_interval.start.offset == 0
        assert field_boundaries["tmp"].j_interval.end.offset == 0

        assert field_boundaries["out_field"].i_interval.start.offset == 0
        assert field_boundaries["out_field"].i_interval.end.offset == 0
        assert field_boundaries["out_field"].j_interval.start.offset == 0
        assert field_boundaries["out_field"].j_interval.end.offset == 0

    def test_horizontal_offset_different_intervals(self):
        oir = StencilFactory(
            vertical_loops__0__sections__0=VerticalLoopSectionFactory(
                interval=IntervalFactory(end=AxisBound.from_end(-4)),
                horizontal_executions__0__body__0=AssignStmtFactory(
                    left__name="tmp", right__name="in_field", right__offset__i=1
                ),
            ),
            vertical_loops__0__sections__1=VerticalLoopSectionFactory(
                interval=IntervalFactory(start=AxisBound.from_end(-4)),
                horizontal_executions__0__body__0=AssignStmtFactory(
                    left__name="out_field", right__name="tmp", right__offset__i=1
                ),
            ),
        )

        oir = oir_iteration_space_computation(oir)
        he1 = oir.vertical_loops[0].sections[0].horizontal_executions[0]
        he2 = oir.vertical_loops[0].sections[1].horizontal_executions[0]

        assert he1.iteration_space.i_interval.start.offset == 0
        assert he1.iteration_space.i_interval.end.offset == 0
        assert he1.iteration_space.j_interval.start.offset == 0
        assert he1.iteration_space.j_interval.end.offset == 0

        assert he2.iteration_space.i_interval.start.offset == 0
        assert he2.iteration_space.i_interval.end.offset == 0
        assert he2.iteration_space.j_interval.start.offset == 0
        assert he2.iteration_space.j_interval.end.offset == 0

        field_boundaries = oir_field_boundary_computation(oir)
        assert field_boundaries["in_field"].i_interval.start.offset == 0
        assert field_boundaries["in_field"].i_interval.end.offset == 1
        assert field_boundaries["in_field"].j_interval.start.offset == 0
        assert field_boundaries["in_field"].j_interval.end.offset == 0

        assert field_boundaries["tmp"].i_interval.start.offset == 0
        assert field_boundaries["tmp"].i_interval.end.offset == 1
        assert field_boundaries["tmp"].j_interval.start.offset == 0
        assert field_boundaries["tmp"].j_interval.end.offset == 0

        assert field_boundaries["out_field"].i_interval.start.offset == 0
        assert field_boundaries["out_field"].i_interval.end.offset == 0
        assert field_boundaries["out_field"].j_interval.start.offset == 0
        assert field_boundaries["out_field"].j_interval.end.offset == 0

    def test_horizontal_offset_different_vertical_loop(self):
        oir = StencilFactory(
            vertical_loops=[
                VerticalLoopFactory(
                    sections__0__horizontal_executions__0__body__0=AssignStmtFactory(
                        left__name="tmp", right__name="in_field", right__offset__i=1
                    )
                ),
                VerticalLoopFactory(
                    sections__0__horizontal_executions__0__body__0=AssignStmtFactory(
                        left__name="out_field", right__name="tmp", right__offset__i=1
                    )
                ),
            ]
        )

        oir = oir_iteration_space_computation(oir)
        he1 = oir.vertical_loops[0].sections[0].horizontal_executions[0]
        he2 = oir.vertical_loops[1].sections[0].horizontal_executions[0]

        assert he1.iteration_space.i_interval.start.offset == 0
        assert he1.iteration_space.i_interval.end.offset == 1
        assert he1.iteration_space.j_interval.start.offset == 0
        assert he1.iteration_space.j_interval.end.offset == 0

        assert he2.iteration_space.i_interval.start.offset == 0
        assert he2.iteration_space.i_interval.end.offset == 0
        assert he2.iteration_space.j_interval.start.offset == 0
        assert he2.iteration_space.j_interval.end.offset == 0

        field_boundaries = oir_field_boundary_computation(oir)
        assert field_boundaries["in_field"].i_interval.start.offset == 0
        assert field_boundaries["in_field"].i_interval.end.offset == 2
        assert field_boundaries["in_field"].j_interval.start.offset == 0
        assert field_boundaries["in_field"].j_interval.end.offset == 0

        assert field_boundaries["tmp"].i_interval.start.offset == 0
        assert field_boundaries["tmp"].i_interval.end.offset == 1
        assert field_boundaries["tmp"].j_interval.start.offset == 0
        assert field_boundaries["tmp"].j_interval.end.offset == 0

        assert field_boundaries["out_field"].i_interval.start.offset == 0
        assert field_boundaries["out_field"].i_interval.end.offset == 0
        assert field_boundaries["out_field"].j_interval.start.offset == 0
        assert field_boundaries["out_field"].j_interval.end.offset == 0
