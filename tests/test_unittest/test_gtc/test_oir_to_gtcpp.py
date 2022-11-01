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

from gtc.common import HorizontalInterval, HorizontalMask, LevelMarker, LoopOrder
from gtc.gtcpp import gtcpp
from gtc.gtcpp.gtcpp_codegen import GTCppCodegen
from gtc.gtcpp.oir_to_gtcpp import OIRToGTCpp

from .oir_utils import (
    AssignStmtFactory,
    HorizontalRestrictionFactory,
    LiteralFactory,
    StencilFactory,
    VariableKOffsetFactory,
    VerticalLoopFactory,
)
from .utils import match


def test_horizontal_mask():
    out_name = "out_field"
    in_name = "in_field"

    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(left__name=out_name, right__name=in_name),
            HorizontalRestrictionFactory(
                mask=HorizontalMask(
                    i=HorizontalInterval.at_endpt(LevelMarker.START, 0),
                    j=HorizontalInterval.full(),
                ),
                body=[AssignStmtFactory(left__name=out_name, right=LiteralFactory())],
            ),
            HorizontalRestrictionFactory(
                mask=HorizontalMask(
                    i=HorizontalInterval.at_endpt(LevelMarker.END, 0),
                    j=HorizontalInterval.full(),
                ),
                body=[AssignStmtFactory(left__name=out_name, right=LiteralFactory())],
            ),
            HorizontalRestrictionFactory(
                mask=HorizontalMask(
                    i=HorizontalInterval.full(),
                    j=HorizontalInterval.at_endpt(LevelMarker.START, 0),
                ),
                body=[AssignStmtFactory(left__name=out_name, right=LiteralFactory())],
            ),
            HorizontalRestrictionFactory(
                mask=HorizontalMask(
                    i=HorizontalInterval.full(),
                    j=HorizontalInterval.at_endpt(LevelMarker.END, 0),
                ),
                body=[AssignStmtFactory(left__name=out_name, right=LiteralFactory())],
            ),
        ]
    )
    gtcpp_program = OIRToGTCpp().visit(testee)
    positional_axis_names = (
        gtcpp_program.walk_values().if_isinstance(gtcpp.Positional).getattr("axis_name").to_set()
    )
    axis_lengths = (
        gtcpp_program.walk_values().if_isinstance(gtcpp.AxisLength).getattr("axis").to_set()
    )

    assert positional_axis_names == {"i", "j"}
    assert axis_lengths == {0, 1}


def test_variable_offset_accessor():
    out_name = "out_field"
    in_name = "in_field"
    index_name = "index"
    oir_stencil = StencilFactory(
        vertical_loops__0=VerticalLoopFactory(
            loop_order=LoopOrder.FORWARD,
            sections__0__horizontal_executions__0__body=[
                AssignStmtFactory(
                    left__name=out_name,
                    right__name=in_name,
                    right__offset=VariableKOffsetFactory(k__name=index_name),
                )
            ],
        ),
    )

    gtcpp_program = OIRToGTCpp().visit(oir_stencil)
    code = GTCppCodegen.apply(gtcpp_program, gt_backend_t="cpu_ifirst")
    print(code)
    match(code, rf"eval\({out_name}\(\)\) = eval\({in_name}\(0, 0, eval\({index_name}\(\)\)\)\)")
