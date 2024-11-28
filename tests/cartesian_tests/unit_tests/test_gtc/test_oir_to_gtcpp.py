# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.cartesian.gtc.common import HorizontalInterval, HorizontalMask, LevelMarker, LoopOrder
from gt4py.cartesian.gtc.gtcpp import gtcpp
from gt4py.cartesian.gtc.gtcpp.gtcpp_codegen import GTCppCodegen
from gt4py.cartesian.gtc.gtcpp.oir_to_gtcpp import OIRToGTCpp

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
                    i=HorizontalInterval.at_endpt(LevelMarker.START, 0), j=HorizontalInterval.full()
                ),
                body=[AssignStmtFactory(left__name=out_name, right=LiteralFactory())],
            ),
            HorizontalRestrictionFactory(
                mask=HorizontalMask(
                    i=HorizontalInterval.at_endpt(LevelMarker.END, 0), j=HorizontalInterval.full()
                ),
                body=[AssignStmtFactory(left__name=out_name, right=LiteralFactory())],
            ),
            HorizontalRestrictionFactory(
                mask=HorizontalMask(
                    i=HorizontalInterval.full(), j=HorizontalInterval.at_endpt(LevelMarker.START, 0)
                ),
                body=[AssignStmtFactory(left__name=out_name, right=LiteralFactory())],
            ),
            HorizontalRestrictionFactory(
                mask=HorizontalMask(
                    i=HorizontalInterval.full(), j=HorizontalInterval.at_endpt(LevelMarker.END, 0)
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
        )
    )

    gtcpp_program = OIRToGTCpp().visit(oir_stencil)
    code = GTCppCodegen.apply(gtcpp_program, gt_backend_t="cpu_ifirst")
    print(code)
    match(code, rf"eval\({out_name}\(\)\) = eval\({in_name}\(0, 0, eval\({index_name}\(\)\)\)\)")
