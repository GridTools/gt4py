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

from gtc.common import DataType, LoopOrder
from gtc.gtcpp.gtcpp_codegen import GTCppCodegen
from gtc.gtcpp.oir_to_gtcpp import OIRToGTCpp

from .oir_utils import (
    AssignStmtFactory,
    FieldDeclFactory,
    HorizontalExecutionFactory,
    StencilFactory,
    VariableKOffsetFactory,
    VerticalLoopFactory,
    VerticalLoopSectionFactory,
)
from .utils import match


def test_variable_offset_accessor():
    out_name = "out_field"
    in_name = "in_field"
    index_name = "index"
    oir_stencil = StencilFactory(
        params=[
            FieldDeclFactory(name=out_name, dtype=DataType.FLOAT32),
            FieldDeclFactory(name=in_name, dtype=DataType.FLOAT32),
            FieldDeclFactory(name=index_name, dtype=DataType.FLOAT32),
        ],
        vertical_loops__0=VerticalLoopFactory(
            loop_order=LoopOrder.FORWARD,
            sections=[
                VerticalLoopSectionFactory(
                    horizontal_executions=[
                        HorizontalExecutionFactory(
                            body=[
                                AssignStmtFactory(
                                    left__name=out_name,
                                    right__name=in_name,
                                    right__offset=VariableKOffsetFactory(k__name=index_name),
                                )
                            ]
                        )
                    ]
                )
            ],
        ),
    )

    gtcpp_program = OIRToGTCpp().visit(oir_stencil)
    code = GTCppCodegen.apply(gtcpp_program, gt_backend_t="cpu_ifirst")
    print(code)
    match(code, r"eval\(out_field\(\)\) = eval\(in_field\(0, 0, eval\(index\(\)\)\)\)")
