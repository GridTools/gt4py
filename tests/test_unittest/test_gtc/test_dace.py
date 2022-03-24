# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from copy import deepcopy

import pytest

from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gt4py.definitions import BuildOptions
from gt4py.frontend.gtscript_frontend import GTScriptFrontend
from gtc.common import AxisBound, DataType
from gtc.dace.dace_to_oir import convert
from gtc.dace.oir_to_dace import OirSDFGBuilder
from gtc.dace.utils import is_sdfg_equal
from gtc.gtir_to_oir import GTIRToOIR
from gtc.oir import Interval, Literal
from gtc.passes.gtir_pipeline import GtirPipeline

from ...test_integration.stencil_definitions import EXTERNALS_REGISTRY as externals_registry
from ...test_integration.stencil_definitions import REGISTRY as stencil_registry
from .oir_utils import (
    AssignStmtFactory,
    HorizontalExecutionFactory,
    StencilFactory,
    VerticalLoopFactory,
    VerticalLoopSectionFactory,
)


def stencil_def_to_oir(stencil_def, externals):

    build_options = BuildOptions(
        name=stencil_def.__name__, module=__name__, rebuild=True, backend_opts={}, build_info=None
    )
    definition_ir = GTScriptFrontend.generate(
        stencil_def, externals=externals, options=build_options
    )
    gtir = GtirPipeline(DefIRToGTIR.apply(definition_ir)).full()
    return GTIRToOIR().visit(gtir)


@pytest.mark.parametrize("stencil_name", stencil_registry.keys())
def test_stencils_roundtrip(stencil_name):

    stencil_def = stencil_registry[stencil_name]
    externals = externals_registry[stencil_name]
    oir = stencil_def_to_oir(stencil_def, externals)
    sdfg = OirSDFGBuilder().visit(oir)

    sdfg_pre = deepcopy(sdfg)

    oir = convert(sdfg, oir.loc)
    sdfg_post = OirSDFGBuilder().visit(oir)
    assert is_sdfg_equal(sdfg_pre, sdfg_post)


def test_same_node_read_write_not_overlap():

    oir = StencilFactory(
        vertical_loops=[
            VerticalLoopFactory(
                sections__0=VerticalLoopSectionFactory(
                    interval=Interval(start=AxisBound.start(), end=AxisBound.from_start(1)),
                    horizontal_executions__0__body__0=AssignStmtFactory(
                        left__name="field", right__name="other"
                    ),
                )
            ),
            VerticalLoopFactory(
                sections__0=VerticalLoopSectionFactory(
                    interval=Interval(start=AxisBound.from_start(1), end=AxisBound.from_start(2)),
                    horizontal_executions__0__body__0=AssignStmtFactory(
                        left__name="field", right__name="field", right__offset__k=-1
                    ),
                )
            ),
        ]
    )
    sdfg = OirSDFGBuilder().visit(oir)
    convert(sdfg, oir.loc)


def test_two_vertical_loops_no_read():
    oir_pre = StencilFactory(
        vertical_loops=[
            VerticalLoopFactory(
                sections__0=VerticalLoopSectionFactory(
                    horizontal_executions=[
                        HorizontalExecutionFactory(
                            body__0=AssignStmtFactory(
                                left__name="field",
                                right=Literal(value="42.0", dtype=DataType.FLOAT32),
                            )
                        )
                    ],
                    interval__end=AxisBound.from_start(3),
                ),
            ),
            VerticalLoopFactory(
                sections__0=VerticalLoopSectionFactory(
                    horizontal_executions=[
                        HorizontalExecutionFactory(
                            body__0=AssignStmtFactory(
                                left__name="field",
                                right=Literal(value="43.0", dtype=DataType.FLOAT32),
                            )
                        )
                    ],
                    interval__start=AxisBound.from_start(3),
                ),
            ),
        ]
    )
    sdfg = OirSDFGBuilder().visit(oir_pre)
    convert(sdfg, oir_pre.loc)
