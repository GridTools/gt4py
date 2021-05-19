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

from gt4py.gtscript import Field, stencil
from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gt4py.definitions import BuildOptions
from gt4py.frontend.gtscript_frontend import GTScriptFrontend
from gtc.gtir_to_oir import GTIRToOIR
from gtc.oir import FieldAccess, BinaryOp
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.passes.oir_optimizations.inlining import MaskInlining


def test_mask_inlining():
    def stencil_def(
        extm: Field[float],
        a4_1: Field[float],
        a4_2: Field[float],
    ):
        with computation(PARALLEL), interval(...):
            if extm != 0.0 and (extm[0, 0, -1] != 0.0 or extm[0, 0, 1] != 0.0):
                a4_2 = a4_1
            else:
                a4_2 = 6.0 * a4_1 - 3.0 * a4_2

    build_options = BuildOptions(name=stencil_def.__name__, module=__name__)
    definition_ir = GTScriptFrontend.generate(
        stencil_def, options=build_options, externals={}
    )
    gtir = GtirPipeline(DefIRToGTIR.apply(definition_ir)).full()

    pre_oir = GTIRToOIR().visit(gtir)
    pre_section = pre_oir.vertical_loops[0].sections[0]
    assert pre_section.horizontal_executions[0].body
    pre_mask = pre_section.horizontal_executions[1].body[0].mask
    assert isinstance(pre_mask, FieldAccess)

    post_oir = MaskInlining().visit(pre_oir)
    post_section = post_oir.vertical_loops[0].sections[0]
    assert not post_section.horizontal_executions[0].body
    post_mask = post_section.horizontal_executions[1].body[0].mask
    assert isinstance(post_mask, BinaryOp)
