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

from dace.transformation.transformation import Transformation

from gtc import oir
from gtc.dace import dace_to_oir
from gtc.dace.oir_to_dace import OirSDFGBuilder
from gtc.dace.utils import iter_vertical_loop_section_sub_sdfgs


def optimize_horizontal_executions(
    stencil: oir.Stencil, transformation: Transformation
) -> oir.Stencil:
    if len(stencil.iter_tree().if_isinstance(oir.AxisPosition).to_list()) > 0:
        raise NotImplementedError("AxisPosition are not yet supported in DaCe transform")
    sdfg = OirSDFGBuilder().visit(stencil)
    api_fields = {param.name for param in stencil.params}
    for subgraph in iter_vertical_loop_section_sub_sdfgs(sdfg):
        subgraph.apply_transformations_repeated(
            transformation, validate=False, options=dict(api_fields=api_fields)
        )
    return dace_to_oir.convert(sdfg)
