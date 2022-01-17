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

import warnings
from typing import List, Type, Union

from gtc import oir
from gtc.dace import dace_to_oir
from gtc.dace.oir_to_dace import OirSDFGBuilder
from gtc.dace.utils import iter_vertical_loop_section_sub_sdfgs


def has_variable_access(stencil: oir.Stencil) -> bool:
    return len(stencil.iter_tree().if_isinstance(oir.VariableKOffset).to_list()) > 0


def optimize_horizontal_executions(
    stencil: oir.Stencil, transformation: Union[Type, List[Type]]
) -> oir.Stencil:
    if has_variable_access(stencil):
        warnings.warn(
            "oir dace optimize_horizontal_executions is not yet supported with variable vertical accesses. See https://github.com/GridTools/gt4py/issues/517"
        )
        return stencil
    sdfg = OirSDFGBuilder().visit(stencil)
    api_fields = {param.name for param in stencil.params}
    for subgraph in iter_vertical_loop_section_sub_sdfgs(sdfg):
        subgraph.apply_transformations_repeated(
            transformation, validate=False, options=dict(api_fields=api_fields)
        )
    return dace_to_oir.convert(sdfg, stencil.loc)
