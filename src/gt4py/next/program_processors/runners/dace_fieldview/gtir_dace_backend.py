# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import dace

from gt4py.next.common import Connectivity, Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors.runners.dace_fieldview import (
    gtir_to_sdfg as gtir_dace_translator,
)
from gt4py.next.type_system import type_specifications as ts


def build_sdfg_from_gtir(
    program: itir.Program,
    arg_types: list[ts.DataType],
    offset_provider: dict[str, Connectivity | Dimension],
) -> dace.SDFG:
    sdfg_genenerator = gtir_dace_translator.GTIRToSDFG(arg_types, offset_provider)
    sdfg = sdfg_genenerator.visit(program)
    assert isinstance(sdfg, dace.SDFG)

    sdfg.simplify()
    return sdfg
