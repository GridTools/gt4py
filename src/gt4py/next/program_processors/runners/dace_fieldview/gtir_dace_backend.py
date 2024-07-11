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

from __future__ import annotations

import dace

from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.program_processors.runners.dace_fieldview import (
    gtir_to_sdfg as gtir_dace_translator,
)


def build_sdfg_from_gtir(
    program: gtir.Program,
    offset_provider: dict[str, gtx_common.Connectivity | gtx_common.Dimension],
) -> dace.SDFG:
    """
    Receives a GTIR program and lowers it to a DaCe SDFG.

    The lowering to SDFG requires that the program node is type-annotated, therefore this function
    runs type ineference as first step.
    As a final step, it runs the `simplify` pass to ensure that the SDFG is in the DaCe canonical form.

    Arguments:
        program: The GTIR program node to be lowered to SDFG
        offset_provider: The definitions of offset providers used by the program node

    Returns:
        An SDFG in the DaCe canonical form (simplified)
    """
    sdfg_genenerator = gtir_dace_translator.GTIRToSDFG(offset_provider)
    # TODO: run type inference on the `program` node before passing it to `GTIRToSDFG`
    sdfg = sdfg_genenerator.visit(program)
    assert isinstance(sdfg, dace.SDFG)

    sdfg.simplify()
    return sdfg
