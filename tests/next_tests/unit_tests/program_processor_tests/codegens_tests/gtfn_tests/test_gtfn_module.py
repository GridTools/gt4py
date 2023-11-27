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

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import languages, stages
from gt4py.next.program_processors.codegens.gtfn import gtfn_module


@pytest.fixture
def fencil_example():
    domain = itir.FunCall(
        fun=itir.SymRef(id="cartesian_domain"),
        args=[
            itir.FunCall(
                fun=itir.SymRef(id="named_range"),
                args=[
                    itir.AxisLiteral(value="X"),
                    itir.Literal(value="0", type=itir.INTEGER_INDEX_BUILTIN),
                    itir.Literal(value="10", type=itir.INTEGER_INDEX_BUILTIN),
                ],
            )
        ],
    )
    fencil = itir.FencilDefinition(
        id="example",
        params=[itir.Sym(id="buf"), itir.Sym(id="sc")],
        function_definitions=[
            itir.FunctionDefinition(
                id="stencil",
                params=[itir.Sym(id="buf"), itir.Sym(id="sc")],
                expr=itir.Literal(value="1", type="float64"),
            )
        ],
        closures=[
            itir.StencilClosure(
                domain=domain,
                stencil=itir.SymRef(id="stencil"),
                output=itir.SymRef(id="buf"),
                inputs=[itir.SymRef(id="buf"), itir.SymRef(id="sc")],
            )
        ],
    )
    IDim = gtx.Dimension("I")
    params = [
        gtx.as_field([IDim], np.empty((1,), dtype=np.float32)),
        np.float32(3.14),
    ]
    return fencil, params


def test_codegen(fencil_example):
    fencil, parameters = fencil_example
    module = gtfn_module.translate_program_cpu(
        stages.ProgramCall(fencil, parameters, {"offset_provider": {}})
    )
    assert module.entry_point.name == fencil.id
    assert any(d.name == "gridtools_cpu" for d in module.library_deps)
    assert module.language is languages.Cpp
