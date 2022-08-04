# GT4Py Project - GridTools Framework
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


import numpy
import pytest

from functional.common import Dimension

#  from functional.fencil_processors import source_modules
from functional.fencil_processors.codegens.gtfn import gtfn_module
from functional.fencil_processors.source_modules import cpp_gen as cpp
from functional.iterator import ir
from functional.iterator.embedded import np_as_located_field


@pytest.fixture
def fencil_example():
    domain = ir.FunCall(
        fun=ir.SymRef(id="cartesian_domain"),
        args=[
            ir.FunCall(
                fun=ir.SymRef(id="named_range"),
                args=[
                    ir.AxisLiteral(value="X"),
                    ir.Literal(value="0", type="int"),
                    ir.Literal(value="10", type="int"),
                ],
            )
        ],
    )
    itir = ir.FencilDefinition(
        id="example",
        params=[ir.Sym(id="buf"), ir.Sym(id="sc")],
        function_definitions=[
            ir.FunctionDefinition(
                id="stencil",
                params=[ir.Sym(id="buf"), ir.Sym(id="sc")],
                expr=ir.Literal(value="1", type="float"),
            )
        ],
        closures=[
            ir.StencilClosure(
                domain=domain,
                stencil=ir.SymRef(id="stencil"),
                output=ir.SymRef(id="buf"),
                inputs=[ir.SymRef(id="buf"), ir.SymRef(id="sc")],
            )
        ],
    )
    params = [
        np_as_located_field(Dimension("I"))(numpy.empty((1,), dtype=numpy.float32)),
        numpy.float32(3.14),
        #  source_modules.BufferParameter("buf", ["I"], numpy.dtype(numpy.float32)),
        #  source_modules.ScalarParameter("sc", numpy.dtype(numpy.float32)),
    ]
    return itir, params


def test_codegen(fencil_example):
    itir, parameters = fencil_example
    module = gtfn_module.create_source_module(itir, *parameters, offset_provider={})
    assert module.entry_point.name == itir.id
    assert any(d.name == "gridtools" for d in module.library_deps)
    assert module.language == cpp.LANGUAGE_ID
