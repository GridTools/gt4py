#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# GT4Py Project - GridTools Framework
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
import numpy as np

from functional.ffront.foast_to_itir import FieldOperatorLowering
from functional.ffront.func_to_foast import FieldOperatorParser
from functional.iterator import ir as itir
from functional.iterator.backends import roundtrip
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import CartesianAxis


def make_domain(dim_name: str, lower: int, upper: int) -> itir.FunCall:
    return itir.FunCall(
        fun=itir.SymRef(id="domain"),
        args=[
            itir.FunCall(
                fun=itir.SymRef(id="named_range"),
                args=[
                    itir.AxisLiteral(value=dim_name),
                    itir.IntLiteral(value=lower),
                    itir.IntLiteral(value=upper),
                ],
            )
        ],
    )


def closure_from_fop(
    node: itir.FunctionDefinition, out_names: list[str], domain: itir.FunCall
) -> itir.StencilClosure:
    return itir.StencilClosure(
        stencil=itir.SymRef(id=node.id),
        inputs=[itir.SymRef(id=sym.id) for sym in node.params],
        outputs=[itir.SymRef(id=name) for name in out_names],
        domain=domain,
    )


def fencil_from_fop(
    node: itir.FunctionDefinition, out_names: list[str], domain: itir.FunCall
) -> itir.FencilDefinition:
    closure = closure_from_fop(node, out_names=out_names, domain=domain)
    return itir.FencilDefinition(
        id=node.id + "_fencil",
        params=[itir.Sym(id=inp.id) for inp in closure.inputs]
        + [itir.Sym(id=out.id) for out in closure.outputs],
        closures=[closure],
    )


def program_from_fop(
    node: itir.FunctionDefinition, out_names: list[str], dim: CartesianAxis, size: int
) -> itir.Program:
    domain = make_domain(dim.value, 0, size)
    return itir.Program(
        function_definitions=[node],
        fencil_definitions=[fencil_from_fop(node, out_names=out_names, domain=domain)],
        setqs=[],
    )


def test_copy():
    size = 10
    IDim = CartesianAxis("IDim")
    a = np_as_located_field(IDim)(np.ones((size)))
    b = np_as_located_field(IDim)(np.zeros((size)))

    def copy(inp):
        return inp

    copy_foast = FieldOperatorParser.apply(copy)
    copy_fundef = FieldOperatorLowering.apply(copy_foast)
    copy_program = program_from_fop(node=copy_fundef, out_names=["out"], dim=IDim, size=size)

    roundtrip.executor(copy_program, a, b, offset_provider={})

    assert np.allclose(a, b)
