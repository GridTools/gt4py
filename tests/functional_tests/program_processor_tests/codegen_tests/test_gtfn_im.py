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


import numpy as np
import pytest

from functional.common import Dimension
from functional.iterator import embedded, ir as itir
from functional.otf import languages, stages
from functional.program_processors.codegens.gtfn import gtfn_ir, gtfn_ir_to_gtfn_im_ir


@pytest.fixture
def fun_pass():
    return gtfn_ir.FunCall(
        fun=gtfn_ir.Lambda(
            params=[gtfn_ir.Sym(id="cs")],
            expr=gtfn_ir.FunCall(fun=gtfn_ir.SymRef(id="cs"), args=[gtfn_ir.SymRef(id="w")]),
        ),
        args=[
            gtfn_ir.Lambda(
                params=[gtfn_ir.Sym(id="x")],
                expr=gtfn_ir.BinaryExpr(
                    op="+",
                    lhs=gtfn_ir.SymRef(id="x"),
                    rhs=gtfn_ir.Literal(value="1.0", type="float64"),
                ),
            )
        ],
    )


@pytest.fixture
def invalid_reduction():
    return gtfn_ir.FunCall(
        fun=gtfn_ir.Lambda(
            params=[gtfn_ir.Sym(id="_cs_1")],
            expr=gtfn_ir.FunCall(
                fun=gtfn_ir.FunCall(
                    fun=gtfn_ir.SymRef(id="reduce"),
                    args=[
                        gtfn_ir.Lambda(
                            params=[
                                gtfn_ir.Sym(id="acc"),
                                gtfn_ir.Sym(id="geofac_grg_x__0"),
                                gtfn_ir.Sym(id="w__1"),
                            ],
                            expr=gtfn_ir.BinaryExpr(
                                op="+",
                                lhs=gtfn_ir.SymRef(id="acc"),
                                rhs=gtfn_ir.BinaryExpr(
                                    op="*",
                                    lhs=gtfn_ir.SymRef(id="geofac_grg_x__0"),
                                    rhs=gtfn_ir.SymRef(id="w__1"),
                                ),
                            ),
                        ),
                        gtfn_ir.Literal(value="0", type="float64"),
                    ],
                ),
                args=[
                    gtfn_ir.SymRef(id="geofac_grg_x"),
                    gtfn_ir.SymRef(id="_cs_1"),
                ],
            ),
        ),
        args=[
            gtfn_ir.FunCall(
                fun=gtfn_ir.SymRef(id="shift"),
                args=[
                    gtfn_ir.SymRef(id="w"),
                    gtfn_ir.OffsetLiteral(value="C2E2CO"),
                ],
            )
        ],
    )


@pytest.fixture
def unrolled_reduction():
    gtfn_ir.FunCall(
        fun=gtfn_ir.Lambda(
            params=[gtfn_ir.Sym(id="_step_3")],
            expr=gtfn_ir.FunCall(
                fun=gtfn_ir.SymRef(id="_step_3"),
                args=[
                    gtfn_ir.FunCall(
                        fun=gtfn_ir.SymRef(id="_step_3"),
                        args=[
                            gtfn_ir.FunCall(
                                fun=gtfn_ir.SymRef(id="_step_3"),
                                args=[
                                    gtfn_ir.Literal(value="0", type="float64"),
                                    gtfn_ir.OffsetLiteral(value=0),
                                ],
                            ),
                            gtfn_ir.OffsetLiteral(value=1),
                        ],
                    ),
                    gtfn_ir.OffsetLiteral(value=2),
                ],
            ),
        ),
        args=[
            gtfn_ir.Lambda(
                params=[gtfn_ir.Sym(id="_acc_1"), gtfn_ir.Sym(id="_i_2")],
                expr=gtfn_ir.BinaryExpr(
                    op="+",
                    lhs=gtfn_ir.SymRef(id="_acc_1"),
                    rhs=gtfn_ir.BinaryExpr(
                        op="*",
                        lhs=gtfn_ir.FunCall(
                            fun=gtfn_ir.SymRef(id="deref"),
                            args=[
                                gtfn_ir.FunCall(
                                    fun=gtfn_ir.SymRef(id="shift"),
                                    args=[
                                        gtfn_ir.SymRef(id="vn"),
                                        gtfn_ir.OffsetLiteral(value="C2E"),
                                        gtfn_ir.SymRef(id="_i_2"),
                                    ],
                                )
                            ],
                        ),
                        rhs=gtfn_ir.FunCall(
                            fun=gtfn_ir.SymRef(id="tuple_get"),
                            args=[
                                gtfn_ir.SymRef(id="_i_2"),
                                gtfn_ir.FunCall(
                                    fun=gtfn_ir.SymRef(id="deref"),
                                    args=[gtfn_ir.SymRef(id="geofac_div")],
                                ),
                            ],
                        ),
                    ),
                ),
            )
        ],
    )


def test_gtfn_im_ir_precond_1(fun_pass):
    is_compat = gtfn_ir_to_gtfn_im_ir.IsImpCompatible()
    is_compat.visit(fun_pass)
    assert not is_compat.compatible
    assert is_compat.incompatible_node.fun == gtfn_ir.SymRef(id="cs")


def test_gtfn_im_ir_precond_2(invalid_reduction):
    is_compat = gtfn_ir_to_gtfn_im_ir.IsImpCompatible()
    is_compat.visit(invalid_reduction, offset_provider={})
    assert not is_compat.compatible
    assert is_compat.incompatible_node.fun.fun == gtfn_ir.SymRef(id="reduce")
