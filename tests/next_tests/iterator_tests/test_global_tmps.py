# GT4Py - GridTools Framework
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

from gt4py.eve.utils import UIDs
from gt4py.next.iterator import ir
from gt4py.next.iterator.runtime import CartesianAxis
from gt4py.next.iterator.transforms.global_tmps import (
    AUTO_DOMAIN,
    FencilWithTemporaries,
    Temporary,
    collect_tmps_info,
    split_closures,
    update_cartesian_domains,
)


def test_split_closures():
    UIDs.reset_sequence()
    testee = ir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[ir.Sym(id="d"), ir.Sym(id="inp"), ir.Sym(id="out")],
        closures=[
            ir.StencilClosure(
                domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
                stencil=ir.Lambda(
                    params=[ir.Sym(id="baz_inp")],
                    expr=ir.FunCall(
                        fun=ir.SymRef(id="deref"),
                        args=[
                            ir.FunCall(
                                fun=ir.FunCall(
                                    fun=ir.SymRef(id="lift"),
                                    args=[
                                        ir.Lambda(
                                            params=[ir.Sym(id="bar_inp")],
                                            expr=ir.FunCall(
                                                fun=ir.SymRef(id="deref"),
                                                args=[
                                                    ir.FunCall(
                                                        fun=ir.FunCall(
                                                            fun=ir.SymRef(id="lift"),
                                                            args=[
                                                                ir.Lambda(
                                                                    params=[ir.Sym(id="foo_inp")],
                                                                    expr=ir.FunCall(
                                                                        fun=ir.SymRef(id="deref"),
                                                                        args=[
                                                                            ir.SymRef(id="foo_inp")
                                                                        ],
                                                                    ),
                                                                )
                                                            ],
                                                        ),
                                                        args=[ir.SymRef(id="bar_inp")],
                                                    )
                                                ],
                                            ),
                                        )
                                    ],
                                ),
                                args=[ir.SymRef(id="baz_inp")],
                            )
                        ],
                    ),
                ),
                output=ir.SymRef(id="out"),
                inputs=[ir.SymRef(id="inp")],
            )
        ],
    )

    expected = ir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[
            ir.Sym(id="d"),
            ir.Sym(id="inp"),
            ir.Sym(id="out"),
            ir.Sym(id="_gtmp_0"),
            ir.Sym(id="_gtmp_1"),
            ir.Sym(id="_gtmp_auto_domain"),
        ],
        closures=[
            ir.StencilClosure(
                domain=AUTO_DOMAIN,
                stencil=ir.Lambda(
                    params=[ir.Sym(id="foo_inp")],
                    expr=ir.FunCall(
                        fun=ir.SymRef(id="deref"),
                        args=[ir.SymRef(id="foo_inp")],
                    ),
                ),
                output=ir.SymRef(id="_gtmp_1"),
                inputs=[ir.SymRef(id="inp")],
            ),
            ir.StencilClosure(
                domain=AUTO_DOMAIN,
                stencil=ir.Lambda(
                    params=[
                        ir.Sym(id="bar_inp"),
                        ir.Sym(id="_lift_1"),
                    ],
                    expr=ir.FunCall(
                        fun=ir.SymRef(id="deref"),
                        args=[
                            ir.SymRef(id="_lift_1"),
                        ],
                    ),
                ),
                output=ir.SymRef(id="_gtmp_0"),
                inputs=[ir.SymRef(id="inp"), ir.SymRef(id="_gtmp_1")],
            ),
            ir.StencilClosure(
                domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
                stencil=ir.Lambda(
                    params=[ir.Sym(id="baz_inp"), ir.Sym(id="_lift_2")],
                    expr=ir.FunCall(
                        fun=ir.SymRef(id="deref"),
                        args=[ir.SymRef(id="_lift_2")],
                    ),
                ),
                output=ir.SymRef(id="out"),
                inputs=[ir.SymRef(id="inp"), ir.SymRef(id="_gtmp_0")],
            ),
        ],
    )
    actual = split_closures(testee)
    assert actual.tmps == [Temporary(id="_gtmp_0"), Temporary(id="_gtmp_1")]
    assert actual.fencil == expected


def test_update_cartesian_domains():
    testee = FencilWithTemporaries(
        fencil=ir.FencilDefinition(
            id="f",
            function_definitions=[],
            params=[
                ir.Sym(id="i"),
                ir.Sym(id="j"),
                ir.Sym(id="k"),
                ir.Sym(id="inp"),
                ir.Sym(id="out"),
                ir.Sym(id="_gtmp_0"),
                ir.Sym(id="_gtmp_1"),
                ir.Sym(id="_gtmp_auto_domain"),
            ],
            closures=[
                ir.StencilClosure(
                    domain=AUTO_DOMAIN,
                    stencil=ir.Lambda(
                        params=[ir.Sym(id="foo_inp")],
                        expr=ir.FunCall(
                            fun=ir.SymRef(id="deref"),
                            args=[ir.SymRef(id="foo_inp")],
                        ),
                    ),
                    output=ir.SymRef(id="_gtmp_1"),
                    inputs=[ir.SymRef(id="inp")],
                ),
                ir.StencilClosure(
                    domain=AUTO_DOMAIN,
                    stencil=ir.SymRef(id="deref"),
                    output=ir.SymRef(id="_gtmp_0"),
                    inputs=[ir.SymRef(id="_gtmp_1")],
                ),
                ir.StencilClosure(
                    domain=ir.FunCall(
                        fun=ir.SymRef(id="cartesian_domain"),
                        args=[
                            ir.FunCall(
                                fun=ir.SymRef(id="named_range"),
                                args=[
                                    ir.AxisLiteral(value=a),
                                    ir.Literal(value="0", type="int"),
                                    ir.SymRef(id=s),
                                ],
                            )
                            for a, s in (("IDim", "i"), ("JDim", "j"), ("KDim", "k"))
                        ],
                    ),
                    stencil=ir.Lambda(
                        params=[ir.Sym(id="baz_inp"), ir.Sym(id="_lift_2")],
                        expr=ir.FunCall(
                            fun=ir.SymRef(id="deref"),
                            args=[
                                ir.FunCall(
                                    fun=ir.FunCall(
                                        fun=ir.SymRef(id="shift"),
                                        args=[
                                            ir.OffsetLiteral(value="I"),
                                            ir.OffsetLiteral(value=1),
                                        ],
                                    ),
                                    args=[ir.SymRef(id="_lift_2")],
                                )
                            ],
                        ),
                    ),
                    output=ir.SymRef(id="out"),
                    inputs=[ir.SymRef(id="inp"), ir.SymRef(id="_gtmp_0")],
                ),
            ],
        ),
        params=[
            ir.Sym(id="i"),
            ir.Sym(id="j"),
            ir.Sym(id="k"),
            ir.Sym(id="inp"),
            ir.Sym(id="out"),
        ],
        tmps=[
            Temporary(id="_gtmp_0"),
            Temporary(id="_gtmp_1"),
        ],
    )
    expected = FencilWithTemporaries(
        fencil=ir.FencilDefinition(
            id="f",
            function_definitions=[],
            params=[
                ir.Sym(id="i"),
                ir.Sym(id="j"),
                ir.Sym(id="k"),
                ir.Sym(id="inp"),
                ir.Sym(id="out"),
                ir.Sym(id="_gtmp_0"),
                ir.Sym(id="_gtmp_1"),
            ],
            closures=[
                ir.StencilClosure(
                    domain=ir.FunCall(
                        fun=ir.SymRef(id="cartesian_domain"),
                        args=[
                            ir.FunCall(
                                fun=ir.SymRef(id="named_range"),
                                args=[
                                    ir.AxisLiteral(value="IDim"),
                                    ir.Literal(value="0", type="int"),
                                    ir.FunCall(
                                        fun=ir.SymRef(id="plus"),
                                        args=[ir.SymRef(id="i"), ir.Literal(value="1", type="int")],
                                    ),
                                ],
                            )
                        ]
                        + [
                            ir.FunCall(
                                fun=ir.SymRef(id="named_range"),
                                args=[
                                    ir.AxisLiteral(value=a),
                                    ir.Literal(value="0", type="int"),
                                    ir.SymRef(id=s),
                                ],
                            )
                            for a, s in (("JDim", "j"), ("KDim", "k"))
                        ],
                    ),
                    stencil=ir.Lambda(
                        params=[ir.Sym(id="foo_inp")],
                        expr=ir.FunCall(
                            fun=ir.SymRef(id="deref"),
                            args=[ir.SymRef(id="foo_inp")],
                        ),
                    ),
                    output=ir.SymRef(id="_gtmp_1"),
                    inputs=[ir.SymRef(id="inp")],
                ),
                ir.StencilClosure(
                    domain=ir.FunCall(
                        fun=ir.SymRef(id="cartesian_domain"),
                        args=[
                            ir.FunCall(
                                fun=ir.SymRef(id="named_range"),
                                args=[
                                    ir.AxisLiteral(value="IDim"),
                                    ir.Literal(value="0", type="int"),
                                    ir.FunCall(
                                        fun=ir.SymRef(id="plus"),
                                        args=[ir.SymRef(id="i"), ir.Literal(value="1", type="int")],
                                    ),
                                ],
                            )
                        ]
                        + [
                            ir.FunCall(
                                fun=ir.SymRef(id="named_range"),
                                args=[
                                    ir.AxisLiteral(value=a),
                                    ir.Literal(value="0", type="int"),
                                    ir.SymRef(id=s),
                                ],
                            )
                            for a, s in (("JDim", "j"), ("KDim", "k"))
                        ],
                    ),
                    stencil=ir.SymRef(id="deref"),
                    output=ir.SymRef(id="_gtmp_0"),
                    inputs=[ir.SymRef(id="_gtmp_1")],
                ),
                ir.StencilClosure(
                    domain=ir.FunCall(
                        fun=ir.SymRef(id="cartesian_domain"),
                        args=[
                            ir.FunCall(
                                fun=ir.SymRef(id="named_range"),
                                args=[
                                    ir.AxisLiteral(value=a),
                                    ir.Literal(value="0", type="int"),
                                    ir.SymRef(id=s),
                                ],
                            )
                            for a, s in (("IDim", "i"), ("JDim", "j"), ("KDim", "k"))
                        ],
                    ),
                    stencil=ir.Lambda(
                        params=[ir.Sym(id="baz_inp"), ir.Sym(id="_lift_2")],
                        expr=ir.FunCall(
                            fun=ir.SymRef(id="deref"),
                            args=[
                                ir.FunCall(
                                    fun=ir.FunCall(
                                        fun=ir.SymRef(id="shift"),
                                        args=[
                                            ir.OffsetLiteral(value="I"),
                                            ir.OffsetLiteral(value=1),
                                        ],
                                    ),
                                    args=[ir.SymRef(id="_lift_2")],
                                )
                            ],
                        ),
                    ),
                    output=ir.SymRef(id="out"),
                    inputs=[ir.SymRef(id="inp"), ir.SymRef(id="_gtmp_0")],
                ),
            ],
        ),
        params=[
            ir.Sym(id="i"),
            ir.Sym(id="j"),
            ir.Sym(id="k"),
            ir.Sym(id="inp"),
            ir.Sym(id="out"),
        ],
        tmps=[
            Temporary(id="_gtmp_0"),
            Temporary(id="_gtmp_1"),
        ],
    )
    actual = update_cartesian_domains(testee, {"I": CartesianAxis("IDim")})
    assert actual == expected


def test_collect_tmps_info():
    tmp_domain = ir.FunCall(
        fun=ir.SymRef(id="cartesian_domain"),
        args=[
            ir.FunCall(
                fun=ir.SymRef(id="named_range"),
                args=[
                    ir.AxisLiteral(value="IDim"),
                    ir.Literal(value="0", type="int"),
                    ir.FunCall(
                        fun=ir.SymRef(id="plus"),
                        args=[ir.SymRef(id="i"), ir.Literal(value="1", type="int")],
                    ),
                ],
            )
        ]
        + [
            ir.FunCall(
                fun=ir.SymRef(id="named_range"),
                args=[
                    ir.AxisLiteral(value=a),
                    ir.Literal(value="0", type="int"),
                    ir.SymRef(id=s),
                ],
            )
            for a, s in (("JDim", "j"), ("KDim", "k"))
        ],
    )
    testee = FencilWithTemporaries(
        fencil=ir.FencilDefinition(
            id="f",
            function_definitions=[],
            params=[
                ir.Sym(id="i"),
                ir.Sym(id="j"),
                ir.Sym(id="k"),
                ir.Sym(id="inp"),
                ir.Sym(id="out"),
                ir.Sym(id="_gtmp_0"),
                ir.Sym(id="_gtmp_1"),
            ],
            closures=[
                ir.StencilClosure(
                    domain=tmp_domain,
                    stencil=ir.Lambda(
                        params=[ir.Sym(id="foo_inp")],
                        expr=ir.FunCall(
                            fun=ir.SymRef(id="deref"),
                            args=[ir.SymRef(id="foo_inp")],
                        ),
                    ),
                    output=ir.SymRef(id="_gtmp_1"),
                    inputs=[ir.SymRef(id="inp")],
                ),
                ir.StencilClosure(
                    domain=tmp_domain,
                    stencil=ir.SymRef(id="deref"),
                    output=ir.SymRef(id="_gtmp_0"),
                    inputs=[ir.SymRef(id="_gtmp_1")],
                ),
                ir.StencilClosure(
                    domain=ir.FunCall(
                        fun=ir.SymRef(id="cartesian_domain"),
                        args=[
                            ir.FunCall(
                                fun=ir.SymRef(id="named_range"),
                                args=[
                                    ir.AxisLiteral(value=a),
                                    ir.Literal(value="0", type="int"),
                                    ir.SymRef(id=s),
                                ],
                            )
                            for a, s in (("IDim", "i"), ("JDim", "j"), ("KDim", "k"))
                        ],
                    ),
                    stencil=ir.Lambda(
                        params=[ir.Sym(id="baz_inp"), ir.Sym(id="_lift_2")],
                        expr=ir.FunCall(
                            fun=ir.SymRef(id="deref"),
                            args=[
                                ir.FunCall(
                                    fun=ir.FunCall(
                                        fun=ir.SymRef(id="shift"),
                                        args=[
                                            ir.OffsetLiteral(value="I"),
                                            ir.OffsetLiteral(value=1),
                                        ],
                                    ),
                                    args=[ir.SymRef(id="_lift_2")],
                                )
                            ],
                        ),
                    ),
                    output=ir.SymRef(id="out"),
                    inputs=[ir.SymRef(id="inp"), ir.SymRef(id="_gtmp_0")],
                ),
            ],
        ),
        params=[
            ir.Sym(id="i"),
            ir.Sym(id="j"),
            ir.Sym(id="k"),
            ir.Sym(id="inp"),
            ir.Sym(id="out"),
        ],
        tmps=[
            Temporary(id="_gtmp_0"),
            Temporary(id="_gtmp_1"),
        ],
    )
    expected = FencilWithTemporaries(
        fencil=testee.fencil,
        params=testee.params,
        tmps=[
            Temporary(id="_gtmp_0", domain=tmp_domain, dtype=3),
            Temporary(id="_gtmp_1", domain=tmp_domain, dtype=3),
        ],
    )
    actual = collect_tmps_info(testee)
    assert actual == expected
