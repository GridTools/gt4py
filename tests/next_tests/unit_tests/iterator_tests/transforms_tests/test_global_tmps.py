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
import copy

import gt4py.next as gtx
from gt4py.eve.utils import UIDs
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.global_tmps import (
    AUTO_DOMAIN,
    FencilWithTemporaries,
    SimpleTemporaryExtractionHeuristics,
    Temporary,
    collect_tmps_info,
    split_closures,
    update_domains,
)


def test_split_closures():
    UIDs.reset_sequence()
    testee = ir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[im.sym("d"), im.sym("inp"), im.sym("out")],
        closures=[
            ir.StencilClosure(
                domain=im.call("cartesian_domain")(),
                stencil=im.lambda_("baz_inp")(
                    im.deref(
                        im.lift(
                            im.lambda_("bar_inp")(
                                im.deref(
                                    im.lift(im.lambda_("foo_inp")(im.deref("foo_inp")))("bar_inp")
                                )
                            )
                        )("baz_inp")
                    )
                ),
                output=im.ref("out"),
                inputs=[im.ref("inp")],
            )
        ],
    )

    expected = ir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[
            im.sym("d"),
            im.sym("inp"),
            im.sym("out"),
            im.sym("_tmp_1"),
            im.sym("_tmp_2"),
            im.sym("_gtmp_auto_domain"),
        ],
        closures=[
            ir.StencilClosure(
                domain=AUTO_DOMAIN,
                stencil=im.lambda_("foo_inp")(im.deref("foo_inp")),
                output=im.ref("_tmp_2"),
                inputs=[im.ref("inp")],
            ),
            ir.StencilClosure(
                domain=AUTO_DOMAIN,
                stencil=im.lambda_("bar_inp", "_tmp_2")(im.deref("_tmp_2")),
                output=im.ref("_tmp_1"),
                inputs=[im.ref("inp"), im.ref("_tmp_2")],
            ),
            ir.StencilClosure(
                domain=im.call("cartesian_domain")(),
                stencil=im.lambda_("baz_inp", "_tmp_1")(im.deref("_tmp_1")),
                output=im.ref("out"),
                inputs=[im.ref("inp"), im.ref("_tmp_1")],
            ),
        ],
    )
    actual = split_closures(testee, offset_provider={})
    assert actual.tmps == [Temporary(id="_tmp_1"), Temporary(id="_tmp_2")]
    assert actual.fencil == expected


def test_split_closures_simple_heuristics():
    UIDs.reset_sequence()
    testee = ir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[im.sym("d"), im.sym("inp"), im.sym("out")],
        closures=[
            ir.StencilClosure(
                domain=im.call("cartesian_domain")(),
                stencil=im.lambda_("foo")(
                    im.let("lifted_it", im.lift(im.lambda_("bar")(im.deref("bar")))("foo"))(
                        im.plus(im.deref("lifted_it"), im.deref(im.shift("I", 1)("lifted_it")))
                    )
                ),
                output=im.ref("out"),
                inputs=[im.ref("inp")],
            )
        ],
    )

    expected = ir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[
            im.sym("d"),
            im.sym("inp"),
            im.sym("out"),
            im.sym("_tmp_1"),
            im.sym("_gtmp_auto_domain"),
        ],
        closures=[
            ir.StencilClosure(
                domain=AUTO_DOMAIN,
                stencil=im.lambda_("bar")(im.deref("bar")),
                output=im.ref("_tmp_1"),
                inputs=[im.ref("inp")],
            ),
            ir.StencilClosure(
                domain=im.call("cartesian_domain")(),
                stencil=im.lambda_("foo", "_tmp_1")(
                    im.plus(im.deref("_tmp_1"), im.deref(im.shift("I", 1)("_tmp_1")))
                ),
                output=im.ref("out"),
                inputs=[im.ref("inp"), im.ref("_tmp_1")],
            ),
        ],
    )
    actual = split_closures(
        testee, extraction_heuristics=SimpleTemporaryExtractionHeuristics, offset_provider={}
    )
    assert actual.tmps == [Temporary(id="_tmp_1")]
    assert actual.fencil == expected


def test_split_closures_lifted_scan():
    UIDs.reset_sequence()

    testee = ir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[im.sym("inp"), im.sym("out")],
        closures=[
            ir.StencilClosure(
                domain=im.call("cartesian_domain")(),
                stencil=im.lambda_("a")(
                    im.call(
                        im.call("scan")(
                            im.lambda_("carry", "b")(im.plus("carry", im.deref("b"))),
                            True,
                            im.literal_from_value(0.0),
                        )
                    )(
                        im.lift(
                            im.call("scan")(
                                im.lambda_("carry", "c")(im.plus("carry", im.deref("c"))),
                                False,
                                im.literal_from_value(0.0),
                            )
                        )("a")
                    )
                ),
                output=im.ref("out"),
                inputs=[im.ref("inp")],
            )
        ],
    )

    expected = ir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[im.sym("inp"), im.sym("out"), im.sym("_tmp_1"), im.sym("_gtmp_auto_domain")],
        closures=[
            ir.StencilClosure(
                domain=AUTO_DOMAIN,
                stencil=im.call("scan")(
                    im.lambda_("carry", "c")(im.plus("carry", im.deref("c"))),
                    False,
                    im.literal_from_value(0.0),
                ),
                output=im.ref("_tmp_1"),
                inputs=[im.ref("inp")],
            ),
            ir.StencilClosure(
                domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
                stencil=im.lambda_("a", "_tmp_1")(
                    im.call(
                        im.call("scan")(
                            im.lambda_("carry", "b")(im.plus("carry", im.deref("b"))),
                            True,
                            im.literal_from_value(0.0),
                        )
                    )("_tmp_1")
                ),
                output=im.ref("out"),
                inputs=[im.ref("inp"), im.ref("_tmp_1")],
            ),
        ],
    )

    actual = split_closures(testee, offset_provider={})
    assert actual.tmps == [Temporary(id="_tmp_1")]
    assert actual.fencil == expected


def test_update_cartesian_domains():
    testee = FencilWithTemporaries(
        fencil=ir.FencilDefinition(
            id="f",
            function_definitions=[],
            params=[
                im.sym(name)
                for name in ("i", "j", "k", "inp", "out", "_gtmp_0", "_gtmp_1", "_gtmp_auto_domain")
            ],
            closures=[
                ir.StencilClosure(
                    domain=AUTO_DOMAIN,
                    stencil=im.lambda_("foo_inp")(im.deref("foo_inp")),
                    output=im.ref("_gtmp_1"),
                    inputs=[im.ref("inp")],
                ),
                ir.StencilClosure(
                    domain=AUTO_DOMAIN,
                    stencil=im.ref("deref"),
                    output=im.ref("_gtmp_0"),
                    inputs=[im.ref("_gtmp_1")],
                ),
                ir.StencilClosure(
                    domain=im.call("cartesian_domain")(
                        *(
                            im.call("named_range")(
                                ir.AxisLiteral(value=a),
                                ir.Literal(value="0", type=ir.INTEGER_INDEX_BUILTIN),
                                im.ref(s),
                            )
                            for a, s in (("IDim", "i"), ("JDim", "j"), ("KDim", "k"))
                        )
                    ),
                    stencil=im.lambda_("baz_inp", "_lift_2")(im.deref(im.shift("I", 1)("_lift_2"))),
                    output=im.ref("out"),
                    inputs=[im.ref("inp"), im.ref("_gtmp_0")],
                ),
            ],
        ),
        params=[im.sym("i"), im.sym("j"), im.sym("k"), im.sym("inp"), im.sym("out")],
        tmps=[Temporary(id="_gtmp_0"), Temporary(id="_gtmp_1")],
    )
    expected = copy.deepcopy(testee)
    assert expected.fencil.params.pop() == im.sym("_gtmp_auto_domain")
    expected.fencil.closures[0].domain = ir.FunCall(
        fun=im.ref("cartesian_domain"),
        args=[
            ir.FunCall(
                fun=im.ref("named_range"),
                args=[
                    ir.AxisLiteral(value="IDim"),
                    im.plus(
                        im.literal("0", ir.INTEGER_INDEX_BUILTIN),
                        im.literal("1", ir.INTEGER_INDEX_BUILTIN),
                    ),
                    im.plus(im.ref("i"), ir.Literal(value="1", type=ir.INTEGER_INDEX_BUILTIN)),
                ],
            )
        ]
        + [
            ir.FunCall(
                fun=im.ref("named_range"),
                args=[
                    ir.AxisLiteral(value=a),
                    im.literal("0", ir.INTEGER_INDEX_BUILTIN),
                    im.ref(s),
                ],
            )
            for a, s in (("JDim", "j"), ("KDim", "k"))
        ],
    )
    expected.fencil.closures[1].domain = ir.FunCall(
        fun=im.ref("cartesian_domain"),
        args=[
            ir.FunCall(
                fun=im.ref("named_range"),
                args=[
                    ir.AxisLiteral(value="IDim"),
                    im.plus(
                        im.literal("0", ir.INTEGER_INDEX_BUILTIN),
                        im.literal("1", ir.INTEGER_INDEX_BUILTIN),
                    ),
                    im.plus(im.ref("i"), ir.Literal(value="1", type=ir.INTEGER_INDEX_BUILTIN)),
                ],
            )
        ]
        + [
            ir.FunCall(
                fun=im.ref("named_range"),
                args=[
                    ir.AxisLiteral(value=a),
                    ir.Literal(value="0", type=ir.INTEGER_INDEX_BUILTIN),
                    im.ref(s),
                ],
            )
            for a, s in (("JDim", "j"), ("KDim", "k"))
        ],
    )
    actual = update_domains(testee, {"I": gtx.Dimension("IDim")}, symbolic_sizes=None)
    assert actual == expected


def test_collect_tmps_info():
    tmp_domain = ir.FunCall(
        fun=im.ref("cartesian_domain"),
        args=[
            ir.FunCall(
                fun=im.ref("named_range"),
                args=[
                    ir.AxisLiteral(value="IDim"),
                    ir.Literal(value="0", type=ir.INTEGER_INDEX_BUILTIN),
                    ir.FunCall(
                        fun=im.ref("plus"),
                        args=[im.ref("i"), ir.Literal(value="1", type=ir.INTEGER_INDEX_BUILTIN)],
                    ),
                ],
            )
        ]
        + [
            ir.FunCall(
                fun=im.ref("named_range"),
                args=[
                    ir.AxisLiteral(value=a),
                    ir.Literal(value="0", type=ir.INTEGER_INDEX_BUILTIN),
                    im.ref(s),
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
                ir.Sym(id="inp", dtype=("float64", False)),
                ir.Sym(id="out", dtype=("float64", False)),
                ir.Sym(id="_gtmp_0"),
                ir.Sym(id="_gtmp_1"),
            ],
            closures=[
                ir.StencilClosure(
                    domain=tmp_domain,
                    stencil=ir.Lambda(
                        params=[ir.Sym(id="foo_inp")],
                        expr=ir.FunCall(fun=im.ref("deref"), args=[im.ref("foo_inp")]),
                    ),
                    output=im.ref("_gtmp_1"),
                    inputs=[im.ref("inp")],
                ),
                ir.StencilClosure(
                    domain=tmp_domain,
                    stencil=im.ref("deref"),
                    output=im.ref("_gtmp_0"),
                    inputs=[im.ref("_gtmp_1")],
                ),
                ir.StencilClosure(
                    domain=ir.FunCall(
                        fun=im.ref("cartesian_domain"),
                        args=[
                            ir.FunCall(
                                fun=im.ref("named_range"),
                                args=[
                                    ir.AxisLiteral(value=a),
                                    ir.Literal(value="0", type=ir.INTEGER_INDEX_BUILTIN),
                                    im.ref(s),
                                ],
                            )
                            for a, s in (("IDim", "i"), ("JDim", "j"), ("KDim", "k"))
                        ],
                    ),
                    stencil=ir.Lambda(
                        params=[ir.Sym(id="baz_inp"), ir.Sym(id="_lift_2")],
                        expr=ir.FunCall(
                            fun=im.ref("deref"),
                            args=[
                                ir.FunCall(
                                    fun=ir.FunCall(
                                        fun=im.ref("shift"),
                                        args=[
                                            ir.OffsetLiteral(value="I"),
                                            ir.OffsetLiteral(value=1),
                                        ],
                                    ),
                                    args=[im.ref("_lift_2")],
                                )
                            ],
                        ),
                    ),
                    output=im.ref("out"),
                    inputs=[im.ref("inp"), im.ref("_gtmp_0")],
                ),
            ],
        ),
        params=[ir.Sym(id="i"), ir.Sym(id="j"), ir.Sym(id="k"), ir.Sym(id="inp"), ir.Sym(id="out")],
        tmps=[Temporary(id="_gtmp_0"), Temporary(id="_gtmp_1")],
    )
    expected = FencilWithTemporaries(
        fencil=testee.fencil,
        params=testee.params,
        tmps=[
            Temporary(id="_gtmp_0", domain=tmp_domain, dtype="float64"),
            Temporary(id="_gtmp_1", domain=tmp_domain, dtype="float64"),
        ],
    )
    actual = collect_tmps_info(testee, offset_provider={})
    assert actual == expected
