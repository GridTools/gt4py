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

from gt4py.eve.pattern_matching import ObjectPattern as P
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.fuse_maps import FuseMaps


def _map(op: ir.Expr, *args: ir.Expr) -> ir.FunCall:
    return ir.FunCall(fun=ir.FunCall(fun=ir.SymRef(id="map_"), args=[op]), args=[*args])


def _map_p(op: ir.Expr | P, *args: ir.Expr | P) -> P:
    return P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="map_"), args=[op]), args=[*args])


def _reduce(op: ir.Expr, init: ir.Expr, *args: ir.Expr) -> ir.FunCall:
    return ir.FunCall(fun=ir.FunCall(fun=ir.SymRef(id="reduce"), args=[op, init]), args=[*args])


def _wrap_in_lambda(fun: ir.Expr, *params: str) -> ir.Lambda:
    return ir.Lambda(
        params=[ir.Sym(id=p) for p in params],
        expr=ir.FunCall(fun=fun, args=[ir.SymRef(id=arg) for arg in params]),
    )


def test_simple():
    testee = _map(
        ir.SymRef(id="plus"),
        ir.SymRef(id="a"),
        _map(ir.SymRef(id="multiplies"), ir.SymRef(id="b"), ir.SymRef(id="c")),
    )

    expected = _map_p(
        P(
            ir.Lambda,
            params=[
                P(ir.Sym),
                P(ir.Sym),
                P(ir.Sym),
            ],  # TODO: can we express that the Sym id's match the SymRef id later?
            expr=P(
                ir.FunCall,
                fun=ir.SymRef(id="plus"),
                args=[
                    P(ir.SymRef),
                    P(
                        ir.FunCall,
                        fun=ir.SymRef(id="multiplies"),
                        args=[P(ir.SymRef), P(ir.SymRef)],
                    ),
                ],
            ),
        ),
        P(ir.SymRef),
        P(ir.SymRef),
        P(ir.SymRef),
    )

    actual = FuseMaps().visit(testee)
    assert expected.match(actual)


def test_simple_with_lambdas():
    testee = _map(
        _wrap_in_lambda(ir.SymRef(id="plus"), "x", "y"),
        ir.SymRef(id="a"),
        _map(
            _wrap_in_lambda(ir.SymRef(id="multiplies"), "z", "w"),
            ir.SymRef(id="b"),
            ir.SymRef(id="c"),
        ),
    )

    expected = _map(
        ir.Lambda(
            params=[ir.Sym(id="x"), ir.Sym(id="z"), ir.Sym(id="w")],
            expr=ir.FunCall(
                fun=ir.SymRef(id="plus"),
                args=[
                    ir.SymRef(id="x"),
                    ir.FunCall(
                        fun=ir.SymRef(id="multiplies"), args=[ir.SymRef(id="z"), ir.SymRef(id="w")]
                    ),
                ],
            ),
        ),
        ir.SymRef(id="a"),
        ir.SymRef(id="b"),
        ir.SymRef(id="c"),
    )

    actual = FuseMaps().visit(testee)
    assert expected == actual


def test_simple_reduce():
    testee = _reduce(
        _wrap_in_lambda(ir.SymRef(id="plus"), "x", "y"),
        ir.SymRef(id="init"),
        _map(
            _wrap_in_lambda(ir.SymRef(id="multiplies"), "z", "w"),
            ir.SymRef(id="a"),
            ir.SymRef(id="b"),
        ),
    )

    expected = _reduce(
        ir.Lambda(
            params=[ir.Sym(id="x"), ir.Sym(id="z"), ir.Sym(id="w")],
            expr=ir.FunCall(
                fun=ir.SymRef(id="plus"),
                args=[
                    ir.SymRef(id="x"),
                    ir.FunCall(
                        fun=ir.SymRef(id="multiplies"), args=[ir.SymRef(id="z"), ir.SymRef(id="w")]
                    ),
                ],
            ),
        ),
        ir.SymRef(id="init"),
        ir.SymRef(id="a"),
        ir.SymRef(id="b"),
    )

    actual = FuseMaps().visit(testee)
    assert expected == actual


def test_nested():
    testee = _map(
        _wrap_in_lambda(ir.SymRef(id="plus"), "x", "y"),
        ir.SymRef(id="a"),
        _map(
            _wrap_in_lambda(ir.SymRef(id="multiplies"), "z", "w"),
            ir.SymRef(id="b"),
            _map(
                _wrap_in_lambda(ir.SymRef(id="divides"), "ww", "www"),
                ir.SymRef(id="c"),
                ir.SymRef(id="d"),
            ),
        ),
    )

    expected = _map(
        ir.Lambda(
            params=[ir.Sym(id="x"), ir.Sym(id="z"), ir.Sym(id="ww"), ir.Sym(id="www")],
            expr=ir.FunCall(
                fun=ir.SymRef(id="plus"),
                args=[
                    ir.SymRef(id="x"),
                    ir.FunCall(
                        fun=ir.SymRef(id="multiplies"),
                        args=[
                            ir.SymRef(id="z"),
                            ir.FunCall(
                                fun=ir.SymRef(id="divides"),
                                args=[ir.SymRef(id="ww"), ir.SymRef(id="www")],
                            ),
                        ],
                    ),
                ],
            ),
        ),
        ir.SymRef(id="a"),
        ir.SymRef(id="b"),
        ir.SymRef(id="c"),
        ir.SymRef(id="d"),
    )

    actual = FuseMaps().visit(testee)
    assert expected == actual


def test_multiple_maps():
    testee = _map(
        _wrap_in_lambda(ir.SymRef(id="plus"), "x", "y"),
        _map(
            _wrap_in_lambda(ir.SymRef(id="multiplies"), "z", "w"),
            ir.SymRef(id="a"),
            ir.SymRef(id="b"),
        ),
        _map(
            _wrap_in_lambda(ir.SymRef(id="multiplies"), "ww", "www"),
            ir.SymRef(id="c"),
            ir.SymRef(id="d"),
        ),
    )

    expected = _map(
        ir.Lambda(
            params=[
                ir.Sym(id="_fuse_maps_3"),
                ir.Sym(id="_fuse_maps_4"),
                ir.Sym(id="_fuse_maps_5"),
                ir.Sym(id="_fuse_maps_6"),
            ],
            expr=ir.FunCall(
                fun=ir.SymRef(id="plus"),
                args=[
                    ir.FunCall(
                        fun=ir.SymRef(id="multiplies"),
                        args=[ir.SymRef(id="_fuse_maps_3"), ir.SymRef(id="_fuse_maps_4")],
                    ),
                    ir.FunCall(
                        fun=ir.SymRef(id="multiplies"),
                        args=[ir.SymRef(id="_fuse_maps_5"), ir.SymRef(id="_fuse_maps_6")],
                    ),
                ],
            ),
        ),
        ir.SymRef(id="a"),
        ir.SymRef(id="b"),
        ir.SymRef(id="c"),
        ir.SymRef(id="d"),
    )
    # expected = _map(
    #     ir.Lambda(
    #         params=[ir.Sym(id="z"), ir.Sym(id="w"), ir.Sym(id="ww"), ir.Sym(id="www")],
    #         expr=ir.FunCall(
    #             fun=ir.SymRef(id="plus"),
    #             args=[
    #                 ir.FunCall(
    #                     fun=ir.SymRef(id="multiplies"), args=[ir.SymRef(id="z"), ir.SymRef(id="w")]
    #                 ),
    #                 ir.FunCall(
    #                     fun=ir.SymRef(id="multiplies"),
    #                     args=[ir.SymRef(id="ww"), ir.SymRef(id="www")],
    #                 ),
    #             ],
    #         ),
    #     ),
    #     ir.SymRef(id="a"),
    #     ir.SymRef(id="b"),
    #     ir.SymRef(id="c"),
    #     ir.SymRef(id="d"),
    # )

    actual = FuseMaps().visit(testee)
    assert expected == actual


def test_multiple_maps_with_colliding_symbol_names():
    testee = _map(
        _wrap_in_lambda(ir.SymRef(id="plus"), "x", "y"),
        _map(
            _wrap_in_lambda(ir.SymRef(id="multiplies"), "z", "w"),
            ir.SymRef(id="a"),
            ir.SymRef(id="b"),
        ),
        _map(
            _wrap_in_lambda(ir.SymRef(id="multiplies"), "z", "w"),
            ir.SymRef(id="c"),
            ir.SymRef(id="d"),
        ),
    )

    # TODO pattern macht
    expected = _map(
        ir.Lambda(
            params=[
                ir.Sym(id="_fuse_maps_3"),
                ir.Sym(id="_fuse_maps_4"),
                ir.Sym(id="_fuse_maps_5"),
                ir.Sym(id="_fuse_maps_6"),
            ],
            expr=ir.FunCall(
                fun=ir.SymRef(id="plus"),
                args=[
                    ir.FunCall(
                        fun=ir.SymRef(id="multiplies"),
                        args=[ir.SymRef(id="_fuse_maps_3"), ir.SymRef(id="_fuse_maps_4")],
                    ),
                    ir.FunCall(
                        fun=ir.SymRef(id="multiplies"),
                        args=[ir.SymRef(id="_fuse_maps_5"), ir.SymRef(id="_fuse_maps_6")],
                    ),
                ],
            ),
        ),
        ir.SymRef(id="a"),
        ir.SymRef(id="b"),
        ir.SymRef(id="c"),
        ir.SymRef(id="d"),
    )

    actual = FuseMaps().visit(testee)
    assert expected == actual
