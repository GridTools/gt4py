# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.eve.pattern_matching import ObjectPattern as P
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms import fuse_maps, inline_lambdas


def _map(op: ir.Expr, *args: ir.Expr) -> ir.FunCall:
    return ir.FunCall(fun=ir.FunCall(fun=ir.SymRef(id="map_"), args=[op]), args=[*args])


def _map_p(op: ir.Expr | P, *args: ir.Expr | P) -> P:
    return P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="map_"), args=[op]), args=[*args])


def _reduce(op: ir.Expr, init: ir.Expr, *args: ir.Expr) -> ir.FunCall:
    return ir.FunCall(fun=ir.FunCall(fun=ir.SymRef(id="reduce"), args=[op, init]), args=[*args])


def _reduce_p(op: ir.Expr | P, init: ir.Expr | P, *args: ir.Expr | P) -> P:
    return P(
        ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="reduce"), args=[op, init]), args=[*args]
    )


def _wrap_in_lambda(fun: ir.Expr, *params: str) -> ir.Lambda:
    return ir.Lambda(
        params=[ir.Sym(id=p) for p in params],
        expr=ir.FunCall(fun=fun, args=[ir.SymRef(id=arg) for arg in params]),
    )


_p_sym = P(ir.Sym)
_p_symref = P(ir.SymRef)


def _apply_fuse_maps(ir: ir.Node) -> ir.Node:
    result = fuse_maps.FuseMaps().visit(ir)
    result = inline_lambdas.InlineLambdas.apply(
        result
    )  # FuseMaps does not inline everything which makes the expected result harder to test
    return result


def test_simple():
    testee = _map(
        ir.SymRef(id="plus"),
        ir.SymRef(id="a"),
        _map(ir.SymRef(id="multiplies"), ir.SymRef(id="b"), ir.SymRef(id="c")),
    )

    expected = _map_p(
        P(
            ir.Lambda,
            params=[_p_sym] * 3,
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
        ir.SymRef(id="a"),
        ir.SymRef(id="b"),
        ir.SymRef(id="c"),
    )

    actual = _apply_fuse_maps(testee)
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

    expected = _map_p(
        P(
            ir.Lambda,
            params=[_p_sym] * 3,
            expr=P(
                ir.FunCall,
                fun=ir.SymRef(id="plus"),
                args=[
                    _p_symref,
                    P(ir.FunCall, fun=ir.SymRef(id="multiplies"), args=[_p_symref, _p_symref]),
                ],
            ),
        ),
        ir.SymRef(id="a"),
        ir.SymRef(id="b"),
        ir.SymRef(id="c"),
    )

    actual = _apply_fuse_maps(testee)
    assert expected.match(actual)


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

    expected = _reduce_p(
        P(
            ir.Lambda,
            params=[_p_sym] * 3,
            expr=P(
                ir.FunCall,
                fun=ir.SymRef(id="plus"),
                args=[
                    _p_symref,
                    P(ir.FunCall, fun=ir.SymRef(id="multiplies"), args=[_p_symref, _p_symref]),
                ],
            ),
        ),
        ir.SymRef(id="init"),
        ir.SymRef(id="a"),
        ir.SymRef(id="b"),
    )

    actual = _apply_fuse_maps(testee)
    assert expected.match(actual)


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

    expected = _map_p(
        P(
            ir.Lambda,
            params=[_p_sym] * 4,
            expr=P(
                ir.FunCall,
                fun=ir.SymRef(id="plus"),
                args=[
                    _p_symref,
                    P(
                        ir.FunCall,
                        fun=ir.SymRef(id="multiplies"),
                        args=[
                            _p_symref,
                            P(ir.FunCall, fun=ir.SymRef(id="divides"), args=[_p_symref, _p_symref]),
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

    actual = _apply_fuse_maps(testee)
    assert expected.match(actual)


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

    expected = _map_p(
        P(
            ir.Lambda,
            params=[_p_sym] * 4,
            expr=P(
                ir.FunCall,
                fun=ir.SymRef(id="plus"),
                args=[
                    P(ir.FunCall, fun=ir.SymRef(id="multiplies"), args=[_p_symref, _p_symref]),
                    P(ir.FunCall, fun=ir.SymRef(id="multiplies"), args=[_p_symref, _p_symref]),
                ],
            ),
        ),
        ir.SymRef(id="a"),
        ir.SymRef(id="b"),
        ir.SymRef(id="c"),
        ir.SymRef(id="d"),
    )

    actual = _apply_fuse_maps(testee)
    assert expected.match(actual)
