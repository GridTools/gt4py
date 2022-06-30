import math
from typing import Iterable

import numpy as np
import pytest

from functional.iterator.builtins import *
from functional.iterator.embedded import (
    NeighborTableOffsetProvider,
    index_field,
    np_as_located_field,
)
from functional.iterator.runtime import CartesianAxis, closure, fendef, fundef, offset

from .test_hdiff import I


def asarray(*lists):
    def _listify(val):
        if isinstance(val, Iterable):
            return val
        else:
            return [val]

    res = list(map(lambda val: np.asarray(_listify(val)), lists))
    return res


IDim = CartesianAxis("IDim")


def asfield(*arrays):
    res = list(map(np_as_located_field(IDim), arrays))
    return res


def fencil(builtin, out, *inps, backend):
    if len(inps) == 1:

        @fundef
        def sten(arg0):
            return builtin(deref(arg0))

        @fendef(offset_provider={})
        def fenimpl(dom, arg0, out):
            closure(dom, sten, out, [arg0])

    elif len(inps) == 2:

        @fundef
        def sten(arg0, arg1):
            return builtin(deref(arg0), deref(arg1))

        @fendef(offset_provider={})
        def fenimpl(dom, arg0, arg1, out):
            closure(dom, sten, out, [arg0, arg1])

    elif len(inps) == 3:

        @fundef
        def sten(arg0, arg1, arg2):
            return builtin(deref(arg0), deref(arg1), deref(arg2))

        @fendef(offset_provider={})
        def fenimpl(dom, arg0, arg1, arg2, out):
            closure(dom, sten, out, [arg0, arg1, arg2])

    else:
        raise AssertionError("Add overload")

    return fenimpl({IDim: range(out.shape[0])}, *inps, out, backend=backend)


@pytest.mark.parametrize(
    "builtin, inputs, expected",
    [
        (plus, [2.0, 3.0], 5.0),
        (minus, [2.0, 3.0], -1.0),
        (multiplies, [2.0, 3.0], 6.0),
        (divides, [6.0, 2.0], 3.0),
        (
            if_,
            [[True, False], [1.0, 1.0], [2.0, 2.0]],
            [1.0, 2.0],
        ),
        (greater, [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]], [True, False, False]),
        (less, [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]], [False, True, False]),
        (eq, [[1, 2], [1, 1]], [True, False]),
        (eq, [[True, False, True], [True, False, False]], [True, True, False]),
        (eq, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], [True, True, True]),
        (not_, [[True, False]], [False, True]),
        (
            and_,
            [[True, True, False, False], [True, False, True, False]],
            [True, False, False, False],
        ),
        (or_, [[True, True, False, False], [True, False, True, False]], [True, True, True, False]),
    ],
)
def test_arithmetic_and_logical_builtins(backend, builtin, inputs, expected):
    backend, validate = backend

    inps = asfield(*asarray(*inputs))
    out = asfield((np.zeros_like(*asarray(expected))))[0]

    fencil(builtin, out, *inps, backend=backend)

    if validate:
        assert np.allclose(np.asarray(out), expected)


@pytest.mark.parametrize(
    "math_fun, ref_impl, inputs",
    [
        # FIXME(ben): what about pow?
        # FIXME(ben): dataset is missing invalid ranges (mostly nan outputs)
        # FIXME(ben): we're not properly testing different datatypes
        (
            abs,
            np.abs,
            ([-1, 1, -1.0, 1.0, 0, -0, 0.0, -0.0],),
        ),
        (
            min,
            # FIXME(ben): what's the signature & semantics of `min` & `max`?
            # `np.min` and python's built-in `min` have very different signatures...
            lambda a, b: np.min(np.stack((a, b)), axis=-1),
            (
                [2, 2.0, 2.0, 3.0, 2, 3, -2, -2.0, -2.0, -3.0, -2, -3],
                [
                    2,
                    2.0,
                    3.0,
                    2.0,
                    3,
                    2,
                    -2,
                    -2.0,
                    -3.0,
                    -2.0,
                    -3,
                    -2,
                ],
            ),
        ),
        (
            max,
            lambda a, b: np.max(np.stack((a, b)), axis=-1),
            (
                [2, 2.0, 2.0, 3.0, 2, 3, -2, -2.0, -2.0, -3.0, -2, -3],
                [
                    2,
                    2.0,
                    3.0,
                    2.0,
                    3,
                    2,
                    -2,
                    -2.0,
                    -3.0,
                    -2.0,
                    -3,
                    -2,
                ],
            ),
        ),
        (
            mod,
            np.mod,
            ([6, 6.0, -6, 6.0, 7, -7.0, 4.8, 4], [2, 2.0, 2.0, -2, 3.0, -3, 1.2, -1.2]),
        ),
        (
            sin,
            np.sin,
            ([0, 0.1, -0.01, np.pi, -2.0 / 3.0 * np.pi, 2.0 * np.pi, 3, 1000, -1000],),
        ),
        (
            cos,
            np.cos,
            ([0, 0.1, -0.01, np.pi, -2.0 / 3.0 * np.pi, 2.0 * np.pi, 3, 1000, -1000],),
        ),
        (
            tan,
            np.tan,
            ([0, 0.1, -0.01, np.pi, -2.0 / 3.0 * np.pi, 2.0 * np.pi, 3, 1000, -1000],),
        ),
        (
            arcsin,
            np.arcsin,
            ([-1.0, -1, -0.7, -0.2, -0.0, 0, 0.0, 0.2, 0.7, 1, 1.0],),
        ),
        (
            arccos,
            np.arccos,
            ([-1.0, -1, -0.7, -0.2, -0.0, 0, 0.0, 0.2, 0.7, 1, 1.0],),
        ),
        (
            arctan,
            np.arctan,
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            sinh,
            np.sinh,
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            cosh,
            np.cosh,
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            tanh,
            np.tanh,
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            arcsinh,
            np.arcsinh,
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            arccosh,
            np.arccosh,
            ([1, 1.0, 1.2, 1.7, 2, 2.0, 100, 103.7, 1000, 1379.89],),
        ),
        (
            arctanh,
            np.arctanh,
            ([-1.0, -1, -0.7, -0.2, -0.0, 0, 0.0, 0.2, 0.7, 1, 1.0],),
        ),
        (
            sqrt,
            np.sqrt,
            (
                [
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.9,
                    1,
                    1.0,
                    2.3,
                    4,
                    4.0,
                    16,
                    16.0,
                    34.7,
                    100,
                    100.0,
                    1000,
                    1337.1337,
                ],
            ),
        ),
        (
            exp,
            np.exp,
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            log,
            np.log,
            (
                [
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.9,
                    1,
                    1.0,
                    2.3,
                    4,
                    4.0,
                    16,
                    16.0,
                    34.7,
                    100,
                    100.0,
                    1000,
                    1337.1337,
                ],
            ),
        ),
        (
            gamma,
            np.frompyfunc(math.gamma, nin=1, nout=1),
            # FIXME(ben): math.gamma throws when it overflows, maybe should instead yield `np.inf`?
            # overflows very quickly, already at `173`
            ([-1002.3, -103.7, -1.2, -0.7, -0.1, 0.1, 0.7, 1.0, 1, 1.2, 100, 103.7, 170.5],),
        ),
        (
            cbrt,
            np.cbrt,
            (
                [
                    -1003.2,
                    -704.3,
                    -100.5,
                    -10.4,
                    -1.5,
                    -1.001,
                    -0.7,
                    -0.01,
                    -0.0,
                    0.0,
                    0.01,
                    0.7,
                    1.001,
                    1.5,
                    10.4,
                    100.5,
                    704.3,
                    1003.2,
                ],
            ),
        ),
        (
            isfinite,
            np.isfinite,
            ([1000, 0, 1, np.pi, -np.inf, np.inf, np.nan, np.nan + 1],),
        ),
        (
            isinf,
            np.isinf,
            ([1000, 0, 1, np.pi, -np.inf, np.inf, np.nan, np.nan + 1],),
        ),
        (
            isnan,
            np.isnan,
            # FIXME(ben): would be good to ensure we have nans with different bit patterns
            ([1000, 0, 1, np.pi, -np.inf, np.inf, np.nan, np.nan + 1],),
        ),
        (
            floor,
            np.floor,
            ([-3.4, -1.5, -0.6, -0.1, -0.0, 0.0, 0.1, 0.6, 1.5, 3.4],),
        ),
        (
            ceil,
            np.ceil,
            ([-3.4, -1.5, -0.6, -0.1, -0.0, 0.0, 0.1, 0.6, 1.5, 3.4],),
        ),
        (
            trunc,
            np.trunc,
            ([-3.4, -1.5, -0.6, -0.1, -0.0, 0.0, 0.1, 0.6, 1.5, 3.4],),
        ),
    ],
)
def test_math_function_builtins(backend, math_fun, ref_impl, inputs):
    backend, validate = backend

    inps = asfield(*asarray(*inputs))
    expected = ref_impl(*inputs)

    out = asfield((np.zeros_like(*asarray(expected))))[0]

    fencil(math_fun, out, *inps, backend=backend)

    if validate:
        assert np.allclose(np.asarray(out), expected)


Neighbor = offset("Neighbor")


@fundef
def _can_deref(inp):
    shifted = shift(Neighbor, 0)(inp)
    return if_(can_deref(shifted), deref(shifted), -1)


@fundef
def _can_deref_lifted(inp):
    def foo(a):
        return deref(a)

    shifted = shift(Neighbor, 0)(lift(foo)(inp))
    return if_(can_deref(shifted), deref(shifted), -1)


@pytest.mark.parametrize("stencil", [_can_deref, _can_deref_lifted])
def test_can_deref(backend, stencil):
    backend, validate = backend

    Node = CartesianAxis("Node")

    inp = np_as_located_field(Node)(np.ones((1,)))
    out = np_as_located_field(Node)(np.asarray([0]))

    no_neighbor_tbl = NeighborTableOffsetProvider(np.array([[None]]), Node, Node, 1)
    stencil[{Node: range(1)}](
        inp, out=out, offset_provider={"Neighbor": no_neighbor_tbl}, backend=backend
    )

    if validate:
        assert np.allclose(np.asarray(out), -1.0)

    a_neighbor_tbl = NeighborTableOffsetProvider(np.array([[0]]), Node, Node, 1)
    stencil[{Node: range(1)}](
        inp, out=out, offset_provider={"Neighbor": a_neighbor_tbl}, backend=backend
    )

    if validate:
        assert np.allclose(np.asarray(out), 1.0)


# def test_can_deref_lifted(backend):
#     backend, validate = backend

#     Neighbor = offset("Neighbor")
#     Node = CartesianAxis("Node")

#     @fundef
#     def _can_deref(inp):
#         shifted = shift(Neighbor, 0)(inp)
#         return if_(can_deref(shifted), 1, -1)

#     inp = np_as_located_field(Node)(np.zeros((1,)))
#     out = np_as_located_field(Node)(np.asarray([0]))

#     no_neighbor_tbl = NeighborTableOffsetProvider(np.array([[None]]), Node, Node, 1)
#     _can_deref[{Node: range(1)}](
#         inp, out=out, offset_provider={"Neighbor": no_neighbor_tbl}, backend=backend
#     )

#     if validate:
#         assert np.allclose(np.asarray(out), -1.0)

#     a_neighbor_tbl = NeighborTableOffsetProvider(np.array([[0]]), Node, Node, 1)
#     _can_deref[{Node: range(1)}](
#         inp, out=out, offset_provider={"Neighbor": a_neighbor_tbl}, backend=backend
#     )

#     if validate:
#         assert np.allclose(np.asarray(out), 1.0)
