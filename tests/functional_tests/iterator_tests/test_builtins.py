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
