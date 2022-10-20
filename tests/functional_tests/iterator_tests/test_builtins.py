import math
from typing import Callable, Iterable

import numpy as np
import pytest

from functional.iterator.builtins import (
    and_,
    can_deref,
    cartesian_domain,
    deref,
    divides,
    eq,
    greater,
    if_,
    less,
    lift,
    minus,
    multiplies,
    named_range,
    not_,
    or_,
    plus,
    shift,
)
from functional.iterator.embedded import NeighborTableOffsetProvider, np_as_located_field
from functional.iterator.runtime import CartesianAxis, closure, fendef, fundef, offset
from functional.program_processors.formatters import type_check
from functional.program_processors.runners.gtfn_cpu import run_gtfn

from .conftest import run_processor
from .math_builtin_test_data import math_builtin_test_data


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


def fencil(builtin, out, *inps, processor, as_column=False):
    column_axis = IDim if as_column else None
    if len(inps) == 1:

        @fundef
        def sten(arg0):
            return builtin(deref(arg0))

        @fendef(offset_provider={}, column_axis=column_axis)
        def fenimpl(size, arg0, out):
            closure(cartesian_domain(named_range(IDim, 0, size)), sten, out, [arg0])

    elif len(inps) == 2:

        @fundef
        def sten(arg0, arg1):
            return builtin(deref(arg0), deref(arg1))

        @fendef(offset_provider={}, column_axis=column_axis)
        def fenimpl(size, arg0, arg1, out):
            closure(cartesian_domain(named_range(IDim, 0, size)), sten, out, [arg0, arg1])

    elif len(inps) == 3:

        @fundef
        def sten(arg0, arg1, arg2):
            return builtin(deref(arg0), deref(arg1), deref(arg2))

        @fendef(offset_provider={}, column_axis=column_axis)
        def fenimpl(size, arg0, arg1, arg2, out):
            closure(cartesian_domain(named_range(IDim, 0, size)), sten, out, [arg0, arg1, arg2])

    else:
        raise AssertionError("Add overload")

    return run_processor(fenimpl, processor, out.shape[0], *inps, out)


@pytest.mark.parametrize("as_column", [False, True])
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
def test_arithmetic_and_logical_builtins(program_processor, builtin, inputs, expected, as_column):
    program_processor, validate = program_processor

    inps = asfield(*asarray(*inputs))
    out = asfield((np.zeros_like(*asarray(expected))))[0]

    fencil(builtin, out, *inps, processor=program_processor, as_column=as_column)

    if validate:
        assert np.allclose(np.asarray(out), expected)


@pytest.mark.parametrize("as_column", [False, True])
@pytest.mark.parametrize("builtin_name, inputs", math_builtin_test_data())
def test_math_function_builtins(program_processor, builtin_name, inputs, as_column):
    from functional.iterator import builtins as it_builtins

    program_processor, validate = program_processor

    if program_processor == type_check.check:
        pytest.xfail("type inference does not yet support math builtins")

    if builtin_name == "gamma":
        # numpy has no gamma function
        ref_impl: Callable = np.vectorize(math.gamma)
    else:
        ref_impl: Callable = getattr(np, builtin_name)

    inps = asfield(*asarray(*inputs))
    expected = ref_impl(*inputs)

    out = asfield((np.zeros_like(*asarray(expected))))[0]

    fencil(
        getattr(it_builtins, builtin_name),
        out,
        *inps,
        processor=program_processor,
        as_column=as_column,
    )

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
def test_can_deref(program_processor, stencil):
    program_processor, validate = program_processor

    if program_processor == run_gtfn:
        pytest.xfail("TODO: gtfn bindings don't support unstructured")

    Node = CartesianAxis("Node")

    inp = np_as_located_field(Node)(np.ones((1,)))
    out = np_as_located_field(Node)(np.asarray([0]))

    no_neighbor_tbl = NeighborTableOffsetProvider(np.array([[None]]), Node, Node, 1)
    run_processor(
        stencil[{Node: range(1)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"Neighbor": no_neighbor_tbl},
    )

    if validate:
        assert np.allclose(np.asarray(out), -1.0)

    a_neighbor_tbl = NeighborTableOffsetProvider(np.array([[0]]), Node, Node, 1)
    run_processor(
        stencil[{Node: range(1)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"Neighbor": a_neighbor_tbl},
    )

    if validate:
        assert np.allclose(np.asarray(out), 1.0)


# def test_can_deref_lifted(program_processor):
#     program_processor, validate = program_processor

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
#         inp, out=out, offset_provider={"Neighbor": no_neighbor_tbl}, program_processor=program_processor
#     )

#     if validate:
#         assert np.allclose(np.asarray(out), -1.0)

#     a_neighbor_tbl = NeighborTableOffsetProvider(np.array([[0]]), Node, Node, 1)
#     _can_deref[{Node: range(1)}](
#         inp, out=out, offset_provider={"Neighbor": a_neighbor_tbl}, program_processor=program_processor
#     )

#     if validate:
#         assert np.allclose(np.asarray(out), 1.0)
