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

import math
from typing import Callable, Iterable

import numpy as np
import pytest

from gt4py.next.iterator import builtins as it_builtins
from gt4py.next.iterator.builtins import (
    and_,
    bool,
    can_deref,
    cartesian_domain,
    cast_,
    deref,
    divides,
    eq,
    float32,
    float64,
    greater,
    greater_equal,
    if_,
    int32,
    int64,
    less,
    less_equal,
    lift,
    minus,
    mod,
    multiplies,
    named_range,
    not_,
    not_eq,
    or_,
    plus,
    shift,
    xor_,
)
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider, np_as_located_field
from gt4py.next.iterator.runtime import CartesianAxis, closure, fendef, fundef, offset
from gt4py.next.program_processors.formatters import type_check
from gt4py.next.program_processors.runners.gtfn_cpu import (
    GTFNExecutor,
    run_gtfn,
    run_gtfn_imperative,
)

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
        def sten(fun, arg0):
            return fun(deref(arg0))

        # keep this indirection to test unapplied builtins (by default transformations will inline)
        @fundef
        def dispatch(arg0):
            return sten(builtin, arg0)

        @fendef(offset_provider={}, column_axis=column_axis)
        def fenimpl(size, arg0, out):
            closure(cartesian_domain(named_range(IDim, 0, size)), dispatch, out, [arg0])

    elif len(inps) == 2:

        @fundef
        def sten(fun, arg0, arg1):
            return fun(deref(arg0), deref(arg1))

        # keep this indirection to test unapplied builtins (by default transformations will inline)
        @fundef
        def dispatch(arg0, arg1):
            return sten(builtin, arg0, arg1)

        @fendef(offset_provider={}, column_axis=column_axis)
        def fenimpl(size, arg0, arg1, out):
            closure(cartesian_domain(named_range(IDim, 0, size)), dispatch, out, [arg0, arg1])

    elif len(inps) == 3:

        @fundef
        def sten(fun, arg0, arg1, arg2):
            return fun(deref(arg0), deref(arg1), deref(arg2))

        # keep this indirection to test unapplied builtins (by default transformations will inline)
        @fundef
        def dispatch(arg0, arg1, arg2):
            return sten(builtin, arg0, arg1, arg2)

        @fendef(offset_provider={}, column_axis=column_axis)
        def fenimpl(size, arg0, arg1, arg2, out):
            closure(cartesian_domain(named_range(IDim, 0, size)), dispatch, out, [arg0, arg1, arg2])

    else:
        raise AssertionError("Add overload")

    return run_processor(fenimpl, processor, out.shape[0], *inps, out)


def arithmetic_and_logical_test_data():
    return [
        # (builtin, inputs, expected)
        (plus, [2.0, 3.0], 5.0),
        (minus, [2.0, 3.0], -1.0),
        (multiplies, [2.0, 3.0], 6.0),
        (divides, [6.0, 2.0], 3.0),
        (
            if_,
            [[True, False], [1.0, 1.0], [2.0, 2.0]],
            [1.0, 2.0],
        ),
        (mod, [5, 2], 1),
        (greater, [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]], [True, False, False]),
        (greater_equal, [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]], [True, False, True]),
        (less, [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]], [False, True, False]),
        (less_equal, [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]], [False, True, True]),
        (eq, [[1, 2], [1, 1]], [True, False]),
        (eq, [[True, False, True], [True, False, False]], [True, True, False]),
        (eq, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], [True, True, True]),
        (not_eq, [[1, 2], [2, 2]], [True, False]),
        (not_, [[True, False]], [False, True]),
        (
            and_,
            [[True, True, False, False], [True, False, True, False]],
            [True, False, False, False],
        ),
        (or_, [[True, True, False, False], [True, False, True, False]], [True, True, True, False]),
        (
            xor_,
            [[True, True, False, False], [True, False, True, False]],
            [False, True, True, False],
        ),
    ]


@pytest.mark.parametrize("as_column", [False, True])
@pytest.mark.parametrize("builtin, inputs, expected", arithmetic_and_logical_test_data())
def test_arithmetic_and_logical_builtins(program_processor, builtin, inputs, expected, as_column):
    program_processor, validate = program_processor

    inps = asfield(*asarray(*inputs))
    out = asfield((np.zeros_like(*asarray(expected))))[0]

    fencil(builtin, out, *inps, processor=program_processor, as_column=as_column)

    if validate:
        assert np.allclose(np.asarray(out), expected)


@pytest.mark.parametrize("builtin, inputs, expected", arithmetic_and_logical_test_data())
def test_arithmetic_and_logical_functors_gtfn(builtin, inputs, expected):
    if builtin == if_:
        pytest.skip("If cannot be used unapplied")
    inps = asfield(*asarray(*inputs))
    out = asfield((np.zeros_like(*asarray(expected))))[0]

    gtfn_without_transforms = GTFNExecutor(
        name="run_gtfn", enable_itir_transforms=False
    )  # avoid inlining the function
    fencil(
        builtin,
        out,
        *inps,
        processor=GTFNExecutor(name="run_gtfn", enable_itir_transforms=False),
    )

    assert np.allclose(np.asarray(out), expected)


@pytest.mark.parametrize("as_column", [False, True])
@pytest.mark.parametrize("builtin_name, inputs", math_builtin_test_data())
def test_math_function_builtins(program_processor, builtin_name, inputs, as_column):
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

    if program_processor == run_gtfn or program_processor == run_gtfn_imperative:
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


# There is no straight-forward way to test cast, because when we define the
# output buffer with the cast-to type an implicit conversion will happen even
# if no explicit cast was done.
# Therefore, we have to set up the test in a way that the explicit cast is required,
# e.g. by a combination of explicit an implicit cast.
# Test setup:
# - Input buffer is setup with the dtype from `input_value`
# - Output buffer is setup with the type of the `expected_value`
# `expected_value` should be chosen with a different type than the explict cast-to type `dtype`.
@pytest.mark.parametrize(
    "input_value, dtype, expected_value",
    [
        (float64("0.1"), float32, float64(float32("0.1"))),
        (int64(42), bool, int64(1)),
        (int64(2147483648), int32, int64(-2147483648)),
        (int64(2147483648), int64, int64(2147483648)),  # int64 does not accidentally down-cast
    ],
)
@pytest.mark.parametrize("as_column", [False, True])
def test_cast(program_processor, as_column, input_value, dtype, expected_value):
    program_processor, validate = program_processor
    column_axis = IDim if as_column else None

    inp = asfield(*asarray(input_value))[0]
    out = asfield((np.zeros_like(*asarray(expected_value))))[0]

    @fundef
    def sten_cast(value):
        return cast_(deref(value), dtype)

    run_processor(
        sten_cast[{IDim: range(1)}],
        program_processor,
        inp,
        out=out,
        offset_provider={},
        column_axis=column_axis,
    )

    if validate:
        assert out[0] == expected_value
