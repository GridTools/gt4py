# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import dataclasses
import math
from typing import Callable, Iterable

import numpy as np
import pytest

import gt4py.next as gtx
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
    as_fieldop,
)
from gt4py.next.iterator.runtime import set_at, closure, fendef, fundef, offset
from gt4py.next.program_processors.runners.gtfn import run_gtfn

from next_tests.integration_tests.feature_tests.math_builtin_test_data import math_builtin_test_data
from next_tests.unit_tests.conftest import program_processor, run_processor


def array_maker(*lists):
    def _listify(val):
        if isinstance(val, Iterable):
            return val
        else:
            return [val]

    res = list(map(lambda val: np.asarray(_listify(val)), lists))
    return res


IDim = gtx.Dimension("IDim")


def field_maker(*arrays):
    res = list(map(gtx.as_field.partial([IDim]), arrays))
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
            domain = cartesian_domain(named_range(IDim, 0, size))

            set_at(as_fieldop(dispatch, domain)(arg0), domain, out)

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
            domain = cartesian_domain(named_range(IDim, 0, size))

            set_at(as_fieldop(dispatch, domain)(arg0, arg1), domain, out)

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
            domain = cartesian_domain(named_range(IDim, 0, size))

            set_at(as_fieldop(dispatch, domain)(arg0, arg1, arg2), domain, out)

    else:
        raise AssertionError("Add overload.")

    return run_processor(fenimpl, processor, out.shape[0], *inps, out)


def arithmetic_and_logical_test_data():
    return [
        # (builtin, inputs, expected)
        (plus, [2.0, 3.0], 5.0),
        (minus, [2.0, 3.0], -1.0),
        (multiplies, [2.0, 3.0], 6.0),
        (divides, [6.0, 2.0], 3.0),
        (if_, [[True, False], [1.0, 1.0], [2.0, 2.0]], [1.0, 2.0]),
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

    inps = field_maker(*array_maker(*inputs))
    out = field_maker((np.zeros_like(*array_maker(expected))))[0]

    fencil(builtin, out, *inps, processor=program_processor, as_column=as_column)

    if validate:
        assert np.allclose(out.asnumpy(), expected)


@pytest.mark.parametrize("builtin, inputs, expected", arithmetic_and_logical_test_data())
def test_arithmetic_and_logical_functors_gtfn(builtin, inputs, expected):
    if builtin == if_:
        pytest.skip("If cannot be used unapplied")
    inps = field_maker(*array_maker(*inputs))
    out = field_maker((np.zeros_like(*array_maker(expected))))[0]

    gtfn_without_transforms = dataclasses.replace(
        run_gtfn,
        executor=run_gtfn.executor.replace(
            translation=run_gtfn.executor.translation.replace(enable_itir_transforms=False),
        ),  # avoid inlining the function
    )

    fencil(builtin, out, *inps, processor=gtfn_without_transforms)

    assert np.allclose(out.asnumpy(), expected)


@pytest.mark.parametrize("as_column", [False, True])
@pytest.mark.parametrize("builtin_name, inputs", math_builtin_test_data())
def test_math_function_builtins(program_processor, builtin_name, inputs, as_column):
    program_processor, validate = program_processor

    if builtin_name == "gamma":
        # numpy has no gamma function
        ref_impl: Callable = np.vectorize(math.gamma)
    else:
        ref_impl: Callable = getattr(np, builtin_name)

    inps = field_maker(*array_maker(*inputs))
    expected = ref_impl(*inputs)

    out = field_maker((np.zeros_like(*array_maker(expected))))[0]

    fencil(
        getattr(it_builtins, builtin_name),
        out,
        *inps,
        processor=program_processor,
        as_column=as_column,
    )

    if validate:
        assert np.allclose(out.asnumpy(), expected)


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

    Node = gtx.Dimension("Node")

    inp = gtx.as_field([Node], np.ones((1,), dtype=np.int32))
    out = gtx.as_field([Node], np.asarray([0], dtype=inp.dtype))

    no_neighbor_tbl = gtx.NeighborTableOffsetProvider(np.array([[-1]]), Node, Node, 1)
    run_processor(
        stencil[{Node: range(1)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"Neighbor": no_neighbor_tbl},
    )

    if validate:
        assert np.allclose(out.asnumpy(), -1.0)

    a_neighbor_tbl = gtx.NeighborTableOffsetProvider(np.array([[0]]), Node, Node, 1)
    run_processor(
        stencil[{Node: range(1)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"Neighbor": a_neighbor_tbl},
    )

    if validate:
        assert np.allclose(out.asnumpy(), 1.0)


# def test_can_deref_lifted(program_processor):
#     program_processor, validate = program_processor

#     Neighbor = offset("Neighbor")
#     Node = gtx.Dimension("Node")

#     @fundef
#     def _can_deref(inp):
#         shifted = shift(Neighbor, 0)(inp)
#         return if_(can_deref(shifted), 1, -1)

#     inp = gtx.as_field([Node], np.zeros((1,)))
#     out = gtx.as_field([Node], np.asarray([0]))

#     no_neighbor_tbl = gtx.NeighborTableOffsetProvider(np.array([[None]]), Node, Node, 1)
#     _can_deref[{Node: range(1)}](
#         inp, out=out, offset_provider={"Neighbor": no_neighbor_tbl}, program_processor=program_processor
#     )

#     if validate:
#         assert np.allclose(np.asarray(out), -1.0)

#     a_neighbor_tbl = gtx.NeighborTableOffsetProvider(np.array([[0]]), Node, Node, 1)
#     _can_deref[{Node: range(1)}](
#         inp, out=out, offset_provider={"Neighbor": a_neighbor_tbl}, program_processor=program_processor
#     )

#     if validate:
#         assert np.allclose(np.asarray(out), 1.0)


@pytest.mark.parametrize(
    "input_value, dtype, np_dtype",
    [
        (float64("0.1"), float32, np.float32),
        (int64(42), bool, builtins.bool),
        (int64(2147483648), int32, np.int32),
        (int64(2147483648), int64, np.int64),
    ],
)
@pytest.mark.parametrize("as_column", [False, True])
def test_cast(program_processor, as_column, input_value, dtype, np_dtype):
    program_processor, validate = program_processor
    column_axis = IDim if as_column else None

    inp = field_maker(np.array([input_value]))[0]

    casted_valued = np_dtype(input_value)

    @fundef
    def sten_cast(it, casted_valued):
        return eq(cast_(deref(it), dtype), deref(casted_valued))

    out = field_maker(np.zeros_like(inp.asnumpy(), dtype=builtins.bool))[0]
    run_processor(
        sten_cast[{IDim: range(1)}],
        program_processor,
        inp,
        casted_valued,
        out=out,
        offset_provider={},
        column_axis=column_axis,
    )

    if validate:
        assert out[0] == True
