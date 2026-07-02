# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import typing

import numpy as np
import pytest

from gt4py import next as gtx
from gt4py.next import errors
from gt4py.next.ffront.decorator import field_operator, program, scan_operator

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, IField, IJKFloatField, KDim, cartesian_case
from next_tests.integration_tests.cases_utils import (
    exec_alloc_descriptor,
)


def _generate_arg_permutations(
    parameters: tuple[str],
) -> typing.Iterable[tuple[tuple[str], tuple[str]]]:
    """
    Given a set of parameters generate all permutations of arguments and keyword arguments.

    In other words give all possible combinations to call a function with positional-or-keyword
    arguments. The return value is an iterable with each element being a tuple of the parameters
    passed as positional arguments and keyword arguments.
    """
    for num_args in range(len(parameters)):
        for kwarg_names in itertools.permutations(parameters[num_args:]):
            yield (parameters[:num_args], kwarg_names)


@pytest.mark.parametrize("arg_spec", _generate_arg_permutations(("a", "b", "c")))
def test_call_field_operator_from_python(cartesian_case, arg_spec: tuple[tuple[str], tuple[str]]):
    @field_operator
    def testee(a: IField, b: IField, c: IField) -> IField:
        return a * 2 * b - c

    args = {name: cases.allocate(cartesian_case, testee, name)() for name in ("a", "b", "c")}
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    # split into arguments we want to pass as positional and keyword
    arg_names, kwarg_names = arg_spec
    pos_args = [args[name] for name in arg_names]
    kw_args = {name: args[name] for name in kwarg_names}

    testee.with_backend(cartesian_case.backend)(*pos_args, **kw_args, out=out)

    expected = args["a"] * 2 * args["b"] - args["c"]

    assert np.allclose(out.asnumpy(), expected.asnumpy())


@pytest.mark.parametrize("arg_spec", _generate_arg_permutations(("a", "b", "out")))
def test_call_program_from_python(cartesian_case, arg_spec):
    @field_operator
    def foo(a: IField, b: IField) -> IField:
        return a + 2 * b

    @program
    def testee(a: IField, b: IField, out: IField):
        foo(a, b, out=out)

    args = {name: cases.allocate(cartesian_case, testee, name)() for name in ("a", "b", "out")}

    # split into arguments we want to pass as positional and keyword
    arg_names, kwarg_names = arg_spec
    pos_args = [args[name] for name in arg_names]
    kw_args = {name: args[name] for name in kwarg_names}

    testee.with_backend(cartesian_case.backend)(*pos_args, **kw_args)

    expected = args["a"] + 2 * args["b"]

    assert np.allclose(args["out"].asnumpy(), expected.asnumpy())


def test_call_field_operator_from_field_operator(cartesian_case):
    @field_operator
    def foo(x: IField, y: IField, z: IField):
        return x + 2 * y + 3 * z

    @field_operator
    def testee(a: IField, b: IField, c: IField) -> IField:
        return foo(a, b, c) + 5 * foo(a, y=b, z=c) + 7 * foo(a, z=c, y=b) + 11 * foo(a, b, z=c)

    def foo_np(x, y, z):
        return x + 2 * y + 3 * z

    def testee_np(a, b, c):
        return (
            foo_np(a, b, c)
            + 5 * foo_np(a, y=b, z=c)
            + 7 * foo_np(a, z=c, y=b)
            + 11 * foo_np(a, b, z=c)
        )

    cases.verify_with_default_data(cartesian_case, testee, ref=testee_np)


def test_call_field_operator_from_program(cartesian_case):
    @field_operator
    def foo(x: IField, y: IField, z: IField) -> IField:
        return x + 2 * y + 3 * z

    @program
    def testee(
        a: IField, b: IField, c: IField, out1: IField, out2: IField, out3: IField, out4: IField
    ):
        foo(a, b, c, out=out1)
        foo(a, y=b, z=c, out=out2)
        foo(a, z=c, y=b, out=out3)
        foo(a, b, z=c, out=out4)

    a, b, c = (cases.allocate(cartesian_case, testee, name)() for name in ("a", "b", "c"))
    out = (
        cases.allocate(cartesian_case, testee, name, strategy=cases.ZeroInitializer())()
        for name in ("out1", "out2", "out3", "out4")
    )

    ref = a + 2 * b + 3 * c

    cases.verify(
        cartesian_case,
        testee,
        a,
        b,
        c,
        *out,
        inout=out,
        ref=(ref, ref, ref, ref),
        comparison=lambda out, ref: all(map(np.allclose, zip(out, ref))),
    )


@pytest.mark.uses_scan
@pytest.mark.uses_scan_in_field_operator
def test_call_scan_operator_from_field_operator(cartesian_case):
    @scan_operator(axis=KDim, forward=True, init=0.0)
    def testee_scan(state: float, x: float, y: float) -> float:
        return state + x + 2.0 * y

    @field_operator
    def testee(a: IJKFloatField, b: IJKFloatField) -> IJKFloatField:
        return (
            testee_scan(a, b)
            + 3.0 * testee_scan(a, y=b)
            + 5.0 * testee_scan(x=a, y=b)
            + 7.0 * testee_scan(y=b, x=a)
        )

    a, b, out = (
        cases.allocate(cartesian_case, testee, name)() for name in ("a", "b", cases.RETURN)
    )
    expected = (1.0 + 3.0 + 5.0 + 7.0) * np.add.accumulate(a.asnumpy() + 2.0 * b.asnumpy(), axis=2)

    cases.verify(cartesian_case, testee, a, b, out=out, ref=expected)


@pytest.mark.uses_scan
def test_call_scan_operator_from_program(cartesian_case):
    @scan_operator(axis=KDim, forward=True, init=0.0)
    def testee_scan(state: float, x: float, y: float) -> float:
        return state + x + 2.0 * y

    @program
    def testee(
        a: IJKFloatField,
        b: IJKFloatField,
        out1: IJKFloatField,
        out2: IJKFloatField,
        out3: IJKFloatField,
        out4: IJKFloatField,
    ):
        testee_scan(a, b, out=out1)
        testee_scan(a, y=b, out=out2)
        testee_scan(x=a, y=b, out=out3)
        testee_scan(y=b, x=a, out=out4)

    a, b = (cases.allocate(cartesian_case, testee, name)() for name in ("a", "b"))
    out = (
        cases.allocate(cartesian_case, testee, name, strategy=cases.ZeroInitializer())()
        for name in ("out1", "out2", "out3", "out4")
    )

    ref = np.add.accumulate(a.asnumpy() + 2 * b.asnumpy(), axis=2)

    cases.verify(
        cartesian_case,
        testee,
        a,
        b,
        *out,
        inout=out,
        ref=(ref, ref, ref, ref),
        comparison=lambda out, ref: all(map(np.allclose, zip(out, ref))),
    )


@pytest.mark.uses_origin
def test_direct_fo_call_with_domain_arg(cartesian_case):
    @field_operator
    def testee(inp: IField) -> IField:
        return inp

    size = cartesian_case.default_sizes[IDim]
    inp = cases.allocate(cartesian_case, testee, "inp").unique()()
    out = cases.allocate(
        cartesian_case, testee, cases.RETURN, strategy=cases.ConstInitializer(42)
    )()
    ref = np.zeros(size)
    ref[0] = ref[-1] = 42
    ref[1:-1] = inp.asnumpy()[1:-1]

    cases.verify(cartesian_case, testee, inp, out=out, domain={IDim: (1, size - 1)}, ref=ref)


@pytest.mark.uses_origin
@pytest.mark.uses_tuple_returns
def test_direct_fo_call_with_domain_arg_tuple_return(cartesian_case):
    @field_operator
    def testee(inp: IField) -> tuple[IField, IField]:
        return (inp, inp)

    size = cartesian_case.default_sizes[IDim]
    inp = cases.allocate(cartesian_case, testee, "inp").unique()()
    out = cases.allocate(
        cartesian_case, testee, cases.RETURN, strategy=cases.ConstInitializer(42)
    )()
    ref = np.zeros(size)
    ref[0] = ref[-1] = 42
    ref[1:-1] = inp.asnumpy()[1:-1]

    cases.verify(cartesian_case, testee, inp, out=out, domain={IDim: (1, size - 1)}, ref=(ref, ref))


def test_missing_arg_field_operator(cartesian_case):
    """Test that calling a field_operator without required args raises an error."""

    @gtx.field_operator(backend=cartesian_case.backend)
    def copy(a: IField) -> IField:
        return a

    a = cases.allocate(cartesian_case, copy, "a")()

    with pytest.raises(errors.MissingArgumentError, match="'out'"):
        _ = copy(a, offset_provider={})


def test_missing_arg_scan_operator(cartesian_case):
    """Test that calling a scan_operator without required args raises an error."""

    @gtx.scan_operator(backend=cartesian_case.backend, axis=KDim, init=0.0, forward=True)
    def sum(state: float, a: float) -> float:
        return state + a

    a = cases.allocate(cartesian_case, sum, "a")()

    with pytest.raises(errors.MissingArgumentError, match="'out'"):
        _ = sum(a, offset_provider={})


def test_missing_arg_program(cartesian_case):
    """Test that calling a program without required args raises an error."""

    @gtx.field_operator
    def copy(a: IField) -> IField:
        return a

    a = cases.allocate(cartesian_case, copy, "a")()
    b = cases.allocate(cartesian_case, copy, cases.RETURN)()

    with pytest.raises(errors.DSLError, match="Invalid call"):

        @gtx.program(backend=cartesian_case.backend)
        def copy_program(a: IField, b: IField) -> IField:
            copy(a)

        _ = copy_program(a, offset_provider={})
