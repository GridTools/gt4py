# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import re
import typing

import numpy as np
import pytest

from gt4py.next import errors, common, constructors
from gt4py.next.ffront.decorator import field_operator, program, scan_operator
from gt4py.next.ffront.fbuiltins import broadcast, int32

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, IField, IJKFloatField, KDim, cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
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

    testee.with_backend(cartesian_case.backend)(
        *pos_args, **kw_args, out=out, offset_provider=cartesian_case.offset_provider
    )

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

    testee.with_backend(cartesian_case.backend)(
        *pos_args, **kw_args, offset_provider=cartesian_case.offset_provider
    )

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


@pytest.mark.uses_scan
def test_scan_wrong_return_type(cartesian_case):
    with pytest.raises(
        errors.DSLError,
        match=(r"Argument 'init' to scan operator 'testee_scan' must have same type as its return"),
    ):

        @scan_operator(axis=KDim, forward=True, init=0)
        def testee_scan(state: int32) -> float:
            return 1.0

        @program
        def testee(qc: cases.IKFloatField, param_1: int32, param_2: float, scalar: float):
            testee_scan(qc, param_1, param_2, scalar, out=(qc, param_1, param_2))


@pytest.mark.uses_scan
def test_scan_wrong_state_type(cartesian_case):
    with pytest.raises(
        errors.DSLError,
        match=(
            r"Argument 'init' to scan operator 'testee_scan' must have same type as 'state' argument"
        ),
    ):

        @scan_operator(axis=KDim, forward=True, init=0)
        def testee_scan(state: float) -> int32:
            return 1

        @program
        def testee(qc: cases.IKFloatField, param_1: int32, param_2: float, scalar: float):
            testee_scan(qc, param_1, param_2, scalar, out=(qc, param_1, param_2))


@pytest.fixture
def bound_args_testee():
    @field_operator
    def fieldop_bound_args() -> cases.IField:
        return broadcast(0, (IDim,))

    @program
    def program_bound_args(arg1: bool, arg2: bool, out: cases.IField):
        # for the test itself we don't care what happens here, but empty programs are not supported
        fieldop_bound_args(out=out)

    return program_bound_args


def test_bind_invalid_arg(cartesian_case, bound_args_testee):
    with pytest.raises(
        TypeError, match="Keyword argument 'inexistent_arg' is not a valid program parameter."
    ):
        bound_args_testee.with_bound_args(inexistent_arg=1)


def test_call_bound_program_with_wrong_args(cartesian_case, bound_args_testee):
    program_with_bound_arg = bound_args_testee.with_bound_args(arg1=True)
    out = cases.allocate(cartesian_case, bound_args_testee, "out")()

    with pytest.raises(TypeError) as exc_info:
        program_with_bound_arg.with_backend(cartesian_case.backend)(out, offset_provider={})

    assert (
        re.search(
            "Function takes 2 positional arguments, but 1 were given.",
            exc_info.value.__cause__.args[0],
        )
        is not None
    )


def test_call_bound_program_with_already_bound_arg(cartesian_case, bound_args_testee):
    program_with_bound_arg = bound_args_testee.with_bound_args(arg2=True)
    out = cases.allocate(cartesian_case, bound_args_testee, "out")()

    with pytest.raises(TypeError) as exc_info:
        program_with_bound_arg.with_backend(cartesian_case.backend)(
            True, out, arg2=True, offset_provider={}
        )

    assert (
        re.search(
            "Parameter 'arg2' already set as a bound argument.", exc_info.value.__cause__.args[0]
        )
        is not None
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
    ref = inp.array_ns.zeros(size)
    ref[0] = ref[-1] = 42
    ref[1:-1] = inp.ndarray[1:-1]

    cases.verify(cartesian_case, testee, inp, out=out, domain={IDim: (1, size - 1)}, ref=ref)
