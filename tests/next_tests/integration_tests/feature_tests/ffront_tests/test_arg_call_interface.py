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

import itertools
import typing

import numpy as np
import pytest

from gt4py.next import errors
from gt4py.next.common import Field
from gt4py.next.ffront.decorator import field_operator, program, scan_operator
from gt4py.next.ffront.fbuiltins import int32, int64
from gt4py.next.program_processors.runners import dace_iterator, gtfn_cpu

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    IDim,
    IField,
    IJKField,
    IJKFloatField,
    JDim,
    KDim,
    cartesian_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    fieldview_backend,
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

    expected = np.asarray(args["a"]) * 2 * np.asarray(args["b"]) - np.asarray(args["c"])

    assert np.allclose(out, expected)


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

    expected = np.asarray(args["a"]) + 2 * np.asarray(args["b"])

    assert np.allclose(args["out"], expected)


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
        a: IField,
        b: IField,
        c: IField,
        out1: IField,
        out2: IField,
        out3: IField,
        out4: IField,
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

    ref = np.asarray(a) + 2 * np.asarray(b) + 3 * np.asarray(c)

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


def test_call_scan_operator_from_field_operator(cartesian_case):
    if cartesian_case.backend in [
        dace_iterator.run_dace_iterator,
        gtfn_cpu.run_gtfn,
        gtfn_cpu.run_gtfn_imperative,
    ]:
        pytest.xfail("Calling scan from field operator not fully supported.")

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
    expected = (1.0 + 3.0 + 5.0 + 7.0) * np.add.accumulate(
        np.asarray(a) + 2.0 * np.asarray(b), axis=2
    )

    cases.verify(cartesian_case, testee, a, b, out=out, ref=expected)


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

    ref = np.add.accumulate(np.asarray(a) + 2 * np.asarray(b), axis=2)

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


def test_scan_wrong_return_type(cartesian_case):
    with pytest.raises(
        errors.DSLError,
        match=(r"Argument `init` to scan operator `testee_scan` must have same type as its return"),
    ):

        @scan_operator(axis=KDim, forward=True, init=0)
        def testee_scan(
            state: int32,
        ) -> float:
            return 1.0

        @program
        def testee(qc: cases.IKFloatField, param_1: int32, param_2: float, scalar: float):
            testee_scan(qc, param_1, param_2, scalar, out=(qc, param_1, param_2))


def test_scan_wrong_state_type(cartesian_case):
    with pytest.raises(
        errors.DSLError,
        match=(
            r"Argument `init` to scan operator `testee_scan` must have same type as `state` argument"
        ),
    ):

        @scan_operator(axis=KDim, forward=True, init=0)
        def testee_scan(
            state: float,
        ) -> int32:
            return 1

        @program
        def testee(qc: cases.IKFloatField, param_1: int32, param_2: float, scalar: float):
            testee_scan(qc, param_1, param_2, scalar, out=(qc, param_1, param_2))
