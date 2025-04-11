# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# TODO(dropd): Remove as soon as `gt4py.next.ffront.decorator` is type checked.
import numpy as np
import pytest

from gt4py import next as gtx
from gt4py.next.ffront.fbuiltins import astype, int32
from gt4py.next.iterator import ir as itir

from next_tests import definitions as test_definitions
from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)


def test_program_gtir_regression(cartesian_case):
    @gtx.field_operator(backend=None)
    def testee_op(a: cases.IField) -> cases.IField:
        return a

    @gtx.program(backend=None)
    def testee(a: cases.IField, out: cases.IField):
        testee_op(a, out=out)

    assert isinstance(testee.gtir, itir.Program)
    assert isinstance(testee.with_backend(cartesian_case.backend).gtir, itir.Program)


def test_frozen(cartesian_case):
    if cartesian_case.backend is None:
        pytest.skip("Frozen Program with embedded execution is not possible.")

    @gtx.field_operator
    def testee_op(a: cases.IField) -> cases.IField:
        return a

    @gtx.program(backend=cartesian_case.backend, frozen=True)
    def testee(a: cases.IField, out: cases.IField):
        testee_op(a, out=out)

    assert isinstance(testee, gtx.ffront.decorator.FrozenProgram)

    # first run should JIT compile
    args_1, kwargs_1 = cases.get_default_data(cartesian_case, testee)
    testee(*args_1, offset_provider=cartesian_case.offset_provider, **kwargs_1)

    # _compiled_program should be set after JIT compiling
    args_2, kwargs_2 = cases.get_default_data(cartesian_case, testee)
    testee._compiled_program(*args_2, offset_provider=cartesian_case.offset_provider, **kwargs_2)

    # and give expected results
    assert np.allclose(kwargs_2["out"].ndarray, args_2[0].ndarray)

    # with_backend returns a new instance, which is frozen but not compiled yet
    assert testee.with_backend(cartesian_case.backend)._compiled_program is None

    # with_grid_type returns a new instance, which is frozen but not compiled yet
    assert testee.with_grid_type(cartesian_case.grid_type)._compiled_program is None


def _always_raise_callable(*args, **kwargs) -> None:
    raise AssertionError("This function should never be called.")


@pytest.fixture
def compile_testee(cartesian_case):
    @gtx.field_operator
    def testee_op(a: cases.IField, b: cases.IField) -> cases.IField:
        return a + b

    @gtx.program(backend=cartesian_case.backend)
    def testee(a: cases.IField, b: cases.IField, out: cases.IField):
        testee_op(a, b, out=out)

    return testee


@pytest.fixture
def compile_testee_scan(cartesian_case):
    @gtx.scan_operator(axis=cases.KDim, forward=True, init=0)
    def testee_op(carry: gtx.int32, inp: gtx.int32) -> gtx.int32:
        return carry + inp

    @gtx.program(backend=cartesian_case.backend)
    def testee(a: cases.KField, out: cases.KField):
        testee_op(a, out=out)

    return testee


def test_compile(cartesian_case, compile_testee):
    if cartesian_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    assert not compile_testee._compiled_programs
    compile_testee.compile(offset_provider_type=cartesian_case.offset_provider)
    assert compile_testee._compiled_programs

    args, kwargs = cases.get_default_data(cartesian_case, compile_testee)

    # make sure the backend is never called
    object.__setattr__(compile_testee, "backend", _always_raise_callable)

    compile_testee(*args, offset_provider=cartesian_case.offset_provider, **kwargs)
    assert np.allclose(kwargs["out"].ndarray, args[0].ndarray + args[1].ndarray)

    # run a second time to check if it still works after the future is resolved
    compile_testee(*args, offset_provider=cartesian_case.offset_provider, **kwargs)
    assert np.allclose(kwargs["out"].ndarray, args[0].ndarray + args[1].ndarray)


def test_compile_twice_errors(cartesian_case, compile_testee):
    if cartesian_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")
    with pytest.raises(RuntimeError):
        compile_testee.compile(offset_provider_type=cartesian_case.offset_provider).compile(
            offset_provider_type=cartesian_case.offset_provider
        )


def test_compile_kwargs(cartesian_case, compile_testee):
    if cartesian_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    assert not compile_testee._compiled_programs
    compile_testee.compile(offset_provider_type=cartesian_case.offset_provider)
    assert compile_testee._compiled_programs

    (a, b), kwargs = cases.get_default_data(cartesian_case, compile_testee)

    # make sure the backend is never called
    object.__setattr__(compile_testee, "backend", _always_raise_callable)

    compile_testee(offset_provider=cartesian_case.offset_provider, b=b, a=a, **kwargs)
    assert np.allclose(kwargs["out"].ndarray, a.ndarray + b.ndarray)


def test_compile_scan(cartesian_case, compile_testee_scan):
    if cartesian_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    assert not compile_testee_scan._compiled_programs
    compile_testee_scan.compile(offset_provider_type=cartesian_case.offset_provider)
    assert compile_testee_scan._compiled_programs

    args, kwargs = cases.get_default_data(cartesian_case, compile_testee_scan)

    # make sure the backend is never called
    object.__setattr__(compile_testee_scan, "backend", _always_raise_callable)

    compile_testee_scan(*args, offset_provider=cartesian_case.offset_provider, **kwargs)
    assert np.allclose(kwargs["out"].ndarray, np.cumsum(args[0].ndarray))


@pytest.fixture
def compile_variants_testee(cartesian_case):
    @gtx.field_operator
    def testee_op(
        field_a: cases.IField,
        scalar_int: int32,
        scalar_float: float,
        scalar_bool: bool,
        field_b: cases.IFloatField,
    ) -> tuple[cases.IField, cases.IFloatField]:
        return (
            (field_a + scalar_int, field_b + scalar_float)
            if scalar_bool
            else (field_a - scalar_int, field_b - scalar_float)
        )

    @gtx.program(backend=cartesian_case.backend)
    def testee(
        field_a: cases.IField,
        scalar_int: int32,
        scalar_float: float,
        scalar_bool: bool,
        field_b: cases.IFloatField,
        out: tuple[cases.IField, cases.IFloatField],
    ):
        testee_op(field_a, scalar_int, scalar_float, scalar_bool, field_b, out=out)

    return testee


def test_compile_variants(cartesian_case, compile_variants_testee):
    if cartesian_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    assert not compile_variants_testee._compiled_programs
    compile_variants_testee.compile(
        scalar_int=[1, 2],
        scalar_float=[3.0, 4.0],
        scalar_bool=[True, False],
        offset_provider_type=cartesian_case.offset_provider,
    )
    assert compile_variants_testee._compiled_programs

    field_a = cases.allocate(cartesian_case, compile_variants_testee, "field_a")()
    field_b = cases.allocate(cartesian_case, compile_variants_testee, "field_b")()

    # make sure the backend is never called
    object.__setattr__(compile_variants_testee, "backend", _always_raise_callable)

    out = cases.allocate(cartesian_case, compile_variants_testee, "out")()

    compile_variants_testee(
        field_a,
        int32(1),
        3.0,
        True,
        field_b,
        out=out,
        offset_provider=cartesian_case.offset_provider,
    )
    assert np.allclose(out[0].ndarray, field_a.ndarray + 1)
    assert np.allclose(out[1].ndarray, field_b.ndarray + 3.0)

    # TODO Can we add a check that we are executing an optimized program?
    # TODO test more variants
    # TODO add test for static tuple values
