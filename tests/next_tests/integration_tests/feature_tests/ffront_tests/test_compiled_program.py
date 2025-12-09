# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from unittest import mock

import numpy as np
import pytest
import time
import contextlib

from gt4py import next as gtx
from gt4py.next import errors, config
from gt4py.next.otf import compiled_program
from gt4py.next.ffront.decorator import Program
from gt4py.next.ffront.fbuiltins import int32, neighbor_sum

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    V2E,
    cartesian_case,
    mesh_descriptor,
    unstructured_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    MeshDescriptor,
    exec_alloc_descriptor,
    simple_mesh,
    skip_value_mesh,
)


_raise_on_compile = mock.Mock()
_raise_on_compile.compile.side_effect = AssertionError("This function should never be called.")


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
def compile_testee_domain(cartesian_case):
    @gtx.field_operator
    def testee_op(a: cases.IField, b: cases.IField) -> cases.IField:
        return a + b

    @gtx.program(backend=cartesian_case.backend)
    def testee(a: cases.IField, b: cases.IField, out: cases.IField, isize: gtx.int32):
        testee_op(a, b, out=out, domain={cases.IDim: (0, isize)})

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

    compile_testee.compile(offset_provider=cartesian_case.offset_provider)

    args, kwargs = cases.get_default_data(cartesian_case, compile_testee)

    # make sure the backend is never called
    object.__setattr__(compile_testee, "backend", _raise_on_compile)

    compile_testee(*args, offset_provider=cartesian_case.offset_provider, **kwargs)
    assert np.allclose(kwargs["out"].ndarray, args[0].ndarray + args[1].ndarray)

    # run a second time to check if it still works after the future is resolved
    compile_testee(*args, offset_provider=cartesian_case.offset_provider, **kwargs)
    assert np.allclose(kwargs["out"].ndarray, args[0].ndarray + args[1].ndarray)


def test_compile_twice_same_program_errors(cartesian_case, compile_testee):
    if cartesian_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")
    with pytest.raises(ValueError):
        compile_testee.compile(offset_provider=cartesian_case.offset_provider).compile(
            offset_provider=cartesian_case.offset_provider
        )


def test_compile_kwargs(cartesian_case, compile_testee):
    if cartesian_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    compile_testee.compile(offset_provider=cartesian_case.offset_provider)

    (a, b), kwargs = cases.get_default_data(cartesian_case, compile_testee)

    # make sure the backend is never called
    object.__setattr__(compile_testee, "backend", _raise_on_compile)

    compile_testee(offset_provider=cartesian_case.offset_provider, b=b, a=a, **kwargs)
    assert np.allclose(kwargs["out"].ndarray, a.ndarray + b.ndarray)


@pytest.mark.uses_scan
def test_compile_scan(cartesian_case, compile_testee_scan):
    if cartesian_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    compile_testee_scan.compile(offset_provider=cartesian_case.offset_provider)

    args, kwargs = cases.get_default_data(cartesian_case, compile_testee_scan)

    # make sure the backend is never called
    object.__setattr__(compile_testee_scan, "backend", _raise_on_compile)

    compile_testee_scan(*args, offset_provider=cartesian_case.offset_provider, **kwargs)
    assert np.allclose(kwargs["out"].ndarray, np.cumsum(args[0].ndarray))


def test_compile_domain(cartesian_case, compile_testee_domain):
    if cartesian_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    compile_testee_domain.compile(offset_provider=cartesian_case.offset_provider)

    args, kwargs = cases.get_default_data(cartesian_case, compile_testee_domain)

    # make sure the backend is never called
    object.__setattr__(compile_testee_domain, "backend", _raise_on_compile)

    compile_testee_domain(
        *args[:-1],
        isize=cartesian_case.default_sizes[cases.IDim],
        offset_provider=cartesian_case.offset_provider,
        **kwargs,
    )
    assert np.allclose(kwargs["out"].ndarray, args[0].ndarray + args[1].ndarray)


@pytest.fixture
def compile_testee_unstructured(unstructured_case):
    if unstructured_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    @gtx.field_operator
    def testee_op(
        e: cases.EField,
    ) -> cases.VField:
        return neighbor_sum(e(V2E), axis=cases.V2EDim)

    @gtx.program(backend=unstructured_case.backend)
    def testee(
        e: cases.EField,
        out: cases.VField,
    ):
        testee_op(e, out=out)

    return testee


def test_compile_unstructured(unstructured_case, compile_testee_unstructured):
    if unstructured_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    compile_testee_unstructured.compile(
        offset_provider=unstructured_case.offset_provider,
    )

    args, kwargs = cases.get_default_data(unstructured_case, compile_testee_unstructured)

    # make sure the backend is never called
    object.__setattr__(compile_testee_unstructured, "backend", _raise_on_compile)

    compile_testee_unstructured(*args, offset_provider=unstructured_case.offset_provider, **kwargs)

    v2e_numpy = unstructured_case.offset_provider[V2E.value].asnumpy()
    assert np.allclose(
        kwargs["out"].asnumpy(),
        np.sum(np.where(v2e_numpy != -1, args[0].asnumpy()[v2e_numpy], 0), axis=1),
    )


# override mesh descriptor to contain only the simple mesh
@pytest.fixture
def mesh_descriptor(exec_alloc_descriptor):
    return simple_mesh(exec_alloc_descriptor.allocator)


@pytest.fixture
def skip_value_mesh_descriptor(exec_alloc_descriptor):
    return skip_value_mesh(exec_alloc_descriptor.allocator)


def test_compile_unstructured_jit(
    unstructured_case, compile_testee_unstructured, skip_value_mesh_descriptor
):
    if unstructured_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    # compiled for skip_value_mesh and simple_mesh
    compile_testee_unstructured.compile(
        offset_provider=[
            skip_value_mesh_descriptor.offset_provider,
            unstructured_case.offset_provider,
        ],
        enable_jit=False,
    )

    # and executing the simple_mesh
    args, kwargs = cases.get_default_data(unstructured_case, compile_testee_unstructured)
    compile_testee_unstructured(*args, offset_provider=unstructured_case.offset_provider, **kwargs)

    v2e_numpy = unstructured_case.offset_provider[V2E.value].asnumpy()
    assert np.allclose(
        kwargs["out"].asnumpy(),
        np.sum(np.where(v2e_numpy != -1, args[0].asnumpy()[v2e_numpy], 0), axis=1),
    )


def test_compile_unstructured_wrong_offset_provider(
    unstructured_case, compile_testee_unstructured, skip_value_mesh_descriptor
):
    if unstructured_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    # compiled for skip_value_mesh
    compile_testee_unstructured.compile(
        offset_provider=skip_value_mesh_descriptor.offset_provider,
        enable_jit=False,
    )

    # but executing the simple_mesh
    args, kwargs = cases.get_default_data(unstructured_case, compile_testee_unstructured)

    # make sure the backend is never called
    object.__setattr__(compile_testee_unstructured, "backend", _raise_on_compile)

    with pytest.raises(RuntimeError, match="No program.*static.*arg.*"):
        compile_testee_unstructured(
            *args, offset_provider=unstructured_case.offset_provider, **kwargs
        )


def test_compile_unstructured_modified_offset_provider(
    unstructured_case, compile_testee_unstructured, skip_value_mesh_descriptor
):
    if unstructured_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    # compiled for skip_value_mesh
    compile_testee_unstructured.compile(
        offset_provider=skip_value_mesh_descriptor.offset_provider,
        enable_jit=False,
    )

    # but executing the simple_mesh
    args, kwargs = cases.get_default_data(unstructured_case, compile_testee_unstructured)

    # make sure the backend is never called
    object.__setattr__(compile_testee_unstructured, "backend", _raise_on_compile)

    with pytest.raises(RuntimeError, match="No program.*static.*arg.*"):
        compile_testee_unstructured(
            *args, offset_provider=unstructured_case.offset_provider, **kwargs
        )


def test_compile_unstructured_for_two_offset_providers(
    unstructured_case, compile_testee_unstructured, skip_value_mesh_descriptor
):
    if unstructured_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    # compiled for skip_value_mesh and simple_mesh
    compile_testee_unstructured.compile(
        offset_provider=[
            skip_value_mesh_descriptor.offset_provider,
            unstructured_case.offset_provider,
        ],
        enable_jit=False,
    )

    # make sure the backend is never called
    object.__setattr__(compile_testee_unstructured, "backend", _raise_on_compile)

    args, kwargs = cases.get_default_data(unstructured_case, compile_testee_unstructured)
    compile_testee_unstructured(*args, offset_provider=unstructured_case.offset_provider, **kwargs)

    v2e_numpy = unstructured_case.offset_provider[V2E.value].asnumpy()
    assert np.allclose(
        kwargs["out"].asnumpy(),
        np.sum(np.where(v2e_numpy != -1, args[0].asnumpy()[v2e_numpy], 0), axis=1),
    )


@pytest.fixture
def compile_variants_field_operator():
    @gtx.field_operator
    def compile_variants_field_operator(
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

    return compile_variants_field_operator


@pytest.fixture
def compile_variants_testee_not_compiled(
    cartesian_case, compile_variants_field_operator
) -> Program:
    if cartesian_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    @gtx.program(backend=cartesian_case.backend)
    def testee(
        field_a: cases.IField,
        scalar_int: int32,
        scalar_float: float,
        scalar_bool: bool,
        field_b: cases.IFloatField,
        out: tuple[cases.IField, cases.IFloatField],
    ):
        compile_variants_field_operator(
            field_a, scalar_int, scalar_float, scalar_bool, field_b, out=out
        )

    return testee


@pytest.fixture
def compile_variants_testee(cartesian_case, compile_variants_testee_not_compiled) -> Program:
    return compile_variants_testee_not_compiled.compile(
        scalar_int=[1, 2],
        scalar_float=[3.0, 4.0],
        scalar_bool=[True, False],
        offset_provider=cartesian_case.offset_provider,
    )


def test_compile_variants(cartesian_case, compile_variants_testee):
    # make sure the backend is never called
    object.__setattr__(compile_variants_testee, "backend", _raise_on_compile)

    assert compile_variants_testee.static_params == ("scalar_int", "scalar_float", "scalar_bool")

    field_a = cases.allocate(cartesian_case, compile_variants_testee, "field_a")()
    field_b = cases.allocate(cartesian_case, compile_variants_testee, "field_b")()

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

    out = cases.allocate(cartesian_case, compile_variants_testee, "out")()
    compile_variants_testee(
        field_a,
        int32(1),
        4.0,
        False,
        field_b,
        out,
        offset_provider=cartesian_case.offset_provider,
    )
    assert np.allclose(out[0].ndarray, field_a.ndarray - 1)
    assert np.allclose(out[1].ndarray, field_b.ndarray - 4.0)


def test_compile_variants_args_and_kwargs(cartesian_case, compile_variants_testee):
    # make sure the backend is never called
    object.__setattr__(compile_variants_testee, "backend", _raise_on_compile)

    field_a = cases.allocate(cartesian_case, compile_variants_testee, "field_a")()
    field_b = cases.allocate(cartesian_case, compile_variants_testee, "field_b")()

    out = cases.allocate(cartesian_case, compile_variants_testee, "out")()
    compile_variants_testee(
        field_a,
        int32(1),
        scalar_float=3.0,
        scalar_bool=True,
        field_b=field_b,
        out=out,
        offset_provider=cartesian_case.offset_provider,
    )
    assert np.allclose(out[0].ndarray, field_a.ndarray + 1)
    assert np.allclose(out[1].ndarray, field_b.ndarray + 3.0)


def test_compile_variants_not_compiled(cartesian_case, compile_variants_testee):
    object.__setattr__(compile_variants_testee, "enable_jit", False)

    field_a = cases.allocate(cartesian_case, compile_variants_testee, "field_a")()
    field_b = cases.allocate(cartesian_case, compile_variants_testee, "field_b")()
    out = cases.allocate(cartesian_case, compile_variants_testee, "out")()

    with pytest.raises(RuntimeError):
        compile_variants_testee(
            field_a,
            int32(3),  # variant does not exist
            4.0,
            False,
            field_b,
            out=out,
            offset_provider=cartesian_case.offset_provider,
        )


def test_compile_variants_not_compiled_but_jit_enabled_on_call(
    cartesian_case, compile_variants_testee
):
    # disable jit on the program
    object.__setattr__(compile_variants_testee, "enable_jit", False)

    field_a = cases.allocate(cartesian_case, compile_variants_testee, "field_a")()
    field_b = cases.allocate(cartesian_case, compile_variants_testee, "field_b")()
    out = cases.allocate(cartesian_case, compile_variants_testee, "out")()

    with pytest.raises(RuntimeError):
        # fails because jit is disabled on the program and not explicitly enabled on the call
        compile_variants_testee(
            field_a,
            int32(3),  # variant does not exist
            4.0,
            False,
            field_b,
            out=out,
            offset_provider=cartesian_case.offset_provider,
        )

    compile_variants_testee(
        field_a,
        int32(3),  # variant does not exist
        4.0,
        False,
        field_b,
        out=out,
        offset_provider=cartesian_case.offset_provider,
        enable_jit=True,  # explicitly enable jit on the call
    )
    assert np.allclose(out[0].ndarray, field_a.ndarray - 3)
    assert np.allclose(out[1].ndarray, field_b.ndarray - 4.0)


def test_compile_variants_config_default_disable_jit(cartesian_case, compile_variants_testee):
    """
    Checks that changing the config default will be picked up at call time.
    """
    field_a = cases.allocate(cartesian_case, compile_variants_testee, "field_a")()
    field_b = cases.allocate(cartesian_case, compile_variants_testee, "field_b")()
    out = cases.allocate(cartesian_case, compile_variants_testee, "out")()

    # One of the 2 cases will be the non-default.
    with mock.patch.object(config, "ENABLE_JIT_DEFAULT", True):
        compile_variants_testee(
            field_a,
            int32(3),  # variant does not exist
            4.0,
            False,
            field_b,
            out=out,
            offset_provider=cartesian_case.offset_provider,
        )
        assert np.allclose(out[0].ndarray, field_a.ndarray - 3)
        assert np.allclose(out[1].ndarray, field_b.ndarray - 4.0)

    with mock.patch.object(config, "ENABLE_JIT_DEFAULT", False):
        with pytest.raises(RuntimeError):
            compile_variants_testee(
                field_a,
                int32(-42),  # other value than before
                4.0,
                False,
                field_b,
                out=out,
                offset_provider=cartesian_case.offset_provider,
            )


def test_compile_variants_not_compiled_then_reset_static_params(
    cartesian_case, compile_variants_testee
):
    """
    This test ensures that after calling ".with_static_params(None)" the previously compiled programs are gone
    and we can compile for the generic version.
    """
    object.__setattr__(compile_variants_testee, "enable_jit", True)

    field_a = cases.allocate(cartesian_case, compile_variants_testee, "field_a")()
    field_b = cases.allocate(cartesian_case, compile_variants_testee, "field_b")()

    # the compile_variants_testee has static_params set and is compiled (in a previous test)
    assert len(compile_variants_testee.static_params) > 0
    assert compile_variants_testee._compiled_programs is not None

    # but now we reset the compiled programs
    testee_static_float_static_bool = compile_variants_testee.with_static_params(None)

    # Here we jit the generic version (because not static params are set)
    out = cases.allocate(cartesian_case, testee_static_float_static_bool, "out")()
    testee_static_float_static_bool(
        field_a,
        int32(3),  # variant did not exist previously, now it's runtime
        4.0,
        False,
        field_b,
        out=out,
        offset_provider=cartesian_case.offset_provider,
    )
    assert np.allclose(out[0].ndarray, field_a.ndarray - 3)
    assert np.allclose(out[1].ndarray, field_b.ndarray - 4.0)

    # make sure the backend is never called form here on
    object.__setattr__(compile_variants_testee, "backend", _raise_on_compile)

    # calling it again will not recompile
    out = cases.allocate(cartesian_case, testee_static_float_static_bool, "out")()
    testee_static_float_static_bool(
        field_a,
        int32(42),
        5.0,
        True,
        field_b,
        out=out,
        offset_provider=cartesian_case.offset_provider,
    )
    assert np.allclose(out[0].ndarray, field_a.ndarray + 42)
    assert np.allclose(out[1].ndarray, field_b.ndarray + 5.0)


def test_compile_variants_not_compiled_then_set_new_static_params(
    cartesian_case, compile_variants_testee
):
    """
    This test ensures that after calling `with_static_params("scalar_float", "scalar_bool")`
    the previously compiled programs are gone and we can compile for the new `static_params`.
    """
    object.__setattr__(compile_variants_testee, "enable_jit", False)

    field_a = cases.allocate(cartesian_case, compile_variants_testee, "field_a")()
    field_b = cases.allocate(cartesian_case, compile_variants_testee, "field_b")()

    # the compile_variants_testee has static_params set and is compiled (in a previous test)
    assert len(compile_variants_testee.static_params) > 0
    assert compile_variants_testee._compiled_programs is not None

    # but now we reset the compiled programs and fix to other static params
    testee_static_float_static_bool = compile_variants_testee.with_static_params(
        "scalar_float", "scalar_bool"
    )
    testee_static_float_static_bool.compile(
        scalar_float=[4.0], scalar_bool=[False], offset_provider=cartesian_case.offset_provider
    )

    # make sure the backend is never called form here on
    object.__setattr__(compile_variants_testee, "backend", _raise_on_compile)

    out = cases.allocate(cartesian_case, testee_static_float_static_bool, "out")()
    testee_static_float_static_bool(
        field_a,
        int32(3),  # variant did not exist previously, now it's runtime
        4.0,
        False,
        field_b,
        out=out,
        offset_provider=cartesian_case.offset_provider,
    )
    assert np.allclose(out[0].ndarray, field_a.ndarray - 3)
    assert np.allclose(out[1].ndarray, field_b.ndarray - 4.0)

    with pytest.raises(RuntimeError):
        compile_variants_testee(
            field_a,
            int32(3),
            4.0,
            True,  # variant does not exist
            field_b,
            out=out,
            offset_provider=cartesian_case.offset_provider,
        )


def test_compile_variants_jit(cartesian_case, compile_variants_testee):
    object.__setattr__(compile_variants_testee, "enable_jit", True)

    field_a = cases.allocate(cartesian_case, compile_variants_testee, "field_a")()
    field_b = cases.allocate(cartesian_case, compile_variants_testee, "field_b")()

    out = cases.allocate(cartesian_case, compile_variants_testee, "out")()
    compile_variants_testee(
        field_a,
        int32(3),  # variant does not exist
        4.0,
        False,
        field_b,
        out=out,
        offset_provider=cartesian_case.offset_provider,
    )
    assert np.allclose(out[0].ndarray, field_a.ndarray - 3)
    assert np.allclose(out[1].ndarray, field_b.ndarray - 4.0)

    # make sure on the second call the backend is not called
    object.__setattr__(compile_variants_testee, "backend", _raise_on_compile)
    out = cases.allocate(cartesian_case, compile_variants_testee, "out")()
    compile_variants_testee(
        field_a,
        int32(3),
        4.0,
        False,
        field_b,
        out=out,
        offset_provider=cartesian_case.offset_provider,
    )
    assert np.allclose(out[0].ndarray, field_a.ndarray - 3)
    assert np.allclose(out[1].ndarray, field_b.ndarray - 4.0)


def test_compile_variants_with_static_params_jit(
    cartesian_case, compile_variants_testee_not_compiled
):
    object.__setattr__(compile_variants_testee_not_compiled, "enable_jit", True)
    testee_with_static_params = compile_variants_testee_not_compiled.with_static_params(
        "scalar_int", "scalar_float", "scalar_bool"
    )

    field_a = cases.allocate(cartesian_case, testee_with_static_params, "field_a")()
    field_b = cases.allocate(cartesian_case, testee_with_static_params, "field_b")()

    out = cases.allocate(cartesian_case, testee_with_static_params, "out")()
    testee_with_static_params(
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

    # make sure the backend is not called on the second call
    object.__setattr__(testee_with_static_params, "backend", _raise_on_compile)

    out = cases.allocate(cartesian_case, testee_with_static_params, "out")()
    testee_with_static_params(
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


def test_compile_variants_decorator_static_params_jit(
    compile_variants_field_operator, cartesian_case
):
    if cartesian_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    @gtx.program(
        backend=cartesian_case.backend,
        enable_jit=True,
        static_params=("scalar_int", "scalar_float", "scalar_bool"),
    )
    def testee(
        field_a: cases.IField,
        scalar_int: int32,
        scalar_float: float,
        scalar_bool: bool,
        field_b: cases.IFloatField,
        out: tuple[cases.IField, cases.IFloatField],
    ):
        compile_variants_field_operator(
            field_a, scalar_int, scalar_float, scalar_bool, field_b, out=out
        )

    field_a = cases.allocate(cartesian_case, testee, "field_a")()
    field_b = cases.allocate(cartesian_case, testee, "field_b")()

    out = cases.allocate(cartesian_case, testee, "out")()
    testee(
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

    # make sure the backend is not called on the second call
    object.__setattr__(testee, "backend", _raise_on_compile)

    out = cases.allocate(cartesian_case, testee, "out")()
    testee(
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


def test_compile_variants_non_existing_param(cartesian_case, compile_variants_testee_not_compiled):
    with pytest.raises(errors.DSLTypeError, match="non_existing_param"):
        compile_variants_testee_not_compiled.compile(non_existing_param=[1], offset_provider={})


def test_compile_variants_wrong_type(cartesian_case, compile_variants_testee_not_compiled):
    with pytest.raises(errors.DSLTypeError, match="'scalar_int'.*expected.*int32"):
        compile_variants_testee_not_compiled.compile(scalar_int=[1.0], offset_provider={})


def test_compile_variants_error_static_field(cartesian_case, compile_variants_testee_not_compiled):
    field_a = cases.allocate(cartesian_case, compile_variants_testee_not_compiled, "field_a")()
    with pytest.raises(errors.DSLTypeError, match="Invalid static argument.*field_a"):
        compile_variants_testee_not_compiled.compile(field_a=[field_a], offset_provider={})


@pytest.fixture
def compile_variants_testee_tuple(cartesian_case):
    if cartesian_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    @gtx.field_operator
    def testee_op(
        field_a: cases.IField,
        int_tuple: tuple[int32, int32],
        field_b: cases.IField,
    ) -> cases.IField:
        return field_a * int_tuple[0] + field_b * int_tuple[1]

    @gtx.program(backend=cartesian_case.backend)
    def testee(
        field_a: cases.IField,
        int_tuple: tuple[int32, int32],
        field_b: cases.IField,
        out: cases.IField,
    ):
        testee_op(field_a, int_tuple, field_b, out=out)

    return testee.compile(
        int_tuple=[(1, 2), (3, 4)],
        offset_provider=cartesian_case.offset_provider,
    )


def test_compile_variants_tuple(cartesian_case, compile_variants_testee_tuple):
    # make sure the backend is never called
    object.__setattr__(compile_variants_testee_tuple, "backend", _raise_on_compile)

    field_a = cases.allocate(cartesian_case, compile_variants_testee_tuple, "field_a")()
    field_b = cases.allocate(cartesian_case, compile_variants_testee_tuple, "field_b")()

    out = cases.allocate(cartesian_case, compile_variants_testee_tuple, "out")()
    compile_variants_testee_tuple(
        field_a,
        (1, 2),
        field_b,
        out=out,
        offset_provider=cartesian_case.offset_provider,
    )
    assert np.allclose(out.asnumpy(), field_a.asnumpy() * 1 + field_b.asnumpy() * 2)

    out = cases.allocate(cartesian_case, compile_variants_testee_tuple, "out")()
    compile_variants_testee_tuple(
        field_a,
        (3, 4),
        field_b,
        out,
        offset_provider=cartesian_case.offset_provider,
    )
    assert np.allclose(out.asnumpy(), field_a.asnumpy() * 3 + field_b.asnumpy() * 4)


def test_synchronous_compilation(cartesian_case, compile_testee):
    # This test is not perfect: only tests that compilation works if '_async_compilation_pool' is not initialized.
    with mock.patch.object(compiled_program, "_async_compilation_pool", None):
        a = cases.allocate(cartesian_case, compile_testee, "a")()
        b = cases.allocate(cartesian_case, compile_testee, "b")()

        out = cases.allocate(cartesian_case, compile_testee, "out")()
        compile_testee(
            a,
            b,
            out=out,
            offset_provider=cartesian_case.offset_provider,
        )
        assert np.allclose(out.ndarray, a.ndarray + b.ndarray)


@pytest.mark.parametrize("synchronous", [True, False], ids=["synchronous", "asynchronous"])
def test_wait_for_compilation(cartesian_case, compile_testee, compile_testee_domain, synchronous):
    if cartesian_case.backend is None:
        pytest.skip("Embedded compiled program doesn't make sense.")

    with (
        mock.patch.object(compiled_program, "_async_compilation_pool", None)
        if synchronous
        else contextlib.nullcontext()
    ):
        compile_testee.compile(offset_provider=cartesian_case.offset_provider)
        # TODO(havogt): currently only tests that the function call does not crash...
        gtx.wait_for_compilation()
        # ... and afterwards compilation still works
        compile_testee_domain.compile(offset_provider=cartesian_case.offset_provider)
