# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from functools import reduce

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import errors, float64, int32
from gt4py.next.ffront.decorator import program, scan_operator

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    IDim,
    JDim,
    KDim,
    Koff,
    cartesian_case,
)
from next_tests.integration_tests.cases_utils import (
    exec_alloc_descriptor,
)


@pytest.mark.uses_scan
def test_scalar_scan(cartesian_case):
    @gtx.scan_operator(axis=KDim, forward=True, init=(0.0))
    def testee_scan(state: float, qc_in: float, scalar: float) -> float:
        qc = qc_in + state + scalar
        return qc

    @gtx.program
    def testee(qc: cases.IKFloatField, scalar: float):
        testee_scan(qc, scalar, out=qc)

    qc = cases.allocate(cartesian_case, testee, "qc").zeros()()
    scalar = 1.0
    isize = cartesian_case.default_sizes[IDim]
    ksize = cartesian_case.default_sizes[KDim]
    expected = np.full((isize, ksize), np.arange(start=1, stop=ksize + 1, step=1).astype(float64))

    cases.verify(cartesian_case, testee, qc, scalar, inout=qc, ref=expected)


@pytest.mark.uses_scan
@pytest.mark.uses_scan_in_field_operator
@pytest.mark.uses_tuple_args
def test_tuple_scalar_scan(cartesian_case):
    @gtx.scan_operator(axis=KDim, forward=True, init=0.0)
    def testee_scan(
        state: float, qc_in: float, tuple_scalar: tuple[float, tuple[float, float]]
    ) -> float:
        return (qc_in + state + tuple_scalar[1][0] + tuple_scalar[1][1]) / tuple_scalar[0]

    @gtx.field_operator
    def testee_op(
        qc: cases.IKFloatField, tuple_scalar: tuple[float, tuple[float, float]]
    ) -> cases.IKFloatField:
        return testee_scan(qc, tuple_scalar)

    qc = cases.allocate(cartesian_case, testee_op, "qc").zeros()()
    tuple_scalar = (1.0, (1.0, 0.0))
    isize = cartesian_case.default_sizes[IDim]
    ksize = cartesian_case.default_sizes[KDim]
    expected = np.full((isize, ksize), np.arange(start=1.0, stop=ksize + 1), dtype=float)
    cases.verify(cartesian_case, testee_op, qc, tuple_scalar, out=qc, ref=expected)


@pytest.mark.uses_cartesian_shift
@pytest.mark.uses_scan
@pytest.mark.uses_scan_in_field_operator
def test_scalar_scan_vertical_offset(cartesian_case):
    @gtx.scan_operator(axis=KDim, forward=True, init=(0.0))
    def testee_scan(state: float, inp: float) -> float:
        return inp

    @gtx.field_operator
    def testee(inp: gtx.Field[[KDim], float]) -> gtx.Field[[KDim], float]:
        return testee_scan(inp(Koff[1]))

    inp = cases.allocate(
        cartesian_case,
        testee,
        "inp",
        extend={KDim: (0, 1)},
        strategy=cases.UniqueInitializer(start=2),
    )()
    out = cases.allocate(cartesian_case, testee, "inp").zeros()()
    ksize = cartesian_case.default_sizes[KDim]
    expected = np.full((ksize), np.arange(start=3, stop=ksize + 3, step=1).astype(float64))

    cases.run(cartesian_case, testee, inp, out=out)

    cases.verify(cartesian_case, testee, inp, out=out, ref=expected)


@pytest.mark.uses_scan
@pytest.mark.uses_scan_in_field_operator
def test_scan_unused_parameter(cartesian_case):
    @gtx.scan_operator(axis=KDim, forward=True, init=(0.0))
    def testee_scan(state: float, inp: float, unused: float) -> float:
        return state + inp

    @gtx.field_operator
    def testee(
        inp: gtx.Field[[KDim], float], unused: gtx.Field[[KDim], float]
    ) -> gtx.Field[[KDim], float]:
        return testee_scan(inp, unused)

    inp = cases.allocate(cartesian_case, testee, "inp")()
    unused = cases.allocate(cartesian_case, testee, "unused")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN).zeros()()

    cases.verify(
        cartesian_case,
        testee,
        inp,
        unused,
        out=out,
        ref=np.cumsum(inp.asnumpy(), axis=0),
    )


@pytest.mark.uses_scan
@pytest.mark.uses_scan_without_field_args
@pytest.mark.parametrize("forward", [True, False])
def test_fieldop_from_scan(cartesian_case, forward):
    init = 1.0
    expected = np.arange(init + 1.0, init + 1.0 + cartesian_case.default_sizes[KDim], 1)
    out = cartesian_case.as_field([KDim], np.zeros((cartesian_case.default_sizes[KDim],)))

    if not forward:
        expected = np.flip(expected)

    @gtx.field_operator
    def add(carry: float, foo: float) -> float:
        return carry + foo

    @gtx.scan_operator(axis=KDim, forward=forward, init=init)
    def simple_scan_operator(carry: float) -> float:
        return add(carry, 1.0)

    cases.verify(cartesian_case, simple_scan_operator, out=out, ref=expected)


@pytest.mark.uses_scan
@pytest.mark.uses_scan_nested
@pytest.mark.uses_scan_in_field_operator
def test_solve_triag(cartesian_case):
    @gtx.scan_operator(axis=KDim, forward=True, init=(0.0, 0.0))
    def tridiag_forward(
        state: tuple[float, float], a: float, b: float, c: float, d: float
    ) -> tuple[float, float]:
        return (c / (b - a * state[0]), (d - a * state[1]) / (b - a * state[0]))

    @gtx.scan_operator(axis=KDim, forward=False, init=0.0)
    def tridiag_backward(x_kp1: float, cp: float, dp: float) -> float:
        return dp - cp * x_kp1

    @gtx.field_operator
    def solve_tridiag(
        a: cases.IJKFloatField,
        b: cases.IJKFloatField,
        c: cases.IJKFloatField,
        d: cases.IJKFloatField,
    ) -> cases.IJKFloatField:
        cp, dp = tridiag_forward(a, b, c, d)
        return tridiag_backward(cp, dp)

    def expected(a, b, c, d):
        shape = tuple(cartesian_case.default_sizes[dim] for dim in [IDim, JDim, KDim])
        matrices = np.zeros(shape + shape[-1:])
        i = np.arange(shape[2])
        matrices[:, :, i[1:], i[:-1]] = a[:, :, 1:]
        matrices[:, :, i, i] = b
        matrices[:, :, i[:-1], i[1:]] = c[:, :, :-1]
        # Changed in NumPY version 2.0: In a linear matrix equation ax = b, the b array
        # is only treated as a shape (M,) column vector if it is exactly 1-dimensional.
        # In all other instances it is treated as a stack of (M, K) matrices. Therefore
        # below we add an extra dimension (K) of size 1. Previously b would be treated
        # as a stack of (M,) vectors if b.ndim was equal to a.ndim - 1.
        # Refer to https://numpy.org/doc/2.0/reference/generated/numpy.linalg.solve.html
        d_ext = np.empty(shape=(*shape, 1))
        d_ext[:, :, :, 0] = d
        x = np.linalg.solve(matrices, d_ext)
        return x[:, :, :, 0]

    cases.verify_with_default_data(cartesian_case, solve_tridiag, ref=expected)


@pytest.mark.uses_scan
def test_ternary_scan(cartesian_case):
    @gtx.scan_operator(axis=KDim, forward=True, init=0.0)
    def simple_scan_operator(carry: float, a: float) -> float:
        return carry if carry > a else carry + 1.0

    k_size = cartesian_case.default_sizes[KDim]
    a = cartesian_case.as_field([KDim], 4.0 * np.ones((k_size,)))
    out = cartesian_case.as_field([KDim], np.zeros((k_size,)))

    cases.verify(
        cartesian_case,
        simple_scan_operator,
        a,
        out=out,
        ref=np.asarray([i if i <= 4.0 else 4.0 + 1 for i in range(1, k_size + 1)]),
    )


@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.uses_scan
@pytest.mark.uses_scan_without_field_args
@pytest.mark.uses_tuple_returns
def test_scan_nested_tuple_output(forward, cartesian_case):
    init = (1, (2, 3))
    k_size = cartesian_case.default_sizes[KDim]
    expected = np.arange(1, 1 + k_size, 1, dtype=int32)
    if not forward:
        expected = np.flip(expected)

    @gtx.scan_operator(axis=KDim, forward=forward, init=init)
    def simple_scan_operator(
        carry: tuple[int32, tuple[int32, int32]],
    ) -> tuple[int32, tuple[int32, int32]]:
        return (carry[0] + 1, (carry[1][0] + 1, carry[1][1] + 1))

    @gtx.program
    def testee(out: tuple[cases.KField, tuple[cases.KField, cases.KField]]):
        simple_scan_operator(out=out)

    cases.verify_with_default_data(
        cartesian_case,
        testee,
        ref=lambda: (expected + 1.0, (expected + 2.0, expected + 3.0)),
        comparison=lambda ref, out: (
            np.all(out[0] == ref[0])
            and np.all(out[1][0] == ref[1][0])
            and np.all(out[1][1] == ref[1][1])
        ),
    )


@pytest.mark.uses_scan
@pytest.mark.uses_tuple_args
def test_scan_nested_tuple_input(cartesian_case):
    init = 1.0
    k_size = cartesian_case.default_sizes[KDim]

    inp1_np = np.ones((k_size,))
    inp2_np = np.arange(0.0, k_size, 1)
    inp1 = cartesian_case.as_field([KDim], inp1_np)
    inp2 = cartesian_case.as_field([KDim], inp2_np)
    out = cartesian_case.as_field([KDim], np.zeros((k_size,)))

    def prev_levels_iterator(i):
        return range(i + 1)

    expected = np.asarray(
        [
            reduce(lambda prev, i: prev + inp1_np[i] + inp2_np[i], prev_levels_iterator(i), init)
            for i in range(k_size)
        ]
    )

    @gtx.scan_operator(axis=KDim, forward=True, init=init)
    def simple_scan_operator(carry: float, a: tuple[float, float]) -> float:
        return carry + a[0] + a[1]

    cases.verify(cartesian_case, simple_scan_operator, (inp1, inp2), out=out, ref=expected)


@pytest.mark.uses_scan
@pytest.mark.uses_scan_in_field_operator
@pytest.mark.uses_tuple_args
def test_scan_different_domain_in_tuple(cartesian_case):
    init = 1.0
    i_size = cartesian_case.default_sizes[IDim]
    k_size = cartesian_case.default_sizes[KDim]

    inp1_np = np.ones((i_size + 1, k_size))  # i_size bigger than in the other argument
    inp2_np = np.fromfunction(lambda i, k: k, shape=(i_size, k_size), dtype=float)
    inp1 = cartesian_case.as_field([IDim, KDim], inp1_np)
    inp2 = cartesian_case.as_field([IDim, KDim], inp2_np)
    out = cartesian_case.as_field([IDim, KDim], np.zeros((i_size, k_size)))

    def prev_levels_iterator(i):
        return range(i + 1)

    expected = np.asarray(
        [
            reduce(
                lambda prev, k: prev + inp1_np[:-1, k] + inp2_np[:, k],
                prev_levels_iterator(k),
                init,
            )
            for k in range(k_size)
        ]
    ).transpose()

    @gtx.scan_operator(axis=KDim, forward=True, init=init)
    def scan_op(carry: float, a: tuple[float, float]) -> float:
        return carry + a[0] + a[1]

    @gtx.field_operator
    def foo(
        inp1: gtx.Field[[IDim, KDim], float], inp2: gtx.Field[[IDim, KDim], float]
    ) -> gtx.Field[[IDim, KDim], float]:
        return scan_op((inp1, inp2))

    cases.verify(cartesian_case, foo, inp1, inp2, out=out, ref=expected)


@pytest.mark.uses_scan
@pytest.mark.uses_scan_in_field_operator
@pytest.mark.uses_tuple_args
def test_scan_tuple_field_scalar_mixed(cartesian_case):
    init = 1.0
    i_size = cartesian_case.default_sizes[IDim]
    k_size = cartesian_case.default_sizes[KDim]

    inp2_np = np.fromfunction(lambda i, k: k, shape=(i_size, k_size), dtype=float)
    inp2 = cartesian_case.as_field([IDim, KDim], inp2_np)
    out = cartesian_case.as_field([IDim, KDim], np.zeros((i_size, k_size)))

    def prev_levels_iterator(i):
        return range(i + 1)

    expected = np.asarray(
        [
            reduce(lambda prev, k: prev + 1.0 + inp2_np[:, k], prev_levels_iterator(k), init)
            for k in range(k_size)
        ]
    ).transpose()

    @gtx.scan_operator(axis=KDim, forward=True, init=init)
    def scan_op(carry: float, a: tuple[float, float]) -> float:
        return carry + a[0] + a[1]

    @gtx.field_operator
    def foo(inp1: float, inp2: gtx.Field[[IDim, KDim], float]) -> gtx.Field[[IDim, KDim], float]:
        return scan_op((inp1, inp2))

    cases.verify(cartesian_case, foo, 1.0, inp2, out=out, ref=expected)


@pytest.mark.uses_scan
def test_scan_wrong_return_type(cartesian_case):
    with pytest.raises(
        errors.DSLError,
        match=(
            r"Argument 'state' to scan operator 'testee_scan' must have same type as its return"
        ),
    ):

        @scan_operator(axis=KDim, forward=True, init=0)
        def testee_scan(state: int32) -> float:
            return 1.0

        @program
        def testee(qc: cases.IKFloatField, param_1: int32, param_2: float, scalar: float):
            testee_scan(qc, param_1, param_2, scalar, out=(qc, param_1, param_2))


@pytest.mark.uses_scan
def test_scan_wrong_init_type(cartesian_case):
    with pytest.raises(
        errors.DSLError,
        match=(
            r"Argument 'init' to scan operator 'testee_scan' must have same type as 'state' argument"
        ),
    ):

        @scan_operator(axis=KDim, forward=True, init=0)
        def testee_scan(state: float) -> float:
            return 1.0

        @program
        def testee(qc: cases.IKFloatField, param_1: int32, param_2: float, scalar: float):
            testee_scan(qc, param_1, param_2, scalar, out=(qc, param_1, param_2))


@pytest.mark.uses_scan
def test_scan_without_carry(cartesian_case):
    with pytest.raises(
        errors.DSLError,
        match=r"Scan operator 'testee_scan' must have at least one argument",
    ):

        @scan_operator(axis=KDim, forward=True, init=0)
        def testee_scan() -> float:
            return 1.0
