# -*- coding: utf-8 -*-
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

from functools import reduce

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import (
    astype,
    broadcast,
    errors,
    float32,
    float64,
    int32,
    int64,
    minimum,
    neighbor_sum,
    where,
)
from gt4py.next.ffront.experimental import as_offset
from gt4py.next.program_processors.runners import gtfn

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    C2E,
    E2V,
    V2E,
    E2VDim,
    IDim,
    Ioff,
    JDim,
    KDim,
    Koff,
    V2EDim,
    Vertex,
    cartesian_case,
    unstructured_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    fieldview_backend,
    reduction_setup,
)


def test_copy(cartesian_case):  # noqa: F811 # fixtures
    @gtx.field_operator
    def testee(a: cases.IJKField) -> cases.IJKField:
        field_tuple = (a, a)
        field_0 = field_tuple[0]
        field_1 = field_tuple[1]
        return field_0

    cases.verify_with_default_data(cartesian_case, testee, ref=lambda a: a)


@pytest.mark.uses_tuple_returns
def test_multicopy(cartesian_case):  # noqa: F811 # fixtures
    @gtx.field_operator
    def testee(a: cases.IJKField, b: cases.IJKField) -> tuple[cases.IJKField, cases.IJKField]:
        return a, b

    cases.verify_with_default_data(cartesian_case, testee, ref=lambda a, b: (a, b))


def test_cartesian_shift(cartesian_case):  # noqa: F811 # fixtures
    @gtx.field_operator
    def testee(a: cases.IJKField) -> cases.IJKField:
        return a(Ioff[1])

    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out, ref=a[1:])


def test_unstructured_shift(unstructured_case):  # noqa: F811 # fixtures
    @gtx.field_operator
    def testee(a: cases.VField) -> cases.EField:
        return a(E2V[0])

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: a[unstructured_case.offset_provider["E2V"].table[:, 0]],
    )


def test_composed_unstructured_shift(unstructured_case):
    @gtx.field_operator
    def composed_shift_unstructured_flat(inp: cases.VField) -> cases.CField:
        return inp(E2V[0])(C2E[0])

    @gtx.field_operator
    def composed_shift_unstructured_intermediate_result(inp: cases.VField) -> cases.CField:
        tmp = inp(E2V[0])
        return tmp(C2E[0])

    @gtx.field_operator
    def shift_e2v(inp: cases.VField) -> cases.EField:
        return inp(E2V[0])

    @gtx.field_operator
    def composed_shift_unstructured(inp: cases.VField) -> cases.CField:
        return shift_e2v(inp)(C2E[0])

    cases.verify_with_default_data(
        unstructured_case,
        composed_shift_unstructured_flat,
        ref=lambda inp: inp[unstructured_case.offset_provider["E2V"].table[:, 0]][
            unstructured_case.offset_provider["C2E"].table[:, 0]
        ],
    )

    cases.verify_with_default_data(
        unstructured_case,
        composed_shift_unstructured_intermediate_result,
        ref=lambda inp: inp[unstructured_case.offset_provider["E2V"].table[:, 0]][
            unstructured_case.offset_provider["C2E"].table[:, 0]
        ],
        comparison=lambda inp, tmp: np.all(inp == tmp),
    )

    cases.verify_with_default_data(
        unstructured_case,
        composed_shift_unstructured,
        ref=lambda inp: inp[unstructured_case.offset_provider["E2V"].table[:, 0]][
            unstructured_case.offset_provider["C2E"].table[:, 0]
        ],
    )


def test_fold_shifts(cartesian_case):  # noqa: F811 # fixtures
    """Shifting the result of an addition should work."""

    @gtx.field_operator
    def testee(a: cases.IJKField, b: cases.IJKField) -> cases.IJKField:
        tmp = a + b(Ioff[1])
        return tmp(Ioff[1])

    a = cases.allocate(cartesian_case, testee, "a").extend({cases.IDim: (0, 1)})()
    b = cases.allocate(cartesian_case, testee, "b").extend({cases.IDim: (0, 2)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, b, out=out, ref=a.ndarray[1:] + b.ndarray[2:])


def test_tuples(cartesian_case):  # noqa: F811 # fixtures
    @gtx.field_operator
    def testee(a: cases.IJKFloatField, b: cases.IJKFloatField) -> cases.IJKFloatField:
        inps = a, b
        scalars = 1.3, float64(5.0), float64("3.4")
        return (inps[0] * scalars[0] + inps[1] * scalars[1]) * scalars[2]

    cases.verify_with_default_data(
        cartesian_case, testee, ref=lambda a, b: (a * 1.3 + b * 5.0) * 3.4
    )


def test_scalar_arg(unstructured_case):  # noqa: F811 # fixtures
    """Test scalar argument being turned into 0-dim field."""

    @gtx.field_operator
    def testee(a: int32) -> cases.VField:
        return broadcast(a + 1, (Vertex,))

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: np.full(
            [unstructured_case.default_sizes[Vertex]],
            a + 1,
            dtype=int32,
        ),
        comparison=lambda a, b: np.all(a == b),
    )


def test_nested_scalar_arg(unstructured_case):  # noqa: F811 # fixtures
    @gtx.field_operator
    def testee_inner(a: int32) -> cases.VField:
        return broadcast(a + 1, (Vertex,))

    @gtx.field_operator
    def testee(a: int32) -> cases.VField:
        return testee_inner(a + 1)

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: np.full([unstructured_case.default_sizes[Vertex]], a + 2, dtype=int32),
    )


@pytest.mark.uses_index_fields
def test_scalar_arg_with_field(cartesian_case):  # noqa: F811 # fixtures
    @gtx.field_operator
    def testee(a: cases.IJKField, b: int32) -> cases.IJKField:
        tmp = b * a
        return tmp(Ioff[1])

    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1)})()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    ref = a[1:] * b

    cases.verify(cartesian_case, testee, a, b, out=out, ref=ref)


def test_scalar_in_domain_spec_and_fo_call(cartesian_case):  # noqa: F811 # fixtures
    pytest.xfail(
        "Scalar arguments not supported to be used in both domain specification "
        "and as an argument to a field operator."
    )

    @gtx.field_operator
    def testee_op(size: gtx.IndexType) -> gtx.Field[[IDim], gtx.IndexType]:
        return broadcast(size, (IDim,))

    @gtx.program
    def testee(size: gtx.IndexType, out: gtx.Field[[IDim], gtx.IndexType]):
        testee_op(size, out=out, domain={IDim: (0, size)})

    size = cartesian_case.default_sizes[IDim]
    out = cases.allocate(cartesian_case, testee, "out").zeros()()

    cases.verify(
        cartesian_case,
        testee,
        size,
        out=out,
        ref=np.full_like(out, size, dtype=gtx.IndexType),
    )


def test_scalar_scan(cartesian_case):  # noqa: F811 # fixtures
    @gtx.scan_operator(axis=KDim, forward=True, init=(0.0))
    def testee_scan(state: float, qc_in: float, scalar: float) -> float:
        qc = qc_in + state + scalar
        return qc

    @gtx.program
    def testee(qc: cases.IKFloatField, scalar: float):
        testee_scan(qc, scalar, out=qc)

    qc = cases.allocate(cartesian_case, testee, "qc").zeros()()
    scalar = 1.0
    ksize = cartesian_case.default_sizes[KDim]
    expected = np.full((ksize, ksize), np.arange(start=1, stop=11, step=1).astype(float64))

    cases.verify(cartesian_case, testee, qc, scalar, inout=qc, ref=expected)


@pytest.mark.uses_scan_in_field_operator
def test_tuple_scalar_scan(cartesian_case):  # noqa: F811 # fixtures
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
    ksize = cartesian_case.default_sizes[KDim]
    expected = np.full((ksize, ksize), np.arange(start=1.0, stop=11.0), dtype=float)
    cases.verify(cartesian_case, testee_op, qc, tuple_scalar, out=qc, ref=expected)


@pytest.mark.uses_index_fields
def test_scalar_scan_vertical_offset(cartesian_case):  # noqa: F811 # fixtures
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


def test_astype_int(cartesian_case):  # noqa: F811 # fixtures
    @gtx.field_operator
    def testee(a: cases.IFloatField) -> gtx.Field[[IDim], int64]:
        b = astype(a, int64)
        return b

    cases.verify_with_default_data(
        cartesian_case,
        testee,
        ref=lambda a: a.astype(int64),
        comparison=lambda a, b: np.all(a == b),
    )


@pytest.mark.uses_tuple_returns
def test_astype_on_tuples(cartesian_case):  # noqa: F811 # fixtures
    @gtx.field_operator
    def field_op_returning_a_tuple(
        a: cases.IFloatField, b: cases.IFloatField
    ) -> tuple[gtx.Field[[IDim], float], gtx.Field[[IDim], float]]:
        tup = (a, b)
        return tup

    @gtx.field_operator
    def cast_tuple(
        a: cases.IFloatField,
        b: cases.IFloatField,
        a_asint: cases.IField,
        b_asint: cases.IField,
    ) -> tuple[gtx.Field[[IDim], bool], gtx.Field[[IDim], bool]]:
        result = astype(field_op_returning_a_tuple(a, b), int32)
        return (
            result[0] == a_asint,
            result[1] == b_asint,
        )

    @gtx.field_operator
    def cast_nested_tuple(
        a: cases.IFloatField,
        b: cases.IFloatField,
        a_asint: cases.IField,
        b_asint: cases.IField,
    ) -> tuple[gtx.Field[[IDim], bool], gtx.Field[[IDim], bool], gtx.Field[[IDim], bool]]:
        result = astype((a, field_op_returning_a_tuple(a, b)), int32)
        return (
            result[0] == a_asint,
            result[1][0] == a_asint,
            result[1][1] == b_asint,
        )

    a = cases.allocate(cartesian_case, cast_tuple, "a")()
    b = cases.allocate(cartesian_case, cast_tuple, "b")()
    a_asint = gtx.np_as_located_field(IDim)(np.asarray(a).astype(int32))
    b_asint = gtx.np_as_located_field(IDim)(np.asarray(b).astype(int32))
    out_tuple = cases.allocate(cartesian_case, cast_tuple, cases.RETURN)()
    out_nested_tuple = cases.allocate(cartesian_case, cast_nested_tuple, cases.RETURN)()

    cases.verify(
        cartesian_case,
        cast_tuple,
        a,
        b,
        a_asint,
        b_asint,
        out=out_tuple,
        ref=(np.full_like(a, True, dtype=bool), np.full_like(b, True, dtype=bool)),
    )

    cases.verify(
        cartesian_case,
        cast_nested_tuple,
        a,
        b,
        a_asint,
        b_asint,
        out=out_nested_tuple,
        ref=(
            np.full_like(a, True, dtype=bool),
            np.full_like(a, True, dtype=bool),
            np.full_like(b, True, dtype=bool),
        ),
    )


def test_astype_bool_field(cartesian_case):  # noqa: F811 # fixtures
    @gtx.field_operator
    def testee(a: cases.IFloatField) -> gtx.Field[[IDim], bool]:
        b = astype(a, bool)
        return b

    cases.verify_with_default_data(
        cartesian_case,
        testee,
        ref=lambda a: a.astype(bool),
        comparison=lambda a, b: np.all(a == b),
    )


@pytest.mark.parametrize("inp", [0.0, 2.0])
def test_astype_bool_scalar(cartesian_case, inp):  # noqa: F811 # fixtures
    @gtx.field_operator
    def testee(inp: float) -> gtx.Field[[IDim], bool]:
        return broadcast(astype(inp, bool), (IDim,))

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, inp, out=out, ref=bool(inp))


def test_astype_float(cartesian_case):  # noqa: F811 # fixtures
    @gtx.field_operator
    def testee(a: cases.IFloatField) -> gtx.Field[[IDim], np.float32]:
        b = astype(a, float32)
        return b

    cases.verify_with_default_data(
        cartesian_case,
        testee,
        ref=lambda a: a.astype(np.float32),
        comparison=lambda a, b: np.all(a == b),
    )


@pytest.mark.uses_dynamic_offsets
def test_offset_field(cartesian_case):
    ref = np.full(
        (cartesian_case.default_sizes[IDim], cartesian_case.default_sizes[KDim]), True, dtype=bool
    )

    @gtx.field_operator
    def testee(a: cases.IKField, offset_field: cases.IKField) -> gtx.Field[[IDim, KDim], bool]:
        a_i = a(as_offset(Ioff, offset_field))
        a_i_k = a_i(as_offset(Koff, offset_field))
        b_i = a(Ioff[1])
        b_i_k = b_i(Koff[1])
        return a_i_k == b_i_k

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1), KDim: (0, 1)})()
    offset_field = cases.allocate(cartesian_case, testee, "offset_field").strategy(
        cases.ConstInitializer(1)
    )()

    cases.verify(
        cartesian_case,
        testee,
        a,
        offset_field,
        out=out,
        offset_provider={"Ioff": IDim, "Koff": KDim},
        ref=np.full_like(offset_field, True, dtype=bool),
        comparison=lambda out, ref: np.all(out == ref),
    )

    assert np.allclose(out, ref)


def test_nested_tuple_return(cartesian_case):
    @gtx.field_operator
    def pack_tuple(
        a: cases.IField, b: cases.IField
    ) -> tuple[cases.IField, tuple[cases.IField, cases.IField]]:
        return (a, (a, b))

    @gtx.field_operator
    def combine(a: cases.IField, b: cases.IField) -> cases.IField:
        packed = pack_tuple(a, b)
        return packed[0] + packed[1][0] + packed[1][1]

    cases.verify_with_default_data(cartesian_case, combine, ref=lambda a, b: a + a + b)


@pytest.mark.uses_reduction_over_lift_expressions
def test_nested_reduction(unstructured_case):
    @gtx.field_operator
    def testee(a: cases.EField) -> cases.EField:
        tmp = neighbor_sum(a(V2E), axis=V2EDim)
        tmp_2 = neighbor_sum(tmp(E2V), axis=E2VDim)
        return tmp_2

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: np.sum(
            np.sum(a[unstructured_case.offset_provider["V2E"].table], axis=1)[
                unstructured_case.offset_provider["E2V"].table
            ],
            axis=1,
        ),
        comparison=lambda a, tmp_2: np.all(a == tmp_2),
    )


@pytest.mark.xfail(reason="Not yet supported in lowering, requires `map_`ing of inner reduce op.")
def test_nested_reduction_shift_first(unstructured_case):
    @gtx.field_operator
    def testee(inp: cases.EField) -> cases.EField:
        tmp = inp(V2E)
        tmp2 = tmp(E2V)
        return neighbor_sum(neighbor_sum(tmp2, axis=V2EDim), axis=E2VDim)

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda inp: np.sum(
            np.sum(inp[unstructured_case.offset_provider["V2E"].table], axis=1)[
                unstructured_case.offset_provider["E2V"].table
            ],
            axis=1,
        ),
    )


@pytest.mark.uses_tuple_returns
def test_tuple_return_2(unstructured_case):
    @gtx.field_operator
    def testee(a: cases.EField, b: cases.EField) -> tuple[cases.VField, cases.VField]:
        tmp = neighbor_sum(a(V2E), axis=V2EDim)
        tmp_2 = neighbor_sum(b(V2E), axis=V2EDim)
        return tmp, tmp_2

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a, b: [
            np.sum(a[unstructured_case.offset_provider["V2E"].table], axis=1),
            np.sum(b[unstructured_case.offset_provider["V2E"].table], axis=1),
        ],
        comparison=lambda a, tmp: (np.all(a[0] == tmp[0]), np.all(a[1] == tmp[1])),
    )


@pytest.mark.uses_constant_fields
def test_tuple_with_local_field_in_reduction_shifted(unstructured_case):
    @gtx.field_operator
    def reduce_tuple_element(e: cases.EField, v: cases.VField) -> cases.EField:
        tup = e(V2E), v
        red = neighbor_sum(tup[0] + v, axis=V2EDim)
        tmp = red(E2V[0])
        return tmp

    cases.verify_with_default_data(
        unstructured_case,
        reduce_tuple_element,
        ref=lambda e, v: np.sum(
            e[unstructured_case.offset_provider["V2E"].table] + np.tile(v, (4, 1)).T, axis=1
        )[unstructured_case.offset_provider["E2V"].table[:, 0]],
    )


@pytest.mark.uses_tuple_args
def test_tuple_arg(cartesian_case):
    @gtx.field_operator
    def testee(a: tuple[tuple[cases.IField, cases.IField], cases.IField]) -> cases.IField:
        return 3 * a[0][0] + a[0][1] + a[1]

    cases.verify_with_default_data(
        cartesian_case, testee, ref=lambda a: 3 * a[0][0] + a[0][1] + a[1]
    )


@pytest.mark.parametrize("forward", [True, False])
def test_fieldop_from_scan(cartesian_case, forward):
    init = 1.0
    expected = np.arange(init + 1.0, init + 1.0 + cartesian_case.default_sizes[IDim], 1)
    out = gtx.as_field([KDim], np.zeros((cartesian_case.default_sizes[KDim],)))

    if not forward:
        expected = np.flip(expected)

    @gtx.field_operator
    def add(carry: float, foo: float) -> float:
        return carry + foo

    @gtx.scan_operator(axis=KDim, forward=forward, init=init)
    def simple_scan_operator(carry: float) -> float:
        return add(carry, 1.0)

    cases.verify(cartesian_case, simple_scan_operator, out=out, ref=expected)


@pytest.mark.uses_lift_expressions
def test_solve_triag(cartesian_case):
    if cartesian_case.backend in [
        gtfn.run_gtfn,
        gtfn.run_gtfn_imperative,
        gtfn.run_gtfn_with_temporaries,
    ]:
        pytest.xfail("Nested `scan`s requires creating temporaries.")
    if cartesian_case.backend == gtfn.run_gtfn_with_temporaries:
        pytest.xfail("Temporary extraction does not work correctly in combination with scans.")

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
        return np.linalg.solve(matrices, d)

    cases.verify_with_default_data(cartesian_case, solve_tridiag, ref=expected)


@pytest.mark.parametrize("left, right", [(2, 3), (3, 2)])
def test_ternary_operator(cartesian_case, left, right):
    @gtx.field_operator
    def testee(a: cases.IField, b: cases.IField, left: int32, right: int32) -> cases.IField:
        return a if left < right else b

    a = cases.allocate(cartesian_case, testee, "a")()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, b, left, right, out=out, ref=(a if left < right else b))

    @gtx.field_operator
    def testee(left: int32, right: int32) -> cases.IField:
        return broadcast(3, (IDim,)) if left > right else broadcast(4, (IDim,))

    e = np.asarray(a) if left < right else np.asarray(b)
    cases.verify(
        cartesian_case,
        testee,
        left,
        right,
        out=out,
        ref=(np.full(e.shape, 3) if left > right else np.full(e.shape, 4)),
    )


@pytest.mark.parametrize("left, right", [(2, 3), (3, 2)])
@pytest.mark.uses_tuple_returns
def test_ternary_operator_tuple(cartesian_case, left, right):
    @gtx.field_operator
    def testee(
        a: cases.IField, b: cases.IField, left: int32, right: int32
    ) -> tuple[cases.IField, cases.IField]:
        return (a, b) if left < right else (b, a)

    a = cases.allocate(cartesian_case, testee, "a")()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(
        cartesian_case, testee, a, b, left, right, out=out, ref=((a, b) if left < right else (b, a))
    )


@pytest.mark.uses_reduction_over_lift_expressions
def test_ternary_builtin_neighbor_sum(unstructured_case):
    @gtx.field_operator
    def testee(a: cases.EField, b: cases.EField) -> cases.VField:
        tmp = neighbor_sum(b(V2E) if 2 < 3 else a(V2E), axis=V2EDim)
        return tmp

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a, b: (
            np.sum(b[unstructured_case.offset_provider["V2E"].table], axis=1)
            if 2 < 3
            else np.sum(a[unstructured_case.offset_provider["V2E"].table], axis=1)
        ),
    )


def test_ternary_scan(cartesian_case):
    if cartesian_case.backend in [gtfn.run_gtfn_with_temporaries]:
        pytest.xfail("Temporary extraction does not work correctly in combination with scans.")

    @gtx.scan_operator(axis=KDim, forward=True, init=0.0)
    def simple_scan_operator(carry: float, a: float) -> float:
        return carry if carry > a else carry + 1.0

    k_size = cartesian_case.default_sizes[KDim]
    a = gtx.as_field([KDim], 4.0 * np.ones((k_size,)))
    out = gtx.as_field([KDim], np.zeros((k_size,)))

    cases.verify(
        cartesian_case,
        simple_scan_operator,
        a,
        out=out,
        ref=np.asarray([i if i <= 4.0 else 4.0 + 1 for i in range(1, k_size + 1)]),
    )


@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.uses_tuple_returns
def test_scan_nested_tuple_output(forward, cartesian_case):
    if cartesian_case.backend in [gtfn.run_gtfn_with_temporaries]:
        pytest.xfail("Temporary extraction does not work correctly in combination with scans.")

    init = (1, (2, 3))
    k_size = cartesian_case.default_sizes[KDim]
    expected = np.arange(1, 1 + k_size, 1, dtype=int32)
    if not forward:
        expected = np.flip(expected)

    @gtx.scan_operator(axis=KDim, forward=forward, init=init)
    def simple_scan_operator(
        carry: tuple[int32, tuple[int32, int32]]
    ) -> tuple[int32, tuple[int32, int32]]:
        return (carry[0] + 1, (carry[1][0] + 1, carry[1][1] + 1))

    @gtx.program
    def testee(out: tuple[cases.KField, tuple[cases.KField, cases.KField]]):
        simple_scan_operator(out=out)

    cases.verify_with_default_data(
        cartesian_case,
        testee,
        ref=lambda: (expected + 1.0, (expected + 2.0, expected + 3.0)),
        comparison=lambda ref, out: np.all(out[0] == ref[0])
        and np.all(out[1][0] == ref[1][0])
        and np.all(out[1][1] == ref[1][1]),
    )


@pytest.mark.uses_tuple_args
def test_scan_nested_tuple_input(cartesian_case):
    init = 1.0
    k_size = cartesian_case.default_sizes[KDim]
    inp1 = gtx.as_field([KDim], np.ones((k_size,)))
    inp2 = gtx.as_field([KDim], np.arange(0.0, k_size, 1))
    out = gtx.as_field([KDim], np.zeros((k_size,)))

    def prev_levels_iterator(i):
        return range(i + 1)

    expected = np.asarray(
        [
            reduce(lambda prev, i: prev + inp1[i] + inp2[i], prev_levels_iterator(i), init)
            for i in range(k_size)
        ]
    )

    @gtx.scan_operator(axis=KDim, forward=True, init=init)
    def simple_scan_operator(carry: float, a: tuple[float, float]) -> float:
        return carry + a[0] + a[1]

    cases.verify(cartesian_case, simple_scan_operator, (inp1, inp2), out=out, ref=expected)


def test_docstring(cartesian_case):
    @gtx.field_operator
    def fieldop_with_docstring(a: cases.IField) -> cases.IField:
        """My docstring."""
        return a

    @gtx.program
    def test_docstring(a: cases.IField):
        """My docstring."""
        fieldop_with_docstring(a, out=a)

    a = cases.allocate(cartesian_case, test_docstring, "a")()

    cases.verify(cartesian_case, test_docstring, a, inout=a, ref=a)


def test_with_bound_args(cartesian_case):
    @gtx.field_operator
    def fieldop_bound_args(a: cases.IField, scalar: int32, condition: bool) -> cases.IField:
        if not condition:
            scalar = 0
        return a + a + scalar

    @gtx.program
    def program_bound_args(a: cases.IField, scalar: int32, condition: bool, out: cases.IField):
        fieldop_bound_args(a, scalar, condition, out=out)

    a = cases.allocate(cartesian_case, program_bound_args, "a")()
    scalar = int32(1)
    ref = a + a + 1
    out = cases.allocate(cartesian_case, program_bound_args, "out")()

    prog_bounds = program_bound_args.with_bound_args(scalar=scalar, condition=True)
    cases.verify(cartesian_case, prog_bounds, a, out, inout=out, ref=ref)


def test_domain(cartesian_case):
    @gtx.field_operator
    def fieldop_domain(a: cases.IField) -> cases.IField:
        return a + a

    @gtx.program
    def program_domain(a: cases.IField, out: cases.IField):
        fieldop_domain(a, out=out, domain={IDim: (minimum(1, 2), 9)})

    a = cases.allocate(cartesian_case, program_domain, "a")()
    out = cases.allocate(cartesian_case, program_domain, "out")()

    cases.verify(cartesian_case, program_domain, a, out, inout=out[1:9], ref=a[1:9] * 2)


def test_domain_input_bounds(cartesian_case):
    if cartesian_case.backend in [
        gtfn.run_gtfn,
        gtfn.run_gtfn_imperative,
        gtfn.run_gtfn_with_temporaries,
    ]:
        pytest.xfail("FloorDiv not fully supported in gtfn.")

    lower_i = 1
    upper_i = 10

    @gtx.field_operator
    def fieldop_domain(a: cases.IField) -> cases.IField:
        return a + a

    @gtx.program
    def program_domain(
        inp: cases.IField, out: cases.IField, lower_i: gtx.IndexType, upper_i: gtx.IndexType
    ):
        fieldop_domain(
            inp,
            out=out,
            domain={IDim: (lower_i, upper_i // 2)},
        )

    inp = cases.allocate(cartesian_case, program_domain, "inp")()
    out = cases.allocate(cartesian_case, fieldop_domain, cases.RETURN)()

    cases.verify(
        cartesian_case,
        program_domain,
        inp,
        out,
        lower_i,
        upper_i,
        inout=out[lower_i : int(upper_i / 2)],
        ref=inp[lower_i : int(upper_i / 2)] * 2,
    )


def test_domain_input_bounds_1(cartesian_case):
    lower_i = 1
    upper_i = 9
    lower_j = 4
    upper_j = 6

    @gtx.field_operator
    def fieldop_domain(a: cases.IJField) -> cases.IJField:
        return a + a

    @gtx.program(backend=cartesian_case.backend)
    def program_domain(
        a: cases.IJField,
        out: cases.IJField,
        lower_i: gtx.IndexType,
        upper_i: gtx.IndexType,
        lower_j: gtx.IndexType,
        upper_j: gtx.IndexType,
    ):
        fieldop_domain(
            a,
            out=out,
            domain={IDim: (1 * lower_i, upper_i + 0), JDim: (lower_j - 0, upper_j)},
        )

    a = cases.allocate(cartesian_case, program_domain, "a")()
    out = cases.allocate(cartesian_case, program_domain, "out")()

    cases.verify(
        cartesian_case,
        program_domain,
        a,
        out,
        lower_i,
        upper_i,
        lower_j,
        upper_j,
        inout=out[1 * lower_i : upper_i + 0, lower_j - 0 : upper_j],
        ref=a[1 * lower_i : upper_i + 0, lower_j - 0 : upper_j] * 2,
    )


def test_domain_tuple(cartesian_case):
    @gtx.field_operator
    def fieldop_domain_tuple(
        a: cases.IJField, b: cases.IJField
    ) -> tuple[cases.IJField, cases.IJField]:
        return (a + b, b)

    @gtx.program
    def program_domain_tuple(
        inp0: cases.IJField,
        inp1: cases.IJField,
        out0: cases.IJField,
        out1: cases.IJField,
    ):
        fieldop_domain_tuple(inp0, inp1, out=(out0, out1), domain={IDim: (1, 9), JDim: (4, 6)})

    inp0 = cases.allocate(cartesian_case, program_domain_tuple, "inp0")()
    inp1 = cases.allocate(cartesian_case, program_domain_tuple, "inp1")()
    out0 = cases.allocate(cartesian_case, program_domain_tuple, "out0")()
    out1 = cases.allocate(cartesian_case, program_domain_tuple, "out1")()

    cases.verify(
        cartesian_case,
        program_domain_tuple,
        inp0,
        inp1,
        out0,
        out1,
        inout=(out0[1:9, 4:6], out1[1:9, 4:6]),
        ref=(inp0[1:9, 4:6] + inp1[1:9, 4:6], inp1[1:9, 4:6]),
    )


def test_where_k_offset(cartesian_case):
    @gtx.field_operator
    def fieldop_where_k_offset(
        inp: cases.IKField, k_index: gtx.Field[[KDim], gtx.IndexType]
    ) -> cases.IKField:
        return where(k_index > 0, inp(Koff[-1]), 2)

    inp = cases.allocate(cartesian_case, fieldop_where_k_offset, "inp")()
    k_index = cases.allocate(
        cartesian_case, fieldop_where_k_offset, "k_index", strategy=cases.IndexInitializer()
    )()
    out = cases.allocate(cartesian_case, fieldop_where_k_offset, "inp")()

    ref = np.where(np.asarray(k_index) > 0, np.roll(inp, 1, axis=1), 2)

    cases.verify(cartesian_case, fieldop_where_k_offset, inp, k_index, out=out, ref=ref)


def test_undefined_symbols(cartesian_case):
    with pytest.raises(errors.DSLError, match="Undeclared symbol"):

        @gtx.field_operator(backend=cartesian_case.backend)
        def return_undefined():
            return undefined_symbol


@pytest.mark.uses_zero_dimensional_fields
def test_zero_dims_fields(cartesian_case):
    @gtx.field_operator
    def implicit_broadcast_scalar(inp: cases.EmptyField):
        return inp

    inp = cases.allocate(cartesian_case, implicit_broadcast_scalar, "inp")()
    out = cases.allocate(cartesian_case, implicit_broadcast_scalar, "inp")()

    cases.verify(cartesian_case, implicit_broadcast_scalar, inp, out=out, ref=np.array(0))


def test_implicit_broadcast_mixed_dim(cartesian_case):
    @gtx.field_operator
    def fieldop_implicit_broadcast(
        zero_dim_inp: cases.EmptyField, inp: cases.IField, scalar: int32
    ) -> cases.IField:
        return inp + zero_dim_inp * scalar

    @gtx.field_operator
    def fieldop_implicit_broadcast_2(inp: cases.IField) -> cases.IField:
        fi = fieldop_implicit_broadcast(1, inp, 2)
        return fi

    cases.verify_with_default_data(
        cartesian_case, fieldop_implicit_broadcast_2, ref=lambda inp: inp + 2
    )


@pytest.mark.uses_tuple_returns
def test_tuple_unpacking(cartesian_case):
    @gtx.field_operator
    def unpack(
        inp: cases.IField,
    ) -> tuple[cases.IField, cases.IField, cases.IField, cases.IField,]:
        a, b, c, d = (inp + 2, inp + 3, inp + 5, inp + 7)
        return a, b, c, d

    cases.verify_with_default_data(
        cartesian_case, unpack, ref=lambda inp: (inp + 2, inp + 3, inp + 5, inp + 7)
    )


@pytest.mark.uses_tuple_returns
def test_tuple_unpacking_star_multi(cartesian_case):
    OutType = tuple[
        cases.IField,
        cases.IField,
        cases.IField,
        cases.IField,
        cases.IField,
        cases.IField,
        cases.IField,
        cases.IField,
        cases.IField,
        cases.IField,
        cases.IField,
        cases.IField,
    ]

    @gtx.field_operator
    def unpack(
        inp: cases.IField,
    ) -> OutType:
        *a, a2, a3 = (inp, inp + 1, inp + 2, inp + 3)
        b1, *b, b3 = (inp + 4, inp + 5, inp + 6, inp + 7)
        c1, c2, *c = (inp + 8, inp + 9, inp + 10, inp + 11)
        return (a[0], a[1], a2, a3, b1, b[0], b[1], b3, c1, c2, c[0], c[1])

    cases.verify_with_default_data(
        cartesian_case,
        unpack,
        ref=lambda inp: (
            inp,
            inp + 1,
            inp + 2,
            inp + 3,
            inp + 4,
            inp + 5,
            inp + 6,
            inp + 7,
            inp + 8,
            inp + 9,
            inp + 10,
            inp + 11,
        ),
    )


def test_tuple_unpacking_too_many_values(cartesian_case):
    with pytest.raises(
        errors.DSLError,
        match=(r"Could not deduce type: Too many values to unpack \(expected 3\)"),
    ):

        @gtx.field_operator(backend=cartesian_case.backend)
        def _star_unpack() -> tuple[int32, float64, int32]:
            a, b, c = (1, 2.0, 3, 4, 5, 6, 7.0)
            return a, b, c


def test_tuple_unpacking_too_many_values(cartesian_case):
    with pytest.raises(errors.DSLError, match=(r"Assignment value must be of type tuple!")):

        @gtx.field_operator(backend=cartesian_case.backend)
        def _invalid_unpack() -> tuple[int32, float64, int32]:
            a, b, c = 1
            return a


def test_constant_closure_vars(cartesian_case):
    from gt4py.eve.utils import FrozenNamespace

    constants = FrozenNamespace(
        PI=np.float64(3.142),
        E=np.float64(2.718),
    )

    @gtx.field_operator
    def consume_constants(input: cases.IFloatField) -> cases.IFloatField:
        return constants.PI * constants.E * input

    cases.verify_with_default_data(
        cartesian_case, consume_constants, ref=lambda input: constants.PI * constants.E * input
    )
