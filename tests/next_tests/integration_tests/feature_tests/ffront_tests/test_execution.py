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

#
from functools import reduce

import numpy as np
import pytest

from gt4py.next.ffront.decorator import field_operator, program, scan_operator
from gt4py.next.ffront.experimental import as_offset
from gt4py.next.ffront.fbuiltins import (
    Dimension,
    Field,
    astype,
    broadcast,
    float32,
    float64,
    int64,
    maximum,
    minimum,
    neighbor_sum,
    where,
)
from gt4py.next.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError
from gt4py.next.iterator.embedded import index_field, np_as_located_field
from gt4py.next.program_processors.runners import gtfn_cpu

from next_tests.integration_tests.feature_tests import cases
from next_tests.integration_tests.feature_tests.cases import (
    E2V,
    V2E,
    E2VDim,
    Edge,
    IDim,
    Ioff,
    JDim,
    Joff,
    KDim,
    Koff,
    V2EDim,
    Vertex,
    cartesian_case,
    no_default_backend,
    unstructured_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    Cell,
    fieldview_backend,
    reduction_setup,
    size,
)


def test_copy(cartesian_case):  # noqa: F811 # fixtures
    @field_operator
    def testee(a: cases.IJKField) -> cases.IJKField:
        field_tuple = (a, a)
        field_0 = field_tuple[0]
        field_1 = field_tuple[1]
        return field_0

    cases.verify_with_default_data(cartesian_case, testee, ref=lambda a: a)


def test_multicopy(cartesian_case):  # noqa: F811 # fixtures
    @field_operator
    def testee(a: cases.IJKField, b: cases.IJKField) -> tuple[cases.IJKField, cases.IJKField]:
        return a, b

    cases.verify_with_default_data(cartesian_case, testee, ref=lambda a, b: (a, b))


def test_cartesian_shift(cartesian_case):  # noqa: F811 # fixtures
    @field_operator
    def testee(a: cases.IJKField) -> cases.IJKField:
        return a(Ioff[1])

    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out, ref=a[1:])


def test_unstructured_shift(unstructured_case):  # noqa: F811 # fixtures
    @field_operator
    def testee(a: cases.VField) -> cases.EField:
        return a(E2V[0])

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: a[unstructured_case.offset_provider["E2V"].table[:, 0]],
    )


def test_composed_unstructured_shift(reduction_setup, fieldview_backend):
    E2V = reduction_setup.E2V
    C2E = reduction_setup.C2E
    e2v_table = reduction_setup.offset_provider["E2V"].table[slice(0, None), 0]
    c2e_table = reduction_setup.offset_provider["C2E"].table[slice(0, None), 0]

    a = np_as_located_field(Vertex)(np.arange(0, reduction_setup.num_vertices, dtype=np.float64))
    b = np_as_located_field(Cell)(np.zeros(reduction_setup.num_cells))

    @field_operator(backend=fieldview_backend)
    def composed_shift_unstructured_flat(inp: Field[[Vertex], float64]) -> Field[[Cell], float64]:
        return inp(E2V[0])(C2E[0])

    @field_operator(backend=fieldview_backend)
    def composed_shift_unstructured_intermediate_result(
        inp: Field[[Vertex], float64]
    ) -> Field[[Cell], float64]:
        tmp = inp(E2V[0])
        return tmp(C2E[0])

    @field_operator(backend=fieldview_backend)
    def shift_e2v(inp: Field[[Vertex], float64]) -> Field[[Edge], float64]:
        return inp(E2V[0])

    @field_operator(backend=fieldview_backend)
    def composed_shift_unstructured(inp: Field[[Vertex], float64]) -> Field[[Cell], float64]:
        return shift_e2v(inp)(C2E[0])

    ref = np.asarray(a)[e2v_table][c2e_table]

    for field_op in [
        composed_shift_unstructured_flat,
        composed_shift_unstructured_intermediate_result,
        composed_shift_unstructured,
    ]:
        field_op(a, out=b, offset_provider=reduction_setup.offset_provider)

        assert np.allclose(b, ref)


def test_fold_shifts(cartesian_case):  # noqa: F811 # fixtures
    """Shifting the result of an addition should work."""

    @field_operator
    def testee(a: cases.IJKField, b: cases.IJKField) -> cases.IJKField:
        tmp = a + b(Ioff[1])
        return tmp(Ioff[1])

    a = cases.allocate(cartesian_case, testee, "a").extend({cases.IDim: (0, 1)})()
    b = cases.allocate(cartesian_case, testee, "b").extend({cases.IDim: (0, 2)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, b, out=out, ref=a[1:] + b[2:])


def test_tuples(cartesian_case):  # noqa: F811 # fixtures
    @field_operator
    def testee(a: cases.IJKFloatField, b: cases.IJKFloatField) -> cases.IJKFloatField:
        inps = a, b
        scalars = 1.3, float64(5.0), float64("3.4")
        return (inps[0] * scalars[0] + inps[1] * scalars[1]) * scalars[2]

    cases.verify_with_default_data(
        cartesian_case, testee, ref=lambda a, b: (a * 1.3 + b * 5.0) * 3.4
    )


def test_scalar_arg(unstructured_case):  # noqa: F811 # fixtures
    """Test scalar argument being turned into 0-dim field."""

    @field_operator
    def testee(a: int) -> cases.VField:
        return broadcast(a + 1, (Vertex,))

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: np.full(
            [unstructured_case.default_sizes[Vertex]],
            a + 1,
            dtype=int,
        ),
        comparison=lambda a, b: np.all(a == b),
    )


def test_nested_scalar_arg(unstructured_case):  # noqa: F811 # fixtures
    @field_operator
    def testee_inner(a: int) -> cases.VField:
        return broadcast(a + 1, (Vertex,))

    @field_operator
    def testee(a: int) -> cases.VField:
        return testee_inner(a + 1)

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: np.full([unstructured_case.default_sizes[Vertex]], a + 2, dtype=int),
    )


def test_scalar_arg_with_field(cartesian_case):  # noqa: F811 # fixtures
    @field_operator
    def testee(a: cases.IJKField, b: int) -> cases.IJKField:
        tmp = b * a
        return tmp(Ioff[1])

    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1)})()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    ref = a.array()[1:] * b

    cases.verify(cartesian_case, testee, a, b, out=out, ref=ref)


def test_scalar_in_domain_spec_and_fo_call(cartesian_case):  # noqa: F811 # fixtures
    if cartesian_case.backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.xfail(
            "Scalar arguments not supported to be used in both domain specification "
            "and as an argument to a field operator."
        )

    @field_operator
    def testee_op(size: int) -> cases.IField:
        return broadcast(size, (IDim,))

    @program
    def testee(size: int, out: cases.IField):
        testee_op(size, out=out, domain={IDim: (0, size)})

    size = cartesian_case.default_sizes[IDim]
    out = cases.allocate(cartesian_case, testee, "out").zeros()()

    cases.verify(
        cartesian_case, testee, size, out=out, ref=np.full_like(out.array(), size, dtype=int)
    )


def test_scalar_scan(cartesian_case):  # noqa: F811 # fixtures
    @scan_operator(axis=KDim, forward=True, init=(0.0))
    def testee_scan(state: float, qc_in: float, scalar: float) -> float:
        qc = qc_in + state + scalar
        return qc

    @program
    def testee(qc: Field[[IDim, KDim], float], scalar: float):
        testee_scan(qc, scalar, out=qc)

    qc = cases.allocate(cartesian_case, testee, "qc").zeros()()
    scalar = 1.0
    ksize = cartesian_case.default_sizes[KDim]
    expected = np.full((ksize, ksize), np.arange(start=1, stop=11, step=1).astype(float64))

    cases.verify(cartesian_case, testee, qc, scalar, inout=qc, ref=expected)


def test_tuple_scalar_scan(cartesian_case):  # noqa: F811 # fixtures
    if cartesian_case.backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.xfail("Scalar tuple arguments are not supported in gtfn yet.")

    @scan_operator(axis=KDim, forward=True, init=0.0)
    def testee_scan(
        state: float, qc_in: float, tuple_scalar: tuple[float, tuple[float, float]]
    ) -> float:
        return (qc_in + state + tuple_scalar[1][0] + tuple_scalar[1][1]) / tuple_scalar[0]

    @field_operator
    def testee_op(
        qc: Field[[IDim, KDim], float], tuple_scalar: tuple[float, tuple[float, float]]
    ) -> Field[[IDim, KDim], float]:
        return testee_scan(qc, tuple_scalar)

    qc = cases.allocate(cartesian_case, testee_op, "qc").zeros()()
    tuple_scalar = (1.0, (1.0, 0.0))
    ksize = cartesian_case.default_sizes[KDim]
    expected = np.full((ksize, ksize), np.arange(start=1, stop=11, step=1).astype(float64))

    cases.verify(cartesian_case, testee_op, qc, tuple_scalar, out=qc, ref=expected)


def test_astype_int(cartesian_case):  # noqa: F811 # fixtures
    @field_operator
    def testee(a: Field[[IDim], float64]) -> Field[[IDim], int64]:
        b = astype(a, int64)
        return b

    cases.verify_with_default_data(
        cartesian_case,
        testee,
        ref=lambda a: a.astype(int),
        comparison=lambda a, b: np.all(a == b),
    )


def test_astype_bool(cartesian_case):  # noqa: F811 # fixtures
    @field_operator
    def testee(a: Field[[IDim], float64]) -> Field[[IDim], bool]:
        b = astype(a, bool)
        return b

    cases.verify_with_default_data(
        cartesian_case,
        testee,
        ref=lambda a: a.astype(bool),
        comparison=lambda a, b: np.all(a == b),
    )


def test_astype_float(cartesian_case):  # noqa: F811 # fixtures
    @field_operator
    def testee(a: Field[[IDim], float64]) -> Field[[IDim], np.float32]:
        b = astype(a, float32)
        return b

    cases.verify_with_default_data(
        cartesian_case,
        testee,
        ref=lambda a: a.astype(np.float32),
        comparison=lambda a, b: np.all(a == b),
    )


def test_offset_field(fieldview_backend):
    a_I_arr = np.random.randn(size, size).astype("float64")
    a_I_float = np_as_located_field(IDim, KDim)(a_I_arr)
    a_I_float_1 = np_as_located_field(IDim, KDim)(
        np.append(np.insert(a_I_arr, size, 0, axis=1), [np.array([0] * (size + 1))], axis=0)
    )
    offset_field_arr = np.ones((size - 1, size - 1), dtype=int64)
    offset_field_comp = np.append(
        np.insert(offset_field_arr, size - 1, 0, axis=1), [np.array([0] * size)], axis=0
    )
    offset_field = np_as_located_field(IDim, KDim)(offset_field_comp)
    out_I_float = np_as_located_field(IDim, KDim)(np.zeros((size, size), dtype=float64))
    out_I_float_1 = np_as_located_field(IDim, KDim)(np.zeros((size, size), dtype=float64))

    @field_operator(backend=fieldview_backend)
    def offset_index_field_fo(
        a: Field[[IDim, KDim], float64],
        offset_field: Field[[IDim, KDim], int64],
    ) -> Field[[IDim, KDim], float64]:
        a_i = a(as_offset(Ioff, offset_field))
        a_i_k = a_i(as_offset(Koff, offset_field))
        return a_i_k

    offset_index_field_fo(
        a_I_float,
        offset_field,
        out=out_I_float,
        offset_provider={"Ioff": IDim, "Koff": KDim},
    )

    @field_operator(backend=fieldview_backend)
    def offset_index_int_fo(a: Field[[IDim, KDim], float64]) -> Field[[IDim, KDim], float64]:
        a_i = a(Ioff[1])
        a_i_k = a_i(Koff[1])
        return a_i_k

    offset_index_int_fo(
        a_I_float_1, out=out_I_float_1, offset_provider={"Ioff": IDim, "Koff": KDim}
    )
    assert np.allclose(
        out_I_float.array()[: size - 1, : size - 1], out_I_float_1.array()[: size - 1, : size - 1]
    )


def test_nested_tuple_return(fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.zeros((size,), dtype=float64))

    @field_operator(backend=fieldview_backend)
    def pack_tuple(
        a: Field[[IDim], float64], b: Field[[IDim], float64]
    ) -> tuple[Field[[IDim], float64], tuple[Field[[IDim], float64], Field[[IDim], float64]]]:
        return (a, (a, b))

    @field_operator(backend=fieldview_backend)
    def combine(a: Field[[IDim], float64], b: Field[[IDim], float64]) -> Field[[IDim], float64]:
        packed = pack_tuple(a, b)
        return packed[0] + packed[1][0] + packed[1][1]

    combine(a_I_float, b_I_float, out=out_I_float, offset_provider={})

    assert np.allclose(2 * a_I_float.array() + b_I_float.array(), out_I_float)


def test_nested_reduction(reduction_setup, fieldview_backend):
    rs = reduction_setup
    V2EDim = rs.V2EDim
    E2VDim = rs.E2VDim
    V2E = rs.V2E
    E2V = rs.E2V

    out = np_as_located_field(Edge)(np.zeros([rs.num_edges], dtype=np.int64))

    @field_operator(backend=fieldview_backend)
    def testee(inp: Field[[Edge], int64]) -> Field[[Edge], int64]:
        tmp = neighbor_sum(inp(V2E), axis=V2EDim)
        return neighbor_sum(tmp(E2V), axis=E2VDim)

    testee(rs.inp, out=out, offset_provider=rs.offset_provider)

    expected = np.sum(np.sum(rs.inp[rs.v2e_table], axis=1)[rs.e2v_table], axis=1)
    assert np.allclose(out, expected)


@pytest.mark.xfail(reason="Not yet supported in lowering, requires `map_`ing of inner reduce op.")
def test_nested_reduction_shift_first(reduction_setup, fieldview_backend):
    rs = reduction_setup
    V2EDim = rs.V2EDim
    E2VDim = rs.E2VDim
    V2E = rs.V2E
    E2V = rs.E2V

    out = np_as_located_field(Edge)(np.zeros([rs.num_edges], dtype=np.int64))

    @field_operator(backend=fieldview_backend)
    def testee(inp: Field[[Edge], int64]) -> Field[[Edge], int64]:
        tmp = inp(V2E)
        tmp2 = tmp(E2V)
        return neighbor_sum(neighbor_sum(tmp2, axis=V2EDim), axis=E2VDim)

    testee(rs.inp, out=out, offset_provider=rs.offset_provider)

    expected = np.sum(np.sum(rs.inp[rs.v2e_table], axis=1)[rs.e2v_table], axis=1)
    assert np.allclose(out, expected)


def test_tuple_return_2(reduction_setup, fieldview_backend):
    rs = reduction_setup
    V2EDim = rs.V2EDim
    V2E = rs.V2E

    @field_operator(backend=fieldview_backend)
    def reduction_tuple(
        a: Field[[Edge], int64], b: Field[[Edge], int64]
    ) -> tuple[Field[[Vertex], int64], Field[[Vertex], int64]]:
        a = neighbor_sum(a(V2E), axis=V2EDim)
        b = neighbor_sum(b(V2E), axis=V2EDim)
        return a, b

    @field_operator(backend=fieldview_backend)
    def combine_tuple(a: Field[[Edge], int64], b: Field[[Edge], int64]) -> Field[[Vertex], int64]:
        packed = reduction_tuple(a, b)
        return packed[0] + packed[1]

    combine_tuple(rs.inp, rs.inp, out=rs.out, offset_provider=rs.offset_provider)

    ref = np.sum(rs.v2e_table, axis=1) * 2
    assert np.allclose(ref, rs.out)


def test_tuple_with_local_field_in_reduction_shifted(reduction_setup, fieldview_backend):
    rs = reduction_setup
    V2EDim = rs.V2EDim
    V2E = rs.V2E
    E2V = rs.E2V

    num_vertices = rs.num_vertices
    num_edges = rs.num_edges

    # TODO(tehrengruber): use different values per location
    a = np_as_located_field(Edge)(np.ones((num_edges,)))
    b = np_as_located_field(Vertex)(2 * np.ones((num_vertices,)))
    out = np_as_located_field(Edge)(np.zeros((num_edges,)))

    @field_operator(backend=fieldview_backend)
    def reduce_tuple_element(
        edge_field: Field[[Edge], float64], vertex_field: Field[[Vertex], float64]
    ) -> Field[[Edge], float64]:
        tup = edge_field(V2E), vertex_field
        red = neighbor_sum(tup[0] + vertex_field, axis=V2EDim)
        return red(E2V[0])

    reduce_tuple_element(a, b, out=out, offset_provider=rs.offset_provider)

    # conn table used is inverted here on purpose
    red = np.sum(np.asarray(a)[rs.v2e_table] + np.asarray(b)[:, np.newaxis], axis=1)
    expected = red[rs.e2v_table][:, 0]

    assert np.allclose(expected, out)


def test_tuple_arg(fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.zeros((size,), dtype=float64))

    @field_operator(backend=fieldview_backend)
    def unpack_tuple(
        inp: tuple[tuple[Field[[IDim], float64], Field[[IDim], float64]], Field[[IDim], float64]]
    ) -> Field[[IDim], float64]:
        return 3.0 * inp[0][0] + inp[0][1] + inp[1]

    unpack_tuple(((a_I_float, b_I_float), a_I_float), out=out_I_float, offset_provider={})

    assert np.allclose(3 * a_I_float.array() + b_I_float.array() + a_I_float.array(), out_I_float)


@pytest.mark.parametrize("forward", [True, False])
def test_fieldop_from_scan(fieldview_backend, forward):
    init = 1.0
    out = np_as_located_field(KDim)(np.zeros((size,)))
    expected = np.arange(init + 1.0, init + 1.0 + size, 1)
    if not forward:
        expected = np.flip(expected)

    @field_operator(backend=fieldview_backend)
    def add(carry: float, foo: float) -> float:
        return carry + foo

    @scan_operator(axis=KDim, forward=forward, init=init, backend=fieldview_backend)
    def simple_scan_operator(carry: float) -> float:
        return add(carry, 1.0)

    simple_scan_operator(out=out, offset_provider={})

    assert np.allclose(expected, out)


def test_solve_triag(fieldview_backend):
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.xfail("Transformation passes fail in putting `scan` to the top.")
    shape = (3, 7, 5)
    rng = np.random.default_rng()
    a_np, b_np, c_np, d_np = (rng.normal(size=shape) for _ in range(4))
    b_np *= 2
    a, b, c, d = (
        np_as_located_field(IDim, JDim, KDim)(np_arr) for np_arr in [a_np, b_np, c_np, d_np]
    )
    out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))

    # compute reference
    matrices = np.zeros(shape + shape[-1:])
    i = np.arange(shape[2])
    matrices[:, :, i[1:], i[:-1]] = a_np[:, :, 1:]
    matrices[:, :, i, i] = b_np
    matrices[:, :, i[:-1], i[1:]] = c_np[:, :, :-1]
    expected = np.linalg.solve(matrices, d_np)

    @scan_operator(axis=KDim, forward=True, init=(0.0, 0.0))
    def tridiag_forward(
        state: tuple[float, float], a: float, b: float, c: float, d: float
    ) -> tuple[float, float]:
        return (c / (b - a * state[0]), (d - a * state[1]) / (b - a * state[0]))

    @scan_operator(axis=KDim, forward=False, init=0.0)
    def tridiag_backward(x_kp1: float, cp: float, dp: float) -> float:
        return dp - cp * x_kp1

    @field_operator(backend=fieldview_backend)
    def solve_tridiag(
        a: Field[[IDim, JDim, KDim], float],
        b: Field[[IDim, JDim, KDim], float],
        c: Field[[IDim, JDim, KDim], float],
        d: Field[[IDim, JDim, KDim], float],
    ) -> Field[[IDim, JDim, KDim], float]:
        cp, dp = tridiag_forward(a, b, c, d)
        return tridiag_backward(cp, dp)

    solve_tridiag(a, b, c, d, out=out, offset_provider={})

    assert np.allclose(expected, out)


@pytest.mark.parametrize("left,right", [(2.0, 3.0), (3.0, 2.0)])
def test_ternary_operator(left, right, fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.zeros((size,), dtype=float64))

    @field_operator(backend=fieldview_backend)
    def ternary_field_op(
        a: Field[[IDim], float], b: Field[[IDim], float], left: float, right: float
    ) -> Field[[IDim], float]:
        return a if left < right else b

    ternary_field_op(a_I_float, b_I_float, left, right, out=out_I_float, offset_provider={})
    e = np.asarray(a_I_float) if left < right else np.asarray(b_I_float)
    assert np.allclose(e, out_I_float)

    @field_operator(backend=fieldview_backend)
    def ternary_field_op_scalars(left: float, right: float) -> Field[[IDim], float]:
        return broadcast(3.0, (IDim,)) if left > right else broadcast(4.0, (IDim,))

    ternary_field_op_scalars(left, right, out=out_I_float, offset_provider={})
    e = np.full(e.shape, 3.0) if left > right else np.full(e.shape, 4.0)
    assert np.allclose(e, out_I_float)


@pytest.mark.parametrize("left,right", [(2.0, 3.0), (3.0, 2.0)])
def test_ternary_operator_tuple(left, right, fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.zeros((size,), dtype=float64))
    out_I_float_1 = np_as_located_field(IDim)(np.zeros((size,), dtype=float64))

    @field_operator(backend=fieldview_backend)
    def ternary_field_op(
        a: Field[[IDim], float], b: Field[[IDim], float], left: float, right: float
    ) -> tuple[Field[[IDim], float], Field[[IDim], float]]:
        return (a, b) if left < right else (b, a)

    ternary_field_op(
        a_I_float, b_I_float, left, right, out=(out_I_float, out_I_float_1), offset_provider={}
    )

    e, f = (
        (np.asarray(a_I_float), np.asarray(b_I_float))
        if left < right
        else (np.asarray(b_I_float), np.asarray(a_I_float))
    )
    assert np.allclose(e, out_I_float)
    assert np.allclose(f, out_I_float_1)


def test_ternary_builtin_neighbor_sum(reduction_setup, fieldview_backend):
    rs = reduction_setup
    V2EDim = rs.V2EDim
    V2E = rs.V2E

    num_vertices = rs.num_vertices
    num_edges = rs.num_edges

    a = np_as_located_field(Edge)(np.ones((num_edges,)))
    b = np_as_located_field(Edge)(2 * np.ones((num_edges,)))
    out = np_as_located_field(Vertex)(np.zeros((num_vertices,)))

    @field_operator(backend=fieldview_backend)
    def ternary_reduce(a: Field[[Edge], float], b: Field[[Edge], float]) -> Field[[Vertex], float]:
        out = neighbor_sum(b(V2E) if 2 < 3 else a(V2E), axis=V2EDim)
        return out

    ternary_reduce(a, b, out=out, offset_provider=rs.offset_provider)

    expected = (
        np.sum(np.asarray(b)[rs.v2e_table], axis=1)
        if 2 < 3
        else np.sum(np.asarray(a)[rs.v2e_table], axis=1)
    )

    assert np.allclose(expected, out)


def test_ternary_scan(fieldview_backend):
    init = 0.0
    a_float = 4
    a = np_as_located_field(KDim)(a_float * np.ones((size,)))
    out = np_as_located_field(KDim)(np.zeros((size,)))
    expected = np.asarray([i if i <= a_float else a_float + 1 for i in range(1, size + 1)])

    @scan_operator(axis=KDim, forward=True, init=init, backend=fieldview_backend)
    def simple_scan_operator(carry: float, a: float) -> float:
        return carry if carry > a else carry + 1.0

    simple_scan_operator(a, out=out, offset_provider={})

    assert np.allclose(expected, out)


@pytest.mark.parametrize("forward", [True, False])
def test_scan_nested_tuple_output(fieldview_backend, forward):
    init = (1.0, (2.0, 3.0))
    out1, out2, out3 = (np_as_located_field(KDim)(np.zeros((size,))) for _ in range(3))
    expected = np.arange(1.0, 1.0 + size, 1)
    if not forward:
        expected = np.flip(expected)

    @scan_operator(axis=KDim, forward=forward, init=init, backend=fieldview_backend)
    def simple_scan_operator(
        carry: tuple[float, tuple[float, float]]
    ) -> tuple[float, tuple[float, float]]:
        return (carry[0] + 1.0, (carry[1][0] + 1.0, carry[1][1] + 1.0))

    simple_scan_operator(out=(out1, (out2, out3)), offset_provider={})

    assert np.allclose(expected + 1.0, out1)
    assert np.allclose(expected + 2.0, out2)
    assert np.allclose(expected + 3.0, out3)


@pytest.mark.parametrize("forward", [True, False])
def test_scan_nested_tuple_input(fieldview_backend, forward):
    init = 1.0
    inp1 = np_as_located_field(KDim)(np.ones((size,)))
    inp2 = np_as_located_field(KDim)(np.arange(0.0, size, 1))
    out = np_as_located_field(KDim)(np.zeros((size,)))

    prev_levels_iterator = lambda i: range(i + 1) if forward else range(size - 1, i - 1, -1)
    expected = np.asarray(
        [
            reduce(lambda prev, i: prev + inp1[i] + inp2[i], prev_levels_iterator(i), init)
            for i in range(size)
        ]
    )

    @scan_operator(axis=KDim, forward=forward, init=init, backend=fieldview_backend)
    def simple_scan_operator(carry: float, a: tuple[float, float]) -> float:
        return carry + a[0] + a[1]

    simple_scan_operator((inp1, inp2), out=out, offset_provider={})

    assert np.allclose(expected, out)


def test_docstring():
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))

    @field_operator
    def fieldop_with_docstring(a: Field[[IDim], float64]) -> Field[[IDim], float64]:
        """My docstring."""
        return a

    @program
    def test_docstring(a: Field[[IDim], float64]):
        """My docstring."""
        fieldop_with_docstring(a, out=a)

    test_docstring(a_I_float, offset_provider={})


def test_domain(fieldview_backend):
    inp = np_as_located_field(IDim, JDim)(np.ones((size, size), dtype=float64))
    out = np_as_located_field(IDim, JDim)(2 * np.ones((size, size), dtype=float64))

    expected = np.array(out)
    expected[1:9, 4:6] = 1 + 1

    @field_operator(backend=fieldview_backend)
    def fieldop_domain(a: Field[[IDim, JDim], float64]) -> Field[[IDim, JDim], float64]:
        return a + a

    @program(backend=fieldview_backend)
    def program_domain(inp: Field[[IDim, JDim], float64], out: Field[[IDim, JDim], float64]):
        fieldop_domain(inp, out=out, domain={IDim: (minimum(1, 2), 9), JDim: (4, maximum(5, 6))})

    program_domain(inp, out, offset_provider={})

    assert np.allclose(expected, out)


def test_domain_input_bounds(fieldview_backend):
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.xfail("FloorDiv not fully supported in gtfn.")
    inp = np_as_located_field(IDim, JDim)(np.ones((size, size), dtype=float64))
    out = np_as_located_field(IDim, JDim)(2 * np.ones((size, size), dtype=float64))

    lower_i = 1
    upper_i = 9
    lower_j = 4
    upper_j = 6

    expected = np.array(out)
    expected[lower_i:upper_i, lower_j:upper_j] = 1 + 1

    @field_operator(backend=fieldview_backend)
    def fieldop_domain(a: Field[[IDim, JDim], float64]) -> Field[[IDim, JDim], float64]:
        return a + a

    @program(backend=fieldview_backend)
    def program_domain(
        inp: Field[[IDim, JDim], float64],
        out: Field[[IDim, JDim], float64],
        lower_i: int64,
        upper_i: int64,
        lower_j: int64,
        upper_j: int64,
    ):
        fieldop_domain(
            inp,
            out=out,
            domain={IDim: (lower_i, upper_i // 1), JDim: (lower_j**1, upper_j)},
        )

    program_domain(inp, out, lower_i, upper_i, lower_j, upper_j, offset_provider={})

    assert np.allclose(expected, out)


def test_domain_input_bounds_1(fieldview_backend):
    a_IJ_float = np_as_located_field(IDim, JDim)(np.ones((size, size), dtype=float64))

    lower_i = 1
    upper_i = 9
    lower_j = 4
    upper_j = 6

    expected = np.array(a_IJ_float)
    expected[lower_i:upper_i, lower_j:upper_j] = 1 + 1

    @field_operator(backend=fieldview_backend)
    def fieldop_domain(a: Field[[IDim, JDim], float64]) -> Field[[IDim, JDim], float64]:
        return a + a

    @program(backend=fieldview_backend)
    def program_domain(
        a: Field[[IDim, JDim], float64],
        lower_i: int64,
        upper_i: int64,
        lower_j: int64,
        upper_j: int64,
    ):
        fieldop_domain(
            a,
            out=a,
            domain={IDim: (1 * lower_i, upper_i + 0), JDim: (lower_j - 0, upper_j)},
        )

    program_domain(a_IJ_float, lower_i, upper_i, lower_j, upper_j, offset_provider={})

    assert np.allclose(expected, a_IJ_float)


def test_domain_tuple(fieldview_backend):
    inp0 = np_as_located_field(IDim, JDim)(np.ones((size, size), dtype=float64))
    inp1 = np_as_located_field(IDim, JDim)(2 * np.ones((size, size), dtype=float64))
    out0 = np_as_located_field(IDim, JDim)(3 * np.ones((size, size), dtype=float64))
    out1 = np_as_located_field(IDim, JDim)(4 * np.ones((size, size), dtype=float64))

    expected0 = np.array(out0)
    expected0[1:9, 4:6] = (np.asarray(inp0) + np.asarray(inp1))[1:9, 4:6]
    expected1 = np.array(out1)
    expected1[1:9, 4:6] = np.asarray(inp1)[1:9, 4:6]

    @field_operator(backend=fieldview_backend)
    def fieldop_domain_tuple(
        a: Field[[IDim, JDim], float64], b: Field[[IDim, JDim], float64]
    ) -> tuple[Field[[IDim, JDim], float64], Field[[IDim, JDim], float64]]:
        return (a + b, b)

    @program(backend=fieldview_backend)
    def program_domain_tuple(
        inp0: Field[[IDim, JDim], float64],
        inp1: Field[[IDim, JDim], float64],
        out0: Field[[IDim, JDim], float64],
        out1: Field[[IDim, JDim], float64],
    ):
        fieldop_domain_tuple(inp0, inp1, out=(out0, out1), domain={IDim: (1, 9), JDim: (4, 6)})

    program_domain_tuple(inp0, inp1, out0, out1, offset_provider={})

    assert np.allclose(expected0, out0)
    assert np.allclose(expected1, out1)


def test_where_k_offset(fieldview_backend):
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.xfail("IndexFields are not supported yet.")
    a = np_as_located_field(IDim, KDim)(np.ones((size, size)))
    out = np_as_located_field(IDim, KDim)(np.zeros((size, size)))
    k_index = index_field(KDim)

    @field_operator(backend=fieldview_backend)
    def fieldop_where_k_offset(
        a: Field[[IDim, KDim], float64],
        k_index: Field[[KDim], int64],
    ) -> Field[[IDim, KDim], float64]:
        return where(k_index > 0, a(Koff[-1]), 2.0)

    fieldop_where_k_offset(a, k_index, out=out, offset_provider={"Koff": KDim})

    expected = np.where(np.arange(0, size, 1)[np.newaxis, :] > 0.0, a, 2.0)

    assert np.allclose(np.asarray(out), expected)


def test_undefined_symbols():
    with pytest.raises(FieldOperatorTypeDeductionError, match="Undeclared symbol"):

        @field_operator
        def return_undefined():
            return undefined_symbol


def test_zero_dims_fields(fieldview_backend):
    inp = np_as_located_field()(np.array(1.0))
    out = np_as_located_field()(np.array(0.0))

    @field_operator(backend=fieldview_backend)
    def implicit_broadcast_scalar(inp: Field[[], float]):
        return inp

    implicit_broadcast_scalar(inp, out=out, offset_provider={})
    assert np.allclose(out, np.array(1.0))


def test_implicit_broadcast_mixed_dims(fieldview_backend):
    input1 = np_as_located_field(IDim)(np.ones((10,)))
    inp = np_as_located_field()(np.array(1.0))
    out = np_as_located_field(IDim)(np.ones((10,)))

    @field_operator(backend=fieldview_backend)
    def fieldop_implicit_broadcast(
        zero_dim_inp: Field[[], float], inp: Field[[IDim], float], scalar: float
    ) -> Field[[IDim], float]:
        return inp + zero_dim_inp * scalar

    @field_operator(backend=fieldview_backend)
    def fieldop_implicit_broadcast_2(inp: Field[[IDim], float]) -> Field[[IDim], float]:
        fi = fieldop_implicit_broadcast(1.0, inp, 1.0)
        return fi

    fieldop_implicit_broadcast_2(input1, out=out, offset_provider={})
    assert np.allclose(out, np.asarray(inp) * 2)


def test_tuple_unpacking(fieldview_backend):
    size = 10
    inp = np_as_located_field(IDim)(np.ones((size,)))
    out1 = np_as_located_field(IDim)(np.ones((size,)))
    out2 = np_as_located_field(IDim)(np.ones((size,)))
    out3 = np_as_located_field(IDim)(np.ones((size,)))
    out4 = np_as_located_field(IDim)(np.ones((size,)))

    @field_operator(backend=fieldview_backend)
    def unpack(
        inp: Field[[IDim], float64],
    ) -> tuple[
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
    ]:
        a, b, c, d = (inp + 2.0, inp + 3.0, inp + 5.0, inp + 7.0)
        return a, b, c, d

    unpack(inp, out=(out1, out2, out3, out4), offset_provider={})

    arr = inp.array()

    assert np.allclose(out1, arr + 2.0)
    assert np.allclose(out2, arr + 3.0)
    assert np.allclose(out3, arr + 5.0)
    assert np.allclose(out4, arr + 7.0)


def test_tuple_unpacking_star_multi(fieldview_backend):
    size = 10
    inp = np_as_located_field(IDim)(np.ones((size,)))
    out = tuple(np_as_located_field(IDim)(np.ones((size,)) * i) for i in range(3 * 4))

    OutType = tuple[
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
    ]

    @field_operator(backend=fieldview_backend)
    def unpack(
        inp: Field[[IDim], float64],
    ) -> OutType:
        *a, a2, a3 = (inp, inp + 1.0, inp + 2.0, inp + 3.0)
        b1, *b, b3 = (inp + 4.0, inp + 5.0, inp + 6.0, inp + 7.0)
        c1, c2, *c = (inp + 8.0, inp + 9.0, inp + 10.0, inp + 11.0)

        return (a[0], a[1], a2, a3, b1, b[0], b[1], b3, c1, c2, c[0], c[1])

    unpack(inp, out=out, offset_provider={})

    for i in range(3 * 4):
        assert np.allclose(out[i], inp.array() + i)


def test_tuple_unpacking_too_many_values(fieldview_backend):
    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=(r"Could not deduce type: Too many values to unpack \(expected 3\)"),
    ):

        @field_operator(backend=fieldview_backend)
        def _star_unpack() -> tuple[int, float64, int]:
            a, b, c = (1, 2.0, 3, 4, 5, 6, 7.0)
            return a, b, c


def test_tuple_unpacking_too_many_values(fieldview_backend):
    with pytest.raises(
        FieldOperatorTypeDeductionError, match=(r"Assignment value must be of type tuple!")
    ):

        @field_operator(backend=fieldview_backend)
        def _invalid_unpack() -> tuple[int, float64, int]:
            a, b, c = 1
            return a


def test_constant_closure_vars(fieldview_backend):
    from gt4py.eve.utils import FrozenNamespace

    constants = FrozenNamespace(
        PI=np.float32(3.142),
        E=np.float32(2.718),
    )

    @field_operator(backend=fieldview_backend)
    def consume_constants(input: Field[[IDim], np.float32]) -> Field[[IDim], np.float32]:
        return constants.PI * constants.E * input

    input = np_as_located_field(IDim)(np.ones((1,), dtype=np.float32))
    output = np_as_located_field(IDim)(np.zeros((1,), dtype=np.float32))
    consume_constants(input, out=output, offset_provider={})
    assert np.allclose(np.asarray(output), constants.PI * constants.E)
