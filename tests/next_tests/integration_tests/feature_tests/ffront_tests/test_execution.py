# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import math
from functools import reduce
from typing import TypeAlias

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import (
    astype,
    broadcast,
    common,
    errors,
    float32,
    float64,
    int32,
    int64,
    minimum,
    neighbor_sum,
    utils as gt_utils,
)
from gt4py.next.ffront.experimental import as_offset

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    C2E,
    E2V,
    V2E,
    E2VDim,
    Edge,
    IDim,
    Ioff,
    JDim,
    KDim,
    Koff,
    V2EDim,
    Vertex,
    cartesian_case,
    unstructured_case,
    unstructured_case_3d,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)


def test_copy(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKField) -> cases.IJKField:
        field_tuple = (a, a)
        field_0 = field_tuple[0]
        field_1 = field_tuple[1]
        return field_0

    cases.verify_with_default_data(cartesian_case, testee, ref=lambda a: a)


@pytest.mark.uses_tuple_returns
def test_multicopy(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKField, b: cases.IJKField) -> tuple[cases.IJKField, cases.IJKField]:
        return a, b

    cases.verify_with_default_data(cartesian_case, testee, ref=lambda a, b: (a, b))


def test_infinity(cartesian_case):
    # TODO(tehrengruber): We actually want a GTIR test with a `nan` literal. This would then
    #  also not raise a ZeroDivisionError error in embedded and roundtrip.
    @gtx.field_operator
    def testee() -> cases.IFloatField:
        return broadcast(1.0 / 0.0, (IDim,))

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    try:
        cases.verify(
            cartesian_case,
            testee,
            out=out,
            comparison=np.array_equal,
            ref=np.full(out.ndarray.shape, math.inf),
        )
    except ZeroDivisionError:
        pass


def test_nan(cartesian_case):
    # TODO(tehrengruber): We actually want a GTIR test with a `nan` literal. This would then
    #  also not raise a ZeroDivisionError error in embedded and roundtrip.
    @gtx.field_operator
    def testee() -> cases.IFloatField:
        return broadcast(0.0 / 0.0, (IDim,))

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    try:
        cases.verify(
            cartesian_case,
            testee,
            out=out,
            comparison=functools.partial(np.array_equal, equal_nan=True),
            ref=np.full(out.ndarray.shape, math.nan),
        )
    except ZeroDivisionError:
        pass


@pytest.mark.uses_cartesian_shift
def test_cartesian_shift(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKField) -> cases.IJKField:
        return a(Ioff[1])

    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out, ref=a[1:])


@pytest.mark.uses_unstructured_shift
def test_unstructured_shift(unstructured_case):
    @gtx.field_operator
    def testee(a: cases.VField) -> cases.EField:
        return a(E2V[0])

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: a[unstructured_case.offset_provider["E2V"].asnumpy()[:, 0]],
    )


def test_horizontal_only_with_3d_mesh(unstructured_case_3d):
    # test field operator operating only on horizontal fields while using an offset provider
    # including a vertical dimension.
    @gtx.field_operator
    def testee(a: cases.VField) -> cases.VField:
        return a

    cases.verify_with_default_data(
        unstructured_case_3d,
        testee,
        ref=lambda a: a,
    )


@pytest.mark.uses_unstructured_shift
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
        ref=lambda inp: inp[unstructured_case.offset_provider["E2V"].asnumpy()[:, 0]][
            unstructured_case.offset_provider["C2E"].asnumpy()[:, 0]
        ],
    )

    cases.verify_with_default_data(
        unstructured_case,
        composed_shift_unstructured_intermediate_result,
        ref=lambda inp: inp[unstructured_case.offset_provider["E2V"].asnumpy()[:, 0]][
            unstructured_case.offset_provider["C2E"].asnumpy()[:, 0]
        ],
        comparison=lambda inp, tmp: np.all(inp == tmp),
    )

    cases.verify_with_default_data(
        unstructured_case,
        composed_shift_unstructured,
        ref=lambda inp: inp[unstructured_case.offset_provider["E2V"].asnumpy()[:, 0]][
            unstructured_case.offset_provider["C2E"].asnumpy()[:, 0]
        ],
    )


@pytest.mark.uses_cartesian_shift
def test_fold_shifts(cartesian_case):
    """Shifting the result of an addition should work."""

    @gtx.field_operator
    def testee(a: cases.IJKField, b: cases.IJKField) -> cases.IJKField:
        tmp = a + b(Ioff[1])
        return tmp(Ioff[1])

    a = cases.allocate(cartesian_case, testee, "a").extend({cases.IDim: (0, 1)})()
    b = cases.allocate(cartesian_case, testee, "b").extend({cases.IDim: (0, 2)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, b, out=out, ref=a.ndarray[1:] + b.ndarray[2:])


def test_tuples(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKFloatField, b: cases.IJKFloatField) -> cases.IJKFloatField:
        inps = a, b
        scalars = 1.3, float64(5.0), float64("3.4")
        return (inps[0] * scalars[0] + inps[1] * scalars[1]) * scalars[2]

    cases.verify_with_default_data(
        cartesian_case, testee, ref=lambda a, b: (a * 1.3 + b * 5.0) * 3.4
    )


def test_scalar_arg(unstructured_case):
    """Test scalar argument being turned into 0-dim field."""

    @gtx.field_operator
    def testee(a: int32) -> cases.VField:
        return broadcast(a + 1, (Vertex,))

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: np.full([unstructured_case.default_sizes[Vertex]], a + 1, dtype=int32),
        comparison=lambda a, b: np.all(a == b),
    )


def test_np_bool_scalar_arg(unstructured_case):
    """Test scalar argument being turned into 0-dim field."""

    @gtx.field_operator
    def testee(a: gtx.bool) -> cases.VBoolField:
        return broadcast(not a, (Vertex,))

    a = np.bool_(True)  # explicitly using a np.bool

    ref = np.full([unstructured_case.default_sizes[Vertex]], not a, dtype=np.bool_)
    out = cases.allocate(unstructured_case, testee, cases.RETURN)()

    cases.verify(unstructured_case, testee, a, out=out, ref=ref)


def test_nested_scalar_arg(unstructured_case):
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


@pytest.mark.uses_tuple_args
def test_scalar_tuple_arg(unstructured_case):
    @gtx.field_operator
    def testee(a: tuple[int32, tuple[int32, int32]]) -> cases.VField:
        return broadcast(a[0] + 2 * a[1][0] + 3 * a[1][1], (Vertex,))

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: np.full(
            [unstructured_case.default_sizes[Vertex]], a[0] + 2 * a[1][0] + 3 * a[1][1], dtype=int32
        ),
    )


@pytest.mark.uses_tuple_args
@pytest.mark.uses_zero_dimensional_fields
def test_zero_dim_tuple_arg(unstructured_case):
    @gtx.field_operator
    def testee(
        a: tuple[gtx.Field[[], int32], tuple[gtx.Field[[], int32], gtx.Field[[], int32]]],
    ) -> cases.VField:
        return broadcast(a[0] + 2 * a[1][0] + 3 * a[1][1], (Vertex,))

    def ref(a):
        a = gt_utils.tree_map(lambda x: x[()])(a)  # unwrap 0d field
        return np.full(
            [unstructured_case.default_sizes[Vertex]], a[0] + 2 * a[1][0] + 3 * a[1][1], dtype=int32
        )

    cases.verify_with_default_data(unstructured_case, testee, ref=ref)


@pytest.mark.uses_tuple_args
def test_mixed_field_scalar_tuple_arg(cartesian_case):
    @gtx.field_operator
    def testee(a: tuple[int32, tuple[int32, cases.IField, int32]]) -> cases.IField:
        return a[0] + 2 * a[1][0] + 3 * a[1][1] + 5 * a[1][2]

    cases.verify_with_default_data(
        cartesian_case,
        testee,
        ref=lambda a: np.full(
            [cartesian_case.default_sizes[IDim]], a[0] + 2 * a[1][0] + 5 * a[1][2], dtype=int32
        )
        + 3 * a[1][1],
    )


@pytest.mark.uses_tuple_args
@pytest.mark.uses_tuple_args_with_different_but_promotable_dims
def test_tuple_arg_with_different_but_promotable_dims(cartesian_case):
    @gtx.field_operator
    def testee(a: tuple[cases.IField, cases.IJField]) -> cases.IJField:
        return a[0] + 2 * a[1]

    cases.verify_with_default_data(
        cartesian_case,
        testee,
        ref=lambda a: a[0][:, np.newaxis] + 2 * a[1],
    )


@pytest.mark.uses_tuple_args
@pytest.mark.xfail(reason="Iterator of tuple approach in lowering does not allow this.")
def test_tuple_arg_with_unpromotable_dims(unstructured_case):
    @gtx.field_operator
    def testee(a: tuple[cases.VField, cases.EField]) -> cases.VField:
        return a[0] + 2 * a[1](V2E[0])

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: a[0][:, np.newaxis] + 2 * a[1],
    )


@pytest.mark.uses_cartesian_shift
def test_scalar_arg_with_field(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKField, b: int32) -> cases.IJKField:
        tmp = b * a
        return tmp(Ioff[1])

    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1)})()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    ref = a[1:] * b

    cases.verify(cartesian_case, testee, a, b, out=out, ref=ref)


@pytest.mark.uses_tuple_args
def test_double_use_scalar(cartesian_case):
    # TODO(tehrengruber): This should be a regression test on ITIR level, but tracing doesn't
    #  work for this case.
    @gtx.field_operator
    def testee(a: int32, b: int32, c: cases.IField) -> cases.IField:
        tmp = a * b
        tmp2 = tmp * tmp
        # important part here is that we use the intermediate twice so that it is
        # not inlined
        return tmp2 * tmp2 * c

    cases.verify_with_default_data(
        cartesian_case, testee, ref=lambda a, b, c: a * b * a * b * a * b * a * b * c
    )


@pytest.mark.uses_scalar_in_domain_and_fo
def test_scalar_in_domain_spec_and_fo_call(cartesian_case):
    @gtx.field_operator
    def testee_op(size: gtx.IndexType) -> gtx.Field[[IDim], gtx.IndexType]:
        return broadcast(size, (IDim,))

    @gtx.program
    def testee(size: gtx.IndexType, out: gtx.Field[[IDim], gtx.IndexType]):
        testee_op(size, out=out, domain={IDim: (0, size)})

    size = cartesian_case.default_sizes[IDim]
    out = cases.allocate(cartesian_case, testee, "out").zeros()()

    cases.verify(
        cartesian_case, testee, size, out=out, ref=np.full_like(out, size, dtype=gtx.IndexType)
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
@pytest.mark.uses_tuple_iterator
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
@pytest.mark.uses_index_fields
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


def test_single_value_field(cartesian_case):
    @gtx.field_operator
    def testee_fo(a: cases.IKField) -> cases.IKField:
        return a

    @gtx.program
    def testee_prog(a: cases.IKField):
        testee_fo(a, out=a[1:2, 3:4])

    a = cases.allocate(cartesian_case, testee_prog, "a")()
    ref = a[1, 3]

    cases.verify(cartesian_case, testee_prog, a, inout=a[1, 3], ref=ref)


def test_astype_int(cartesian_case):
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


def test_astype_int_local_field(unstructured_case):
    @gtx.field_operator
    def testee(a: gtx.Field[[Vertex], np.float64]) -> gtx.Field[[Edge], int64]:
        tmp = astype(a(E2V), int64)
        return neighbor_sum(tmp, axis=E2VDim)

    e2v_table = unstructured_case.offset_provider["E2V"].asnumpy()

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: np.sum(a.astype(int64)[e2v_table], axis=1, initial=0),
        comparison=lambda a, b: np.all(a == b),
    )


@pytest.mark.uses_tuple_returns
def test_astype_on_tuples(cartesian_case):
    @gtx.field_operator
    def field_op_returning_a_tuple(
        a: cases.IFloatField, b: cases.IFloatField
    ) -> tuple[gtx.Field[[IDim], float], gtx.Field[[IDim], float]]:
        tup = (a, b)
        return tup

    @gtx.field_operator
    def cast_tuple(
        a: cases.IFloatField, b: cases.IFloatField, a_asint: cases.IField, b_asint: cases.IField
    ) -> tuple[gtx.Field[[IDim], bool], gtx.Field[[IDim], bool]]:
        result = astype(field_op_returning_a_tuple(a, b), int32)
        return (result[0] == a_asint, result[1] == b_asint)

    @gtx.field_operator
    def cast_nested_tuple(
        a: cases.IFloatField, b: cases.IFloatField, a_asint: cases.IField, b_asint: cases.IField
    ) -> tuple[gtx.Field[[IDim], bool], gtx.Field[[IDim], bool], gtx.Field[[IDim], bool]]:
        result = astype((a, field_op_returning_a_tuple(a, b)), int32)
        return (result[0] == a_asint, result[1][0] == a_asint, result[1][1] == b_asint)

    a = cases.allocate(cartesian_case, cast_tuple, "a")()
    b = cases.allocate(cartesian_case, cast_tuple, "b")()
    a_asint = cartesian_case.as_field([IDim], a.asnumpy().astype(int32))
    b_asint = cartesian_case.as_field([IDim], b.asnumpy().astype(int32))
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
        ref=(
            np.full_like(a.asnumpy(), True, dtype=bool),
            np.full_like(b.asnumpy(), True, dtype=bool),
        ),
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
            np.full_like(a.asnumpy(), True, dtype=bool),
            np.full_like(a.asnumpy(), True, dtype=bool),
            np.full_like(b.asnumpy(), True, dtype=bool),
        ),
    )


def test_astype_bool_field(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IFloatField) -> gtx.Field[[IDim], bool]:
        b = astype(a, bool)
        return b

    cases.verify_with_default_data(
        cartesian_case, testee, ref=lambda a: a.astype(bool), comparison=lambda a, b: np.all(a == b)
    )


@pytest.mark.parametrize("inp", [0.0, 2.0])
def test_astype_bool_scalar(cartesian_case, inp):
    @gtx.field_operator
    def testee(inp: float) -> gtx.Field[[IDim], bool]:
        return broadcast(astype(inp, bool), (IDim,))

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, inp, out=out, ref=bool(inp))


def test_astype_float(cartesian_case):
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


int_alias: TypeAlias = int64


def test_astype_alias(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IFloatField) -> gtx.Field[[IDim], int_alias]:
        b = astype(a, int_alias)
        return b

    cases.verify_with_default_data(
        cartesian_case,
        testee,
        ref=lambda a: a.astype(int_alias),
        comparison=lambda a, b: np.all(a == b),
    )


def test_type_constructor_alias(cartesian_case):
    @gtx.field_operator
    def testee() -> gtx.Field[[IDim], int_alias]:
        return broadcast(int_alias(42), (IDim,))

    ref = cases.allocate(
        cartesian_case, testee, cases.RETURN, strategy=cases.ConstInitializer(42)
    )()

    cases.verify_with_default_data(
        cartesian_case,
        testee,
        ref=lambda: ref,
    )


@pytest.mark.uses_dynamic_offsets
def test_offset_field(cartesian_case):
    ref = np.full(
        (cartesian_case.default_sizes[IDim], cartesian_case.default_sizes[KDim]), True, dtype=bool
    )

    @gtx.field_operator
    def testee(a: cases.IKField, offset_field: cases.IKField) -> gtx.Field[[IDim, KDim], bool]:
        a_i = a(as_offset(Ioff, offset_field))
        # note: this leads to an access to offset_field in
        # IDim: (0, out.size[I]), KDim: (0, out.size[K]+1)
        a_i_k = a_i(as_offset(Koff, offset_field))
        b_i = a(Ioff[1])
        b_i_k = b_i(Koff[1])
        return a_i_k == b_i_k

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1), KDim: (0, 1)})()
    offset_field = (
        cases.allocate(cartesian_case, testee, "offset_field")
        .strategy(cases.ConstInitializer(1))
        .extend({KDim: (0, 1)})()
    )  # see comment at a_i_k for domain bounds

    cases.verify(
        cartesian_case,
        testee,
        a,
        offset_field,
        out=out,
        offset_provider={"Ioff": IDim, "Koff": KDim},
        ref=ref,
        comparison=lambda out, ref: np.all(out == ref),
    )


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


@pytest.mark.uses_unstructured_shift
def test_nested_reduction(unstructured_case):
    @gtx.field_operator
    def testee(a: cases.VField) -> cases.VField:
        tmp = neighbor_sum(a(E2V), axis=E2VDim)
        tmp_2 = neighbor_sum(tmp(V2E), axis=V2EDim)
        return tmp_2

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: np.sum(
            np.sum(a[unstructured_case.offset_provider["E2V"].asnumpy()], axis=1, initial=0)[
                unstructured_case.offset_provider["V2E"].asnumpy()
            ],
            axis=1,
            where=unstructured_case.offset_provider["V2E"].asnumpy() != common._DEFAULT_SKIP_VALUE,
        ),
        comparison=lambda a, tmp_2: np.all(a == tmp_2),
    )


@pytest.mark.uses_unstructured_shift
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
            np.sum(inp[unstructured_case.offset_provider["V2E"].asnumpy()], axis=1)[
                unstructured_case.offset_provider["E2V"].asnumpy()
            ],
            axis=1,
        ),
    )


@pytest.mark.uses_unstructured_shift
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
            np.sum(a[unstructured_case.offset_provider["V2E"].asnumpy()], axis=1),
            np.sum(b[unstructured_case.offset_provider["V2E"].asnumpy()], axis=1),
        ],
        comparison=lambda a, tmp: (np.all(a[0] == tmp[0]), np.all(a[1] == tmp[1])),
    )


@pytest.mark.uses_unstructured_shift
@pytest.mark.uses_constant_fields
def test_tuple_with_local_field_in_reduction_shifted(unstructured_case):
    @gtx.field_operator
    def reduce_tuple_element(e: cases.EField, v: cases.VField) -> cases.EField:
        tup = e(V2E), v
        red = neighbor_sum(tup[0] + v, axis=V2EDim)
        tmp = red(E2V[0])
        return tmp

    v2e = unstructured_case.offset_provider["V2E"]
    cases.verify_with_default_data(
        unstructured_case,
        reduce_tuple_element,
        ref=lambda e, v: np.sum(
            e[v2e.asnumpy()] + np.tile(v, (v2e.shape[1], 1)).T,
            axis=1,
            initial=0,
            where=v2e.asnumpy() != common._DEFAULT_SKIP_VALUE,
        )[unstructured_case.offset_provider["E2V"].asnumpy()[:, 0]],
    )


@pytest.mark.uses_tuple_args
def test_tuple_arg(cartesian_case):
    @gtx.field_operator
    def testee(a: tuple[tuple[cases.IField, cases.IField], cases.IField]) -> cases.IField:
        return 3 * a[0][0] + a[0][1] + a[1]

    cases.verify_with_default_data(
        cartesian_case, testee, ref=lambda a: 3 * a[0][0] + a[0][1] + a[1]
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

    e = a if left < right else b
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


@pytest.mark.uses_constant_fields
@pytest.mark.uses_unstructured_shift
def test_ternary_builtin_neighbor_sum(unstructured_case):
    @gtx.field_operator
    def testee(a: cases.EField, b: cases.EField) -> cases.VField:
        tmp = neighbor_sum(b(V2E) if 2 < 3 else a(V2E), axis=V2EDim)
        return tmp

    v2e_table = unstructured_case.offset_provider["V2E"].asnumpy()
    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a, b: (
            np.sum(b[v2e_table], axis=1, initial=0, where=v2e_table != common._DEFAULT_SKIP_VALUE)
        ),
    )


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
        comparison=lambda ref, out: np.all(out[0] == ref[0])
        and np.all(out[1][0] == ref[1][0])
        and np.all(out[1][1] == ref[1][1]),
    )


@pytest.mark.uses_scan
@pytest.mark.uses_tuple_args
@pytest.mark.uses_tuple_iterator
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
@pytest.mark.uses_tuple_iterator
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
@pytest.mark.uses_tuple_iterator
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


def test_domain(cartesian_case):
    @gtx.field_operator
    def fieldop_domain(a: cases.IField) -> cases.IField:
        return a + a

    @gtx.program
    def program_domain(a: cases.IField, size: int32, out: cases.IField):
        fieldop_domain(a, out=out, domain={IDim: (minimum(1, 2), size)})

    a = cases.allocate(cartesian_case, program_domain, "a")()
    out = cases.allocate(cartesian_case, program_domain, "out")()
    size = cartesian_case.default_sizes[IDim]
    ref = out.asnumpy().copy()  # ensure we are not writing to out outside the domain
    ref[1:size] = a.asnumpy()[1:size] * 2

    cases.verify(cartesian_case, program_domain, a, size, out, inout=out, ref=ref)


@pytest.mark.uses_floordiv
def test_domain_input_bounds(cartesian_case):
    lower_i = 1
    upper_i = cartesian_case.default_sizes[IDim] + 1

    @gtx.field_operator
    def fieldop_domain(a: cases.IField) -> cases.IField:
        return a + a

    @gtx.program
    def program_domain(
        inp: cases.IField, out: cases.IField, lower_i: gtx.IndexType, upper_i: gtx.IndexType
    ):
        fieldop_domain(inp, out=out, domain={IDim: (lower_i, upper_i // 2)})

    inp = cases.allocate(cartesian_case, program_domain, "inp")()
    out = cases.allocate(cartesian_case, fieldop_domain, cases.RETURN)()

    ref = out.asnumpy().copy()
    ref[lower_i : int(upper_i / 2)] = inp.asnumpy()[lower_i : int(upper_i / 2)] * 2

    cases.verify(cartesian_case, program_domain, inp, out, lower_i, upper_i, inout=out, ref=ref)


def test_domain_input_bounds_1(cartesian_case):
    lower_i = 1
    upper_i = cartesian_case.default_sizes[IDim]
    lower_j = cartesian_case.default_sizes[JDim] - 3
    upper_j = cartesian_case.default_sizes[JDim] - 1

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
            a, out=out, domain={IDim: (1 * lower_i, upper_i + 0), JDim: (lower_j - 0, upper_j)}
        )

    a = cases.allocate(cartesian_case, program_domain, "a")()
    out = cases.allocate(cartesian_case, program_domain, "out")()

    ref = out.asnumpy().copy()
    ref[1 * lower_i : upper_i + 0, lower_j - 0 : upper_j] = (
        a.asnumpy()[1 * lower_i : upper_i + 0, lower_j - 0 : upper_j] * 2
    )

    cases.verify(
        cartesian_case,
        program_domain,
        a,
        out,
        lower_i,
        upper_i,
        lower_j,
        upper_j,
        inout=out,
        ref=ref,
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
        isize: int32,
        jsize: int32,
    ):
        fieldop_domain_tuple(
            inp0, inp1, out=(out0, out1), domain={IDim: (1, isize), JDim: (jsize - 2, jsize)}
        )

    inp0 = cases.allocate(cartesian_case, program_domain_tuple, "inp0")()
    inp1 = cases.allocate(cartesian_case, program_domain_tuple, "inp1")()
    out0 = cases.allocate(cartesian_case, program_domain_tuple, "out0")()
    out1 = cases.allocate(cartesian_case, program_domain_tuple, "out1")()

    isize = cartesian_case.default_sizes[IDim]
    jsize = cartesian_case.default_sizes[JDim] - 1
    ref0 = out0.asnumpy().copy()
    ref0[1:isize, jsize - 2 : jsize] = (
        inp0.asnumpy()[1:isize, jsize - 2 : jsize] + inp1.asnumpy()[1:isize, jsize - 2 : jsize]
    )
    ref1 = out1.asnumpy().copy()
    ref1[1:isize, jsize - 2 : jsize] = inp1.asnumpy()[1:isize, jsize - 2 : jsize]

    cases.verify(
        cartesian_case,
        program_domain_tuple,
        inp0,
        inp1,
        out0,
        out1,
        isize,
        jsize,
        inout=(out0, out1),
        ref=(ref0, ref1),
    )


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

    cases.verify(cartesian_case, implicit_broadcast_scalar, inp, out=out, ref=np.array(1))


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
    def unpack(inp: cases.IField) -> tuple[cases.IField, cases.IField, cases.IField, cases.IField]:
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
    def unpack(inp: cases.IField) -> OutType:
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
    with pytest.raises(errors.DSLError, match=(r"Too many values to unpack \(expected 3\).")):

        @gtx.field_operator(backend=cartesian_case.backend)
        def _star_unpack() -> tuple[int32, float64, int32]:
            a, b, c = (1, 2.0, 3, 4, 5, 6, 7.0)
            return a, b, c


def test_tuple_unpacking_too_few_values(cartesian_case):
    with pytest.raises(
        errors.DSLError, match=(r"Assignment value must be of type tuple, got 'int32'.")
    ):

        @gtx.field_operator(backend=cartesian_case.backend)
        def _invalid_unpack() -> tuple[int32, float64, int32]:
            a, b, c = 1
            return a


def test_constant_closure_vars(cartesian_case):
    from gt4py.eve.utils import FrozenNamespace

    constants = FrozenNamespace(PI=np.float64(3.142), E=np.float64(2.718))

    @gtx.field_operator
    def consume_constants(input: cases.IFloatField) -> cases.IFloatField:
        return constants.PI * constants.E * input

    cases.verify_with_default_data(
        cartesian_case, consume_constants, ref=lambda input: constants.PI * constants.E * input
    )


def test_local_index_premapped_field(request, unstructured_case):
    if request.node.get_closest_marker(pytest.mark.uses_mesh_with_skip_values.name):
        pytest.skip("This test only works with non-skip value meshes.")

    @gtx.field_operator
    def testee(inp: gtx.Field[[Edge], int32]) -> gtx.Field[[Vertex], int32]:
        shifted = inp(V2E)
        return shifted[V2EDim(0)] + shifted[V2EDim(1)] + shifted[V2EDim(2)] + shifted[V2EDim(3)]

    inp = cases.allocate(unstructured_case, testee, "inp")()

    v2e_table = unstructured_case.offset_provider["V2E"].asnumpy()
    cases.verify(
        unstructured_case,
        testee,
        inp,
        out=cases.allocate(unstructured_case, testee, cases.RETURN)(),
        ref=np.sum(inp.asnumpy()[v2e_table], axis=1),
    )
