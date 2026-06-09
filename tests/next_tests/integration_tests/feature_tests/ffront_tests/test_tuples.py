# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import (
    broadcast,
    errors,
    float64,
    int32,
    neighbor_sum,
    utils as gt_utils,
)

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    V2E,
    IDim,
    V2EDim,
    Vertex,
    cartesian_case,
    unstructured_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)


@pytest.mark.uses_tuple_returns
def test_multicopy(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKField, b: cases.IJKField) -> tuple[cases.IJKField, cases.IJKField]:
        return a, b

    cases.verify_with_default_data(cartesian_case, testee, ref=lambda a, b: (a, b))


def test_tuples(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKFloatField, b: cases.IJKFloatField) -> cases.IJKFloatField:
        inps = a, b
        scalars = 1.3, float64(5.0), float64("3.4")
        return (inps[0] * scalars[0] + inps[1] * scalars[1]) * scalars[2]

    cases.verify_with_default_data(
        cartesian_case, testee, ref=lambda a, b: (a * 1.3 + b * 5.0) * 3.4
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
        ref=lambda a: (
            np.full(
                [cartesian_case.default_sizes[IDim]], a[0] + 2 * a[1][0] + 5 * a[1][2], dtype=int32
            )
            + 3 * a[1][1]
        ),
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


@pytest.mark.uses_tuple_args
def test_tuple_arg(cartesian_case):
    @gtx.field_operator
    def testee(a: tuple[tuple[cases.IField, cases.IField], cases.IField]) -> cases.IField:
        return 3 * a[0][0] + a[0][1] + a[1]

    cases.verify_with_default_data(
        cartesian_case, testee, ref=lambda a: 3 * a[0][0] + a[0][1] + a[1]
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
