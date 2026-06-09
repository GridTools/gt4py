# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import TypeAlias

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import astype, broadcast, float32, int32, int64, neighbor_sum

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    E2V,
    E2VDim,
    Edge,
    IDim,
    Vertex,
    cartesian_case,
    unstructured_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)


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
