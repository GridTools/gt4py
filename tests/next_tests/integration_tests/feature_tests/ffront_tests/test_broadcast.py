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
from gt4py.next import broadcast, float64, int32

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, JDim, Joff, cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)


def test_broadcast_simple(cartesian_case):
    @gtx.field_operator
    def simple_broadcast(inp: cases.IField) -> cases.IJField:
        return broadcast(inp, (IDim, JDim))

    cases.verify_with_default_data(
        cartesian_case, simple_broadcast, ref=lambda inp: inp[:, np.newaxis]
    )


def test_broadcast_scalar(cartesian_case):
    size = cartesian_case.default_sizes[IDim]

    @gtx.field_operator
    def scalar_broadcast() -> gtx.Field[[IDim], float64]:
        return broadcast(float(1.0), (IDim,))

    cases.verify_with_default_data(cartesian_case, scalar_broadcast, ref=lambda: np.ones(size))


def test_broadcast_two_fields(cartesian_case):
    @gtx.field_operator
    def broadcast_two_fields(inp1: cases.IField, inp2: gtx.Field[[JDim], int32]) -> cases.IJField:
        a = broadcast(inp1, (IDim, JDim))
        b = broadcast(inp2, (IDim, JDim))
        return a + b

    cases.verify_with_default_data(
        cartesian_case, broadcast_two_fields, ref=lambda a, b: a[:, np.newaxis] + b[np.newaxis, :]
    )


@pytest.mark.uses_cartesian_shift
def test_broadcast_shifted(cartesian_case):
    @gtx.field_operator
    def simple_broadcast(inp: cases.IField) -> cases.IJField:
        bcasted = broadcast(inp, (IDim, JDim))
        return bcasted(Joff[1])

    cases.verify_with_default_data(
        cartesian_case, simple_broadcast, ref=lambda inp: inp[:, np.newaxis]
    )


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
