# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import math

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import broadcast, errors, int32

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)


def test_copy(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKField) -> cases.IJKField:
        field_tuple = (a, a)
        field_0 = field_tuple[0]
        field_1 = field_tuple[1]
        return field_0

    cases.verify_with_default_data(cartesian_case, testee, ref=lambda a: a)


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
