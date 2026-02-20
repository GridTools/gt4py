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

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, JDim, Joff, cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)

pytestmark = pytest.mark.uses_cartesian_shift


@gtx.field_operator
def lap(in_field: gtx.Field[[IDim, JDim], "float"]) -> gtx.Field[[IDim, JDim], "float"]:
    return (
        -4.0 * in_field
        + in_field(IDim + 1)
        + in_field(JDim + 1)
        + in_field(IDim - 1)
        + in_field(JDim - 1)
    )


@gtx.field_operator
def skewedlap(in_field: gtx.Field[[IDim, JDim], "float"]) -> gtx.Field[[IDim, JDim], "float"]:
    return (
        -4.0 * in_field
        + in_field(IDim + 1, JDim + 1)
        + in_field(IDim + 1, JDim - 1)
        + in_field(IDim - 1, JDim + 1)
        + in_field(IDim - 1, JDim - 1)
    )


@gtx.field_operator
def laplap(in_field: gtx.Field[[IDim, JDim], "float"]) -> gtx.Field[[IDim, JDim], "float"]:
    return lap(lap(in_field))


@gtx.program
def lap_program(
    in_field: gtx.Field[[IDim, JDim], "float"], out_field: gtx.Field[[IDim, JDim], "float"]
):
    lap(in_field, out=out_field[1:-1, 1:-1])


@gtx.program
def skewedlap_program(
    in_field: gtx.Field[[IDim, JDim], "float"],
    out_field: gtx.Field[[IDim, JDim], "float"],
):
    skewedlap(in_field, out=out_field[1:-1, 1:-1])


@gtx.program
def laplap_program(
    in_field: gtx.Field[[IDim, JDim], "float"], out_field: gtx.Field[[IDim, JDim], "float"]
):
    laplap(in_field, out=out_field[2:-2, 2:-2])


def square(inp):
    """Compute the square of the field entries"""
    return inp[:, :] * inp[:, :]


def lap_ref(inp):
    """Compute the laplacian using numpy"""
    return -4.0 * inp[1:-1, 1:-1] + inp[2:, 1:-1] + inp[1:-1, 2:] + inp[:-2, 1:-1] + inp[1:-1, :-2]


def skewedlap_ref(inp):
    """Compute the laplacian using numpy"""
    return -4.0 * inp[1:-1, 1:-1] + inp[2:, 2:] + inp[2:, :-2] + inp[:-2, 2:] + inp[:-2, :-2]


def test_ffront_lap(cartesian_case):
    in_field = cases.allocate(cartesian_case, lap_program, "in_field")()
    in_field = square(in_field)
    out_field = cases.allocate(cartesian_case, lap_program, "out_field")()

    cases.verify(
        cartesian_case,
        lap_program,
        in_field,
        out_field,
        inout=out_field[1:-1, 1:-1],
        ref=lap_ref(in_field.ndarray),
    )


def test_ffront_skewedlap(cartesian_case):
    in_field = cases.allocate(cartesian_case, skewedlap_program, "in_field")()
    in_field = square(in_field)
    out_field = cases.allocate(cartesian_case, skewedlap_program, "out_field")()

    cases.verify(
        cartesian_case,
        skewedlap_program,
        in_field,
        out_field,
        inout=out_field[1:-1, 1:-1],
        ref=skewedlap_ref(in_field.ndarray),
    )


def test_ffront_laplap(cartesian_case):
    in_field = cases.allocate(cartesian_case, laplap_program, "in_field")()
    in_field = square(in_field)
    out_field = cases.allocate(cartesian_case, laplap_program, "out_field")()

    cases.verify(
        cartesian_case,
        laplap_program,
        in_field,
        out_field,
        inout=out_field[2:-2, 2:-2],
        ref=lap_ref(lap_ref(in_field.array_ns.asarray(in_field.ndarray))),
    )
