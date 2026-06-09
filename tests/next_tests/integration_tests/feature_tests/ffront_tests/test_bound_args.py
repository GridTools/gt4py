# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import re

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import int32
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, cartesian_case
from next_tests.integration_tests.cases_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)


def test_with_bound_args(cartesian_case):
    @gtx.field_operator
    def fieldop_bound_args(a: cases.IField, scalar: int32, condition: bool) -> cases.IField:
        if not condition:
            scalar = 0
        return a + scalar

    @gtx.program
    def program_bound_args(a: cases.IField, scalar: int32, condition: bool, out: cases.IField):
        fieldop_bound_args(a, scalar, condition, out=out)

    a = cases.allocate(cartesian_case, program_bound_args, "a")()
    scalar = int32(1)
    ref = a + scalar
    out = cases.allocate(cartesian_case, program_bound_args, "out")()

    prog_bounds = program_bound_args.with_bound_args(scalar=scalar, condition=True)
    cases.verify(cartesian_case, prog_bounds, a, out, inout=out, ref=ref)


def test_with_bound_args_order_args(cartesian_case):
    @gtx.field_operator
    def fieldop_args(a: cases.IField, condition: bool, scalar: int32) -> cases.IField:
        scalar = 0 if not condition else scalar
        return a + scalar

    @gtx.program(backend=cartesian_case.backend)
    def program_args(a: cases.IField, condition: bool, scalar: int32, out: cases.IField):
        fieldop_args(a, condition, scalar, out=out)

    a = cases.allocate(cartesian_case, program_args, "a")()
    out = cases.allocate(cartesian_case, program_args, "out")()

    prog_bounds = program_args.with_bound_args(condition=True)
    prog_bounds(a=a, scalar=int32(1), out=out, offset_provider={})
    assert np.allclose(out.asnumpy(), a.asnumpy() + int32(1))


@pytest.fixture
def bound_args_testee():
    @field_operator
    def fieldop_bound_args() -> cases.IField:
        return broadcast(0, (IDim,))

    @program
    def program_bound_args(arg1: bool, arg2: bool, out: cases.IField):
        # for the test itself we don't care what happens here, but empty programs are not supported
        fieldop_bound_args(out=out)

    return program_bound_args


def test_bind_invalid_arg(cartesian_case, bound_args_testee):
    with pytest.raises(
        TypeError, match="Keyword argument 'inexistent_arg' is not a valid program parameter."
    ):
        bound_args_testee.with_bound_args(inexistent_arg=1)


def test_call_bound_program_with_wrong_args(cartesian_case, bound_args_testee):
    program_with_bound_arg = bound_args_testee.with_bound_args(arg1=True)
    out = cases.allocate(cartesian_case, bound_args_testee, "out")()

    with pytest.raises(TypeError) as exc_info:
        program_with_bound_arg.with_backend(cartesian_case.backend)(out, offset_provider={})

    assert (
        re.search(
            "Function takes 2 positional arguments, but 1 were given.",
            exc_info.value.__cause__.args[0],
        )
        is not None
    )


def test_call_bound_program_with_already_bound_arg(cartesian_case, bound_args_testee):
    program_with_bound_arg = bound_args_testee.with_bound_args(arg2=True)
    out = cases.allocate(cartesian_case, bound_args_testee, "out")()

    with pytest.raises(TypeError) as exc_info:
        program_with_bound_arg.with_backend(cartesian_case.backend)(
            True, out, arg2=True, offset_provider={}
        )

    assert (
        re.search(
            "Parameter 'arg2' already set as a bound argument.", exc_info.value.__cause__.args[0]
        )
        is not None
    )
