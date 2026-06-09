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
from gt4py.next import broadcast, int32, minimum

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    IDim,
    JDim,
    cartesian_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
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


@pytest.mark.uses_program_with_sliced_out_arguments
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


@pytest.mark.uses_program_with_sliced_out_arguments
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
