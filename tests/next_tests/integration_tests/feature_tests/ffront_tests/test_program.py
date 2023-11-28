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

# TODO(tehrengruber): All field operators and programs should be executable
#  as is at some point. Adopt tests to also run on the regular python objects.
import re

import numpy as np
import pytest

import gt4py.next as gtx

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, Ioff, JDim, cartesian_case, fieldview_backend
from next_tests.past_common_fixtures import (
    copy_program_def,
    copy_restrict_program_def,
    double_copy_program_def,
    float64,
    identity_def,
)


def test_identity_fo_execution(cartesian_case, identity_def):
    identity = gtx.field_operator(identity_def, backend=cartesian_case.backend)

    in_field = cases.allocate(cartesian_case, identity, "in_field").strategy(
        cases.ConstInitializer(1)
    )()
    out_field = cases.allocate(cartesian_case, identity, "in_field").strategy(
        cases.ConstInitializer(0)
    )()

    cases.verify(
        cartesian_case,
        identity,
        in_field,
        out=out_field,
        ref=np.ones((cartesian_case.default_sizes[IDim])),
    )


@pytest.mark.uses_cartesian_shift
def test_shift_by_one_execution(cartesian_case):
    @gtx.field_operator
    def shift_by_one(in_field: cases.IFloatField) -> cases.IFloatField:
        return in_field(Ioff[1])

    # direct call to field operator
    # TODO(tehrengruber): slicing located fields not supported currently
    # shift_by_one(in_field, out=out_field[:-1], offset_provider={"Ioff": IDim})

    @gtx.program
    def shift_by_one_program(in_field: cases.IFloatField, out_field: cases.IFloatField):
        shift_by_one(in_field, out=out_field[:-1])

    in_field = cases.allocate(cartesian_case, shift_by_one_program, "in_field").extend(
        {IDim: (0, 1)}
    )()
    out_field = cases.allocate(cartesian_case, shift_by_one_program, "out_field")()

    cases.verify(
        cartesian_case,
        shift_by_one_program,
        in_field,
        out_field,
        inout=out_field[:-1],
        ref=in_field[1:-1],
    )


def test_copy_execution(cartesian_case, copy_program_def):
    copy_program = gtx.program(copy_program_def, backend=cartesian_case.backend)

    cases.verify_with_default_data(cartesian_case, copy_program, ref=lambda in_field: in_field)


def test_double_copy_execution(cartesian_case, double_copy_program_def):
    double_copy_program = gtx.program(double_copy_program_def, backend=cartesian_case.backend)

    cases.verify_with_default_data(
        cartesian_case, double_copy_program, ref=lambda in_field, intermediate_field: in_field
    )


def test_copy_restricted_execution(cartesian_case, copy_restrict_program_def):
    copy_restrict_program = gtx.program(copy_restrict_program_def, backend=cartesian_case.backend)

    cases.verify_with_default_data(
        cartesian_case,
        copy_restrict_program,
        ref=lambda in_field: np.array(
            [
                in_field[i] if i in range(1, 2) else 0
                for i in range(0, cartesian_case.default_sizes[IDim])
            ]
        ),
    )


def test_calling_fo_from_fo_execution(cartesian_case):
    @gtx.field_operator
    def pow_two(field: cases.IFloatField) -> cases.IFloatField:
        return field * field

    @gtx.field_operator
    def pow_three(field: cases.IFloatField) -> cases.IFloatField:
        return field * pow_two(field)

    @gtx.program
    def fo_from_fo_program(in_field: cases.IFloatField, out: cases.IFloatField):
        pow_three(in_field, out=out)

    cases.verify_with_default_data(
        cartesian_case,
        fo_from_fo_program,
        ref=lambda in_field: in_field**3,
    )


def test_tuple_program_return_constructed_inside(cartesian_case):
    @gtx.field_operator
    def pack_tuple(
        a: cases.IFloatField, b: cases.IFloatField
    ) -> tuple[cases.IFloatField, cases.IFloatField]:
        return (a, b)

    @gtx.program
    def prog(
        a: cases.IFloatField,
        b: cases.IFloatField,
        out_a: cases.IFloatField,
        out_b: cases.IFloatField,
    ):
        pack_tuple(a, b, out=(out_a, out_b))

    a = cases.allocate(cartesian_case, prog, "a")()
    b = cases.allocate(cartesian_case, prog, "b")()
    out_a = cases.allocate(cartesian_case, prog, "out_a")()
    out_b = cases.allocate(cartesian_case, prog, "out_b")()

    cases.run(cartesian_case, prog, a, b, out_a, out_b, offset_provider={})

    assert np.allclose((a.asnumpy(), b.asnumpy()), (out_a.asnumpy(), out_b.asnumpy()))


def test_tuple_program_return_constructed_inside_with_slicing(cartesian_case):
    @gtx.field_operator
    def pack_tuple(
        a: cases.IFloatField, b: cases.IFloatField
    ) -> tuple[cases.IFloatField, cases.IFloatField]:
        return (a, b)

    @gtx.program
    def prog(
        a: cases.IFloatField,
        b: cases.IFloatField,
        out_a: cases.IFloatField,
        out_b: cases.IFloatField,
    ):
        pack_tuple(a, b, out=(out_a[1:], out_b[1:]))

    a = cases.allocate(cartesian_case, prog, "a").strategy(cases.ConstInitializer(1))()
    b = cases.allocate(cartesian_case, prog, "b").strategy(cases.ConstInitializer(2))()
    out_a = cases.allocate(cartesian_case, prog, "out_a").strategy(cases.ConstInitializer(0))()
    out_b = cases.allocate(cartesian_case, prog, "out_b").strategy(cases.ConstInitializer(0))()

    cases.run(cartesian_case, prog, a, b, out_a, out_b, offset_provider={})

    assert np.allclose(
        (a[1:].asnumpy(), b[1:].asnumpy()), (out_a[1:].asnumpy(), out_b[1:].asnumpy())
    )
    assert out_a[0] == 0 and out_b[0] == 0


def test_tuple_program_return_constructed_inside_nested(cartesian_case):
    @gtx.field_operator
    def pack_tuple(
        a: cases.IFloatField, b: cases.IFloatField, c: cases.IFloatField
    ) -> tuple[tuple[cases.IFloatField, cases.IFloatField], cases.IFloatField]:
        return ((a, b), c)

    @gtx.program
    def prog(
        a: cases.IFloatField,
        b: cases.IFloatField,
        c: cases.IFloatField,
        out_a: cases.IFloatField,
        out_b: cases.IFloatField,
        out_c: cases.IFloatField,
    ):
        pack_tuple(a, b, c, out=((out_a, out_b), out_c))

    a = cases.allocate(cartesian_case, prog, "a").strategy(cases.ConstInitializer(1))()
    b = cases.allocate(cartesian_case, prog, "b").strategy(cases.ConstInitializer(2))()
    c = cases.allocate(cartesian_case, prog, "b").strategy(cases.ConstInitializer(3))()
    out_a = cases.allocate(cartesian_case, prog, "out_a").strategy(cases.ConstInitializer(0))()
    out_b = cases.allocate(cartesian_case, prog, "out_b").strategy(cases.ConstInitializer(0))()
    out_c = cases.allocate(cartesian_case, prog, "out_c").strategy(cases.ConstInitializer(0))()

    cases.run(cartesian_case, prog, a, b, c, out_a, out_b, out_c, offset_provider={})

    assert np.allclose(
        (a.asnumpy(), b.asnumpy(), c.asnumpy()), (out_a.asnumpy(), out_b.asnumpy(), out_c.asnumpy())
    )


def test_wrong_argument_type(cartesian_case, copy_program_def):
    copy_program = gtx.program(copy_program_def, backend=cartesian_case.backend)

    inp = cartesian_case.as_field([JDim], np.ones((cartesian_case.default_sizes[JDim],)))
    out = cases.allocate(cartesian_case, copy_program, "out").strategy(cases.ConstInitializer(1))()

    with pytest.raises(TypeError) as exc_info:
        # program is defined on Field[[IDim], ...], but we call with
        #  Field[[JDim], ...]
        copy_program(inp, out, offset_provider={})

    msgs = [
        "- Expected argument `in_field` to be of type `Field\[\[IDim], float64\]`,"
        " but got `Field\[\[JDim\], float64\]`.",
    ]
    for msg in msgs:
        assert re.search(msg, exc_info.value.__cause__.args[0]) is not None


@pytest.mark.checks_specific_error
def test_dimensions_domain(cartesian_case):
    @gtx.field_operator
    def empty_domain_fieldop(a: cases.IJField):
        return a

    @gtx.program
    def empty_domain_program(a: cases.IJField, out_field: cases.IJField):
        empty_domain_fieldop(a, out=out_field, domain={JDim: (0, 1), IDim: (0, 1)})

    a = cases.allocate(cartesian_case, empty_domain_program, "a")()
    out_field = cases.allocate(cartesian_case, empty_domain_program, "out_field")()

    with pytest.raises(
        ValueError,
        match=(r"Dimensions in out field and field domain are not equivalent"),
    ):
        cases.run(cartesian_case, empty_domain_program, a, out_field, offset_provider={})
