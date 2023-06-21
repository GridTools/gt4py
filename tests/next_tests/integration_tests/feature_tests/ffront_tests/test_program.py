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

from next_tests.integration_tests.feature_tests import cases
from next_tests.integration_tests.feature_tests.cases import (
    IDim,
    Ioff,
    JDim,
    cartesian_case,
    fieldview_backend,
)
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
        inout=out_field.array()[:-1],
        ref=in_field.array()[1:-1],
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

    assert np.allclose((a, b), (out_a, out_b))


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

    assert np.allclose((a.array()[1:], b.array()[1:]), (out_a.array()[1:], out_b.array()[1:]))
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

    assert np.allclose((a, b, c), (out_a, out_b, out_c))


def test_wrong_argument_type(cartesian_case, copy_program_def):
    copy_program = gtx.program(copy_program_def, backend=cartesian_case.backend)

    inp = gtx.np_as_located_field(JDim)(np.ones((cartesian_case.default_sizes[JDim],)))
    out = cases.allocate(cartesian_case, copy_program, "out").strategy(cases.ConstInitializer(1))()

    with pytest.raises(ValueError) as exc_info:
        # program is defined on Field[[IDim], ...], but we call with
        #  Field[[JDim], ...]
        copy_program(inp, out, offset_provider={})

    msgs = [
        "- Expected argument `in_field` to be of type `Field\[\[IDim], float64\]`,"
        " but got `Field\[\[JDim\], float64\]`.",
    ]
    for msg in msgs:
        assert re.search(msg, exc_info.value.__cause__.args[0]) is not None


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
        empty_domain_program(a, out_field, offset_provider={})


def test_input_kwargs(fieldview_backend):
    size = 10
    input_1 = gtx.np_as_located_field(IDim, JDim)(np.ones((size, size)))
    input_2 = gtx.np_as_located_field(IDim, JDim)(np.ones((size, size)) * 2)
    input_3 = gtx.np_as_located_field(IDim, JDim)(np.ones((size, size)) * 3)

    expected = np.asarray(input_3) * np.asarray(input_1) - np.asarray(input_2)

    @gtx.field_operator(backend=fieldview_backend)
    def fieldop_input_kwargs(
        a: gtx.Field[[IDim, JDim], float64],
        b: gtx.Field[[IDim, JDim], float64],
        c: gtx.Field[[IDim, JDim], float64],
    ) -> gtx.Field[[IDim, JDim], float64]:
        return c * a - b

    out = gtx.np_as_located_field(IDim, JDim)(np.zeros((size, size)))
    fieldop_input_kwargs(input_1, b=input_2, c=input_3, out=out, offset_provider={})
    assert np.allclose(expected, out)

    @gtx.program(backend=fieldview_backend)
    def program_input_kwargs(
        a: gtx.Field[[IDim, JDim], float64],
        b: gtx.Field[[IDim, JDim], float64],
        c: gtx.Field[[IDim, JDim], float64],
        out: gtx.Field[[IDim, JDim], float64],
    ):
        fieldop_input_kwargs(a, b, c, out=out)

    out = gtx.np_as_located_field(IDim, JDim)(np.zeros((size, size)))
    program_input_kwargs(input_1, b=input_2, c=input_3, out=out, offset_provider={})
    assert np.allclose(expected, out)

    out = gtx.np_as_located_field(IDim, JDim)(np.zeros((size, size)))
    program_input_kwargs(a=input_1, b=input_2, c=input_3, out=out, offset_provider={})
    assert np.allclose(expected, out)

    with pytest.raises(ValueError, match="Invalid argument types in call to `program_input_kwargs`!"):
        program_input_kwargs(input_2, input_3, a=input_1, out=out, offset_provider={})
