# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple

import pytest

import gt4py.next as gtx
from gt4py.next import float64


IDim = gtx.Dimension("IDim")
Ioff = gtx.FieldOffset("Ioff", source=IDim, target=(IDim,))
JDim = gtx.Dimension("JDim")
Joff = gtx.FieldOffset("Joff", source=JDim, target=(JDim,))


# TODO(tehrengruber): Improve test structure. Identity needs to be decorated
#  in order to be used inside a program. This is unfortunate as a bug inside
#  the decorator may result in failing tests before the actual test is run.
#  A better way would be to first test everything field operator related,
#  including the decorator and then continue with program tests that then
#  can safely use the field operator decorator inside the fixtures.
@pytest.fixture
def identity_def():
    def identity(in_field: gtx.Field[[IDim], "float64"]) -> gtx.Field[[IDim], "float64"]:
        return in_field

    return identity


@pytest.fixture
def make_tuple_op():
    @gtx.field_operator()
    def make_tuple_op_impl(
        inp: gtx.Field[[IDim], float64],
    ) -> Tuple[gtx.Field[[IDim], float64], gtx.Field[[IDim], float64]]:
        return inp, inp

    return make_tuple_op_impl


@pytest.fixture
def copy_program_def(identity_def):
    identity = gtx.field_operator(identity_def)

    def copy_program(in_field: gtx.Field[[IDim], "float64"], out: gtx.Field[[IDim], "float64"]):
        identity(in_field, out=out)

    return copy_program


@pytest.fixture
def double_copy_program_def(identity_def):
    identity = gtx.field_operator(identity_def)

    def double_copy_program(
        in_field: gtx.Field[[IDim], "float64"],
        intermediate_field: gtx.Field[[IDim], "float64"],
        out: gtx.Field[[IDim], "float64"],
    ):
        identity(in_field, out=intermediate_field)
        identity(intermediate_field, out=out)

    return double_copy_program


@pytest.fixture
def copy_restrict_program_def(identity_def):
    identity = gtx.field_operator(identity_def)

    def copy_restrict_program(
        in_field: gtx.Field[[IDim], "float64"], out: gtx.Field[[IDim], "float64"]
    ):
        identity(in_field, out=out[1:2])

    return copy_restrict_program


@pytest.fixture
def invalid_call_sig_program_def(identity_def):
    identity = gtx.field_operator(identity_def)

    def invalid_call_sig_program(
        in_field: gtx.Field[[IDim], "float64"], out_field: gtx.Field[[IDim], "float64"]
    ):
        identity(in_field, out_field)

    return invalid_call_sig_program


@pytest.fixture
def invalid_out_slice_dims_program_def(identity_def):
    identity = gtx.field_operator(identity_def)

    def invalid_out_slice_dims_program(
        in_field: gtx.Field[[IDim], "float64"], out_field: gtx.Field[[IDim], "float64"]
    ):
        identity(in_field, out=out_field[1:2, 3:4])

    return invalid_out_slice_dims_program
