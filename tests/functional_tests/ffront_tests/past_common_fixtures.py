# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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
from typing import Tuple

import pytest

from functional.common import Field
from functional.ffront.decorator import field_operator
from functional.ffront.fbuiltins import Dimension, FieldOffset


float64 = float
IDim = Dimension("IDim")
Ioff = FieldOffset("Ioff", source=IDim, target=(IDim,))


# TODO(tehrengruber): Improve test structure. Identity needs to be decorated
#  in order to be used inside a program. This is unfortunate as a bug inside
#  the decorator may result in failing tests before the actual test is run.
#  A better way would be to first test everything field operator related,
#  including the decorator and then continue with program tests that then
#  can safely use the field operator decorator inside the fixtures.
@pytest.fixture
def identity_def():
    def identity(in_field: Field[[IDim], "float64"]) -> Field[[IDim], "float64"]:
        return in_field

    return identity


@pytest.fixture
def make_tuple_op():
    @field_operator()
    def make_tuple_op_impl(
        inp: Field[[IDim], float64]
    ) -> Tuple[Field[[IDim], float64], Field[[IDim], float64]]:
        return inp, inp

    return make_tuple_op_impl


@pytest.fixture
def copy_program_def(identity_def):
    identity = field_operator(identity_def)

    def copy_program(in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]):
        identity(in_field, out=out_field)

    return copy_program


@pytest.fixture
def double_copy_program_def(identity_def):
    identity = field_operator(identity_def)

    def double_copy_program(
        in_field: Field[[IDim], "float64"],
        intermediate_field: Field[[IDim], "float64"],
        out_field: Field[[IDim], "float64"],
    ):
        identity(in_field, out=intermediate_field)
        identity(intermediate_field, out=out_field)

    return double_copy_program


@pytest.fixture
def copy_restrict_program_def(identity_def):
    identity = field_operator(identity_def)

    def copy_restrict_program(
        in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]
    ):
        identity(in_field, out=out_field[1:2])

    return copy_restrict_program


@pytest.fixture
def invalid_call_sig_program_def(identity_def):
    identity = field_operator(identity_def)

    def invalid_call_sig_program(
        in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]
    ):
        identity(in_field, out_field)

    return invalid_call_sig_program


@pytest.fixture
def invalid_out_slice_dims_program_def(identity_def):
    identity = field_operator(identity_def)

    def invalid_out_slice_dims_program(
        in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]
    ):
        identity(in_field, out=out_field[1:2, 3:4])

    return invalid_out_slice_dims_program
