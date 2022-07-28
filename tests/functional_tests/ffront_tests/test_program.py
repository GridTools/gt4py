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
# TODO(tehrengruber): All field operators and programs should be executable
#  as is at some point. Adopt tests to also run on the regular python objects.
import pathlib
import re

import numpy as np
import pytest

import eve
from eve.pattern_matching import ObjectPattern as P
from functional.common import Field, GridType, GTTypeError
from functional.fencil_processors.runners import roundtrip
from functional.ffront import common_types, program_ast as past
from functional.ffront.decorator import field_operator, program
from functional.ffront.func_to_past import ProgramParser
from functional.ffront.past_passes.type_deduction import ProgramTypeError
from functional.ffront.past_to_itir import ProgramLowering
from functional.iterator import ir as itir
from functional.iterator.embedded import np_as_located_field

from .past_common_fixtures import (
    IDim,
    Ioff,
    copy_program_def,
    copy_restrict_program_def,
    double_copy_program_def,
    float64,
    identity_def,
    invalid_call_sig_program_def,
    invalid_out_slice_dims_program_def,
)


fieldview_backend = roundtrip.executor


def test_identity_fo_execution(identity_def):
    size = 10
    in_field = np_as_located_field(IDim)(np.ones((size)))
    out_field = np_as_located_field(IDim)(np.zeros((size)))
    identity = field_operator(identity_def, backend=fieldview_backend)

    identity(in_field, out=out_field, offset_provider={})

    assert np.allclose(in_field, out_field)


def test_shift_by_one_execution():
    size = 10
    in_field = np_as_located_field(IDim)(np.arange(0, size, 1))
    out_field = np_as_located_field(IDim)(np.zeros((size)))
    out_field_ref = np_as_located_field(IDim)(
        np.array([i + 1 if i in range(0, size - 1) else 0 for i in range(0, size)])
    )

    @field_operator
    def shift_by_one(in_field: Field[[IDim], "float64"]) -> Field[[IDim], "float64"]:
        return in_field(Ioff[1])

    # direct call to field operator
    # TODO(tehrengruber): slicing located fields not supported currently
    # shift_by_one(in_field, out=out_field[:-1], offset_provider={"Ioff": IDim})

    @program
    def shift_by_one_program(
        in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]
    ):
        shift_by_one(in_field, out=out_field[:-1])

    shift_by_one_program.with_backend(fieldview_backend)(
        in_field, out_field, offset_provider={"Ioff": IDim}
    )

    assert np.allclose(out_field, out_field_ref)


def test_copy_execution(copy_program_def):
    size = 10
    in_field = np_as_located_field(IDim)(np.ones((size)))
    out_field = np_as_located_field(IDim)(np.zeros((size)))
    copy_program = program(copy_program_def, backend=fieldview_backend)

    copy_program(in_field, out_field, offset_provider={})

    assert np.allclose(in_field, out_field)


def test_double_copy_execution(double_copy_program_def):
    size = 10
    in_field = np_as_located_field(IDim)(np.ones((size)))
    intermediate_field = np_as_located_field(IDim)(np.zeros((size)))
    out_field = np_as_located_field(IDim)(np.zeros((size)))
    double_copy_program = program(double_copy_program_def, backend=fieldview_backend)

    double_copy_program(in_field, intermediate_field, out_field, offset_provider={})

    assert np.allclose(in_field, out_field)


def test_copy_restricted_execution(copy_restrict_program_def):
    size = 10
    in_field = np_as_located_field(IDim)(np.ones((size)))
    out_field = np_as_located_field(IDim)(np.zeros((size)))
    out_field_ref = np_as_located_field(IDim)(
        np.array([1 if i in range(1, 2) else 0 for i in range(0, size)])
    )
    copy_restrict_program = program(copy_restrict_program_def, backend=fieldview_backend)

    copy_restrict_program(in_field, out_field, offset_provider={})

    assert np.allclose(out_field_ref, out_field)


def test_calling_fo_from_fo_execution(identity_def):
    size = 10
    in_field = np_as_located_field(IDim)(2 * np.ones((size)))
    out_field = np_as_located_field(IDim)(np.zeros((size)))
    out_field_ref = np_as_located_field(IDim)(2 * 2 * 2 * np.ones((size)))

    @field_operator
    def pow_two(field: Field[[IDim], "float64"]) -> Field[[IDim], "float64"]:
        return field * field

    @field_operator
    def pow_three(field: Field[[IDim], "float64"]) -> Field[[IDim], "float64"]:
        return field * pow_two(field)

    @program(backend=fieldview_backend)
    def fo_from_fo_program(in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]):
        pow_three(in_field, out=out_field)

    fo_from_fo_program(in_field, out_field, offset_provider={})

    assert np.allclose(out_field, out_field_ref)
