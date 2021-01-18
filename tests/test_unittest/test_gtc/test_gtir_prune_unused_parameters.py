# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
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

from gtc import common
from gtc.gtir import FieldDecl, ScalarDecl
from gtc.passes.gtir_prune_unused_parameters import prune_unused_parameters

from .gtir_utils import ParAssignStmtBuilder, StencilBuilder


A_ARITHMETIC_TYPE = common.DataType.FLOAT32


def test_all_parameters_used():
    field_param = FieldDecl(name="field", dtype=A_ARITHMETIC_TYPE)
    scalar_param = ScalarDecl(name="scalar", dtype=A_ARITHMETIC_TYPE)
    testee = (
        StencilBuilder()
        .add_param(field_param)
        .add_param(scalar_param)
        .add_par_assign_stmt(ParAssignStmtBuilder("field", "scalar").build())
        .build()
    )
    expected_params = [field_param, scalar_param]

    result = prune_unused_parameters(testee)

    assert expected_params == result.params


def test_unused_are_removed():
    field_param = FieldDecl(name="field", dtype=A_ARITHMETIC_TYPE)
    unused_field_param = FieldDecl(name="unused_field", dtype=A_ARITHMETIC_TYPE)
    scalar_param = ScalarDecl(name="scalar", dtype=A_ARITHMETIC_TYPE)
    unused_scalar_param = ScalarDecl(name="unused_scalar", dtype=A_ARITHMETIC_TYPE)
    testee = (
        StencilBuilder()
        .add_param(field_param)
        .add_param(unused_field_param)
        .add_param(scalar_param)
        .add_param(unused_scalar_param)
        .add_par_assign_stmt(ParAssignStmtBuilder("field", "scalar").build())
        .build()
    )
    expected_params = [field_param, scalar_param]

    result = prune_unused_parameters(testee)

    assert expected_params == result.params
