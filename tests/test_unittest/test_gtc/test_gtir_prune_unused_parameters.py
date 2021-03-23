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

from gtc.passes.gtir_prune_unused_parameters import prune_unused_parameters

from .gtir_utils import FieldDeclFactory, ParAssignStmtFactory, ScalarDeclFactory, StencilFactory


def test_all_parameters_used():
    field_param = FieldDeclFactory()
    scalar_param = ScalarDeclFactory()
    testee = StencilFactory(
        params=[field_param, scalar_param],
        vertical_loops__0__body__0=ParAssignStmtFactory(
            left__name=field_param.name, right__name=scalar_param.name
        ),
    )
    expected_params = [field_param, scalar_param]

    result = prune_unused_parameters(testee)

    assert expected_params == result.params


def test_unused_are_removed():
    field_param = FieldDeclFactory()
    unused_field_param = FieldDeclFactory()
    scalar_param = ScalarDeclFactory()
    unused_scalar_param = ScalarDeclFactory()
    testee = StencilFactory(
        params=[field_param, unused_field_param, scalar_param, unused_scalar_param],
        vertical_loops__0__body__0=ParAssignStmtFactory(
            left__name=field_param.name, right__name=scalar_param.name
        ),
    )
    expected_params = [field_param, scalar_param]

    result = prune_unused_parameters(testee)

    assert expected_params == result.params
