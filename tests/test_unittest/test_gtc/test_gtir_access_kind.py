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

from gt4py.definitions import AccessKind
from gtc.passes.gtir_access_kind import compute_access_kinds

from .gtir_utils import FieldDeclFactory, ParAssignStmtFactory, ScalarDeclFactory, StencilFactory


def test_access_read_and_write():
    field_in_param = FieldDeclFactory()
    field_out_param = FieldDeclFactory()
    testee = StencilFactory(
        params=[field_in_param, field_out_param],
        vertical_loops__0__body__0=ParAssignStmtFactory(
            left__name=field_out_param.name, right__name=field_in_param.name
        ),
    )
    access = compute_access_kinds(testee)

    assert access[field_in_param.name] == AccessKind.READ
    assert access[field_out_param.name] == AccessKind.WRITE


def test_access_readwrite():
    field_inout_param = FieldDeclFactory()
    field_out_param = FieldDeclFactory()
    scalar_param = ScalarDeclFactory()
    testee = StencilFactory(
        params=[field_inout_param, field_out_param, scalar_param],
        vertical_loops__0__body=[
            ParAssignStmtFactory(
                left__name=field_out_param.name, right__name=field_inout_param.name
            ),
            ParAssignStmtFactory(left__name=field_inout_param.name, right__name=scalar_param.name),
        ],
    )
    access = compute_access_kinds(testee)

    assert access[field_out_param.name] == AccessKind.WRITE
    assert access[field_inout_param.name] == AccessKind.READ_WRITE


def test_access_write_only():
    field_inout_param = FieldDeclFactory()
    field_out_param = FieldDeclFactory()
    scalar_param = ScalarDeclFactory()
    testee = StencilFactory(
        params=[field_inout_param, field_out_param, scalar_param],
        vertical_loops__0__body=[
            ParAssignStmtFactory(left__name=field_inout_param.name, right__name=scalar_param.name),
            ParAssignStmtFactory(
                left__name=field_out_param.name, right__name=field_inout_param.name
            ),
        ],
    )
    access = compute_access_kinds(testee)

    assert access[field_out_param.name] == AccessKind.WRITE
    assert access[field_inout_param.name] == AccessKind.WRITE
