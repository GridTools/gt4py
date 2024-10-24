# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.cartesian.gtc.passes.gtir_prune_unused_parameters import prune_unused_parameters

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
