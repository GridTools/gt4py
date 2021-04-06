# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.ir.nodes import Domain, IterationOrder

from ..analysis_setup import AnalysisPass
from ..definition_setup import TAssign, TComputationBlock, TDefinition, TScalarLiteral


def test_constant_folding_assign_once(
    constant_folding_pass: AnalysisPass,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="assign_once", domain=ijk_domain, fields=["out", "in"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("tmp", TScalarLiteral(value=1.0), (0, 0, 0)),
                TAssign("out", "tmp", (0, 0, 0)),
            )
        )
        .build_transform()
    )
    transform_data = constant_folding_pass(transform_data)
    assert not transform_data.implementation_ir.temporary_fields


def test_constant_folding_assign_twice(
    constant_folding_pass: AnalysisPass,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="assign_once", domain=ijk_domain, fields=["out", "in"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("tmp", TScalarLiteral(value=1.0), (0, 0, 0)),
            ),
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("tmp", TScalarLiteral(value=2.0), (0, 0, 0)),
            ),
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("out", "tmp", (0, 0, 0)),
            ),
        )
        .build_transform()
    )
    transform_data = constant_folding_pass(transform_data)
    assert transform_data.implementation_ir.temporary_fields == ["tmp"]
