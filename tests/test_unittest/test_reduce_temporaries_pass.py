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
from ..definition_setup import TAssign, TComputationBlock, TDefinition


def test_reduce_temporaries_parallel(
    reduce_temporaries_pass: AnalysisPass,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="parallel", domain=ijk_domain, fields=["out", "in"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("tmp", "in", (1, 0, 0)),
                TAssign("out", "tmp", (0, 1, 0)),
            )
        )
        .build_transform()
    )
    transform_data = reduce_temporaries_pass(transform_data)
    assert len(transform_data.implementation_ir.fields["tmp"].axes) == 3


def test_reduce_temporaries_forward(
    reduce_temporaries_pass: AnalysisPass,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="forward", domain=ijk_domain, fields=["out", "in"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.FORWARD).add_statements(
                TAssign("tmp", "in", (1, 0, 0)),
                TAssign("out", "tmp", (0, 1, 0)),
            )
        )
        .build_transform()
    )
    transform_data = reduce_temporaries_pass(transform_data)
    assert len(transform_data.implementation_ir.fields["tmp"].axes) == 2


def test_reduce_temporaries_k_offset(
    reduce_temporaries_pass: AnalysisPass,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="forward", domain=ijk_domain, fields=["out", "in"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.FORWARD).add_statements(
                TAssign("tmp", "in", (1, 0, 0)),
                TAssign("out", "tmp", (0, 0, 1)),
            )
        )
        .build_transform()
    )
    transform_data = reduce_temporaries_pass(transform_data)
    assert len(transform_data.implementation_ir.fields["tmp"].axes) == 3


def test_reduce_temporaries_diff_intervals(
    reduce_temporaries_pass: AnalysisPass,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="forward", domain=ijk_domain, fields=["out", "in"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.FORWARD, start=0, end=0).add_statements(
                TAssign("tmp", "in", (1, 0, 0))
            ),
            TComputationBlock(order=IterationOrder.FORWARD, start=1, end=-1).add_statements(
                TAssign("out", "tmp", (0, 1, 0)),
            ),
        )
        .build_transform()
    )
    transform_data = reduce_temporaries_pass(transform_data)
    assert len(transform_data.implementation_ir.fields["tmp"].axes) == 3
