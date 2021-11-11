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

from typing import Tuple

from gt4py.ir.nodes import Domain, IterationOrder

from ..analysis_setup import AnalysisPass
from ..definition_setup import TAssign, TComputationBlock, TDefinition


def test_write_after_read_ij_extended(
    normalize_blocks_pass: AnalysisPass,
    iteration_order: IterationOrder,
    ij_offset: Tuple[int, int, int],
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="ij_extended", domain=ijk_domain, fields=["out", "in", "inout"])
        .add_blocks(
            TComputationBlock(order=iteration_order).add_statements(
                TAssign("tmp", "inout", (0, 0, 0)),
                TAssign("out", "tmp", ij_offset),
                TAssign("inout", "in", (0, 0, 0)),
            )
        )
        .build_transform()
    )
    transform_data = normalize_blocks_pass(transform_data)
    assert len(transform_data.blocks) == 3


def test_write_after_read_ij_offset(
    normalize_blocks_pass: AnalysisPass,
    iteration_order: IterationOrder,
    ij_offset: Tuple[int, int, int],
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="ij_offset_readonly", domain=ijk_domain, fields=["out", "in", "inout"])
        .add_blocks(
            TComputationBlock(order=iteration_order).add_statements(
                TAssign("out", "in", ij_offset), TAssign("inout", "in", (0, 0, 0))
            )
        )
        .build_transform()
    )
    transform_data = normalize_blocks_pass(transform_data)
    assert len(transform_data.blocks) == 2
