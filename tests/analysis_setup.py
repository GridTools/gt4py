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

from typing import Callable, Iterator

import pytest

from gt4py.analysis import TransformData
from gt4py.analysis.passes import (
    BuildIIRPass,
    ComputeExtentsPass,
    ComputeUsedSymbolsPass,
    DemoteLocalTemporariesToVariablesPass,
    InitInfoPass,
    MergeBlocksPass,
    NormalizeBlocksPass,
)


AnalysisPass = Callable[[TransformData], TransformData]


@pytest.fixture()
def init_pass() -> Iterator[AnalysisPass]:
    def inner(data: TransformData) -> TransformData:
        InitInfoPass.apply(data)
        return data

    yield inner


@pytest.fixture()
def normalize_blocks_pass(init_pass: AnalysisPass) -> Iterator[AnalysisPass]:
    def inner(data: TransformData) -> TransformData:
        NormalizeBlocksPass.apply(init_pass(data))
        return data

    yield inner


@pytest.fixture()
def compute_extents_pass(normalize_blocks_pass: AnalysisPass) -> Iterator[AnalysisPass]:
    def inner(data: TransformData) -> TransformData:
        ComputeExtentsPass.apply(normalize_blocks_pass(data))
        return data

    yield inner


@pytest.fixture()
def merge_blocks_pass(compute_extents_pass: AnalysisPass) -> Iterator[AnalysisPass]:
    def inner(data: TransformData) -> TransformData:
        MergeBlocksPass.apply(compute_extents_pass(data))
        return data

    yield inner


@pytest.fixture()
def compute_used_symbols_pass(
    merge_blocks_pass: AnalysisPass,
) -> Iterator[AnalysisPass]:
    def inner(data: TransformData) -> TransformData:
        ComputeUsedSymbolsPass.apply(merge_blocks_pass(data))
        return data

    yield inner


@pytest.fixture()
def build_iir_pass(compute_used_symbols_pass: AnalysisPass) -> Iterator[AnalysisPass]:
    def inner(data: TransformData) -> TransformData:
        BuildIIRPass.apply(compute_used_symbols_pass(data))
        return data

    yield inner


@pytest.fixture()
def demote_locals_pass(build_iir_pass: AnalysisPass) -> Iterator[AnalysisPass]:
    def inner(data: TransformData) -> TransformData:
        DemoteLocalTemporariesToVariablesPass.apply(build_iir_pass(data))
        return data

    yield inner
