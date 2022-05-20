# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from gtc.passes.oir_optimizations.vertical_loop_merging import AdjacentLoopMerging
from gtc.passes.oir_pipeline import DefaultPipeline

from .oir_utils import StencilFactory


def test_no_skipping():
    pipeline = DefaultPipeline()
    pipeline.run(StencilFactory())
    assert pipeline.steps == DefaultPipeline.all_steps()


def test_skip():
    skip = [AdjacentLoopMerging]
    pipeline = DefaultPipeline(skip=skip)
    pipeline.run(StencilFactory())
    assert all(s not in pipeline.steps for s in skip)
