# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.cartesian.gtc.passes.oir_optimizations.caches import FillFlushToLocalKCaches
from gt4py.cartesian.gtc.passes.oir_optimizations.vertical_loop_merging import AdjacentLoopMerging
from gt4py.cartesian.gtc.passes.oir_pipeline import DefaultPipeline

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


def test_add_steps():
    add_steps = [FillFlushToLocalKCaches]
    pipeline = DefaultPipeline(add_steps=add_steps)
    pipeline.run(StencilFactory())
    assert all(s in pipeline.add_steps for s in add_steps)
