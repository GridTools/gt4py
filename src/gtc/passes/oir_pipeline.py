# -*- coding: utf-8 -*-
#
# GT4Py - GridTools Framework
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

from typing import Callable, Dict, Optional, Protocol, Sequence, Tuple, Type, Union

from eve.visitors import NodeVisitor
from gtc import oir
from gtc.passes.oir_dace_optimizations.horizontal_execution_merging import (
    graph_merge_horizontal_executions,
)
from gtc.passes.oir_optimizations.caches import (
    FillFlushToLocalKCaches,
    IJCacheDetection,
    KCacheDetection,
    PruneKCacheFills,
    PruneKCacheFlushes,
)
from gtc.passes.oir_optimizations.horizontal_execution_merging import GreedyMerging, OnTheFlyMerging
from gtc.passes.oir_optimizations.inlining import MaskInlining
from gtc.passes.oir_optimizations.mask_stmt_merging import MaskStmtMerging
from gtc.passes.oir_optimizations.pruning import NoFieldAccessPruning
from gtc.passes.oir_optimizations.temporaries import (
    LocalTemporariesToScalars,
    WriteBeforeReadTemporariesToScalars,
)
from gtc.passes.oir_optimizations.vertical_loop_merging import AdjacentLoopMerging


PASS_T = Union[Callable[[oir.Stencil], oir.Stencil], Type[NodeVisitor]]


class ClassMethodPass(Protocol):
    __func__: Callable[[oir.Stencil], oir.Stencil]


def hash_step(step: Callable) -> int:
    return hash(step)


class OirPipeline:
    """
    OIR passes pipeline runs passes in order and allows skipping.

    May only call existing passes and may not contain any pass logic itself.
    """

    def __init__(self, node: oir.Stencil):
        self.oir = node
        self._cache: Dict[Tuple[int, ...], oir.Stencil] = {}

    def steps(self) -> Sequence[PASS_T]:
        return [
            graph_merge_horizontal_executions,
            GreedyMerging,
            AdjacentLoopMerging,
            LocalTemporariesToScalars,
            WriteBeforeReadTemporariesToScalars,
            OnTheFlyMerging,
            MaskStmtMerging,
            MaskInlining,
            NoFieldAccessPruning,
            IJCacheDetection,
            KCacheDetection,
            PruneKCacheFills,
            PruneKCacheFlushes,
            FillFlushToLocalKCaches,
        ]

    def apply(self, steps: Sequence[PASS_T]) -> oir.Stencil:
        result = self.oir
        for step in steps:
            if isinstance(step, type) and issubclass(step, NodeVisitor):
                result = step().visit(result)
            else:
                result = step(result)
        return result

    def _get_cached(self, steps: Sequence[PASS_T]) -> Optional[oir.Stencil]:
        return self._cache.get(tuple(hash_step(step) for step in steps))

    def _set_cached(self, steps: Sequence[PASS_T], node: oir.Stencil) -> oir.Stencil:
        return self._cache.setdefault(tuple(hash_step(step) for step in steps), node)

    def _should_execute_step(self, step: PASS_T, skip: Sequence[PASS_T]) -> bool:
        skip_hashes = [hash_step(skip_step) for skip_step in skip]
        if hash_step(step) in skip_hashes:
            return False
        return True

    def full(self, skip: Sequence[PASS_T] = None) -> oir.Stencil:
        skip = skip or []
        pipeline = [step for step in self.steps() if self._should_execute_step(step, skip)]
        return self._get_cached(pipeline) or self._set_cached(pipeline, self.apply(pipeline))
