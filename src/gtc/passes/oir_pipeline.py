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

from abc import ABC, abstractmethod
from typing import Callable, Protocol, Sequence, Type, Union

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
from gtc.passes.oir_optimizations.horizontal_execution_merging import OnTheFlyMerging
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


class OirPipeline(ABC):
    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def run(self, oir: oir.Stencil) -> oir.Stencil:
        pass


class DefaultOirPipeline(OirPipeline):
    """
    OIR passes pipeline runs passes in order and allows skipping.

    May only call existing passes and may not contain any pass logic itself.
    """

    def __init__(self, *, skip: Sequence[PASS_T]):
        self.skip = skip

    @staticmethod
    def all_steps() -> Sequence[PASS_T]:
        return [
            graph_merge_horizontal_executions,
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

    @property
    def _executed_steps(self) -> Sequence[PASS_T]:
        hash_skips = {hash_step(step) for step in self.skip}
        return [step for step in self.all_steps() if hash_step(step) not in hash_skips]

    def __hash__(self) -> int:
        return hash(self._executed_steps)

    def __repr__(self) -> str:
        return str([step.__name__ for step in self._executed_steps])

    def run(self, oir: oir.Stencil) -> oir.Stencil:
        for step in self._executed_steps:
            if isinstance(step, type) and issubclass(step, NodeVisitor):
                oir = step().visit(oir)
            else:
                oir = step(oir)
        return oir
