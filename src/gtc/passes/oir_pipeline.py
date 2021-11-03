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

from abc import abstractmethod
from typing import Callable, Optional, Protocol, Sequence, Type, Union

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


PassT = Union[Callable[[oir.Stencil], oir.Stencil], Type[NodeVisitor]]


class OirPipeline(Protocol):
    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError("Missing implementation of __hash__")

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError("Missing implementation of __repr__")

    @abstractmethod
    def run(self, oir: oir.Stencil) -> oir.Stencil:
        raise NotImplementedError("Missing implementation of run")


class DefaultPipeline(OirPipeline):
    """
    OIR passes pipeline runs passes in order and allows skipping.

    May only call existing passes and may not contain any pass logic itself.
    """

    def __init__(self, *, skip: Optional[Sequence[PassT]] = None):
        self.skip = skip or []

    @staticmethod
    def all_steps() -> Sequence[PassT]:
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
    def steps(self) -> Sequence[PassT]:
        return [step for step in self.all_steps() if step not in self.skip]

    def __hash__(self) -> int:
        return hash(repr(self))

    def __repr__(self) -> str:
        return str([step.__name__ for step in self.steps])

    def run(self, oir: oir.Stencil) -> oir.Stencil:
        for step in self.steps:
            if isinstance(step, type) and issubclass(step, NodeVisitor):
                oir = step().visit(oir)
            else:
                oir = step(oir)
        return oir
