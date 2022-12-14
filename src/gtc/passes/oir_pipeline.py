# GT4Py - GridTools Framework
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

from abc import abstractmethod
from typing import Callable, Optional, Protocol, Sequence, Type, Union

from gt4py.cartesian.gtc import oir
from gt4py.cartesian.gtc.passes.oir_optimizations.caches import (
    IJCacheDetection,
    KCacheDetection,
    PruneKCacheFills,
    PruneKCacheFlushes,
)
from gt4py.cartesian.gtc.passes.oir_optimizations.horizontal_execution_merging import (
    HorizontalExecutionMerging,
    OnTheFlyMerging,
)
from gt4py.cartesian.gtc.passes.oir_optimizations.inlining import MaskInlining
from gt4py.cartesian.gtc.passes.oir_optimizations.mask_stmt_merging import MaskStmtMerging
from gt4py.cartesian.gtc.passes.oir_optimizations.pruning import (
    NoFieldAccessPruning,
    UnreachableStmtPruning,
)
from gt4py.cartesian.gtc.passes.oir_optimizations.temporaries import (
    LocalTemporariesToScalars,
    WriteBeforeReadTemporariesToScalars,
)
from gt4py.cartesian.gtc.passes.oir_optimizations.vertical_loop_merging import AdjacentLoopMerging
from gt4py.eve.visitors import NodeVisitor


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

    def __init__(
        self, *, skip: Optional[Sequence[PassT]] = None, add_steps: Optional[Sequence[PassT]] = None
    ):
        self.skip = list(skip or [])
        self.add_steps = list(add_steps or [])

    @staticmethod
    def all_steps() -> Sequence[PassT]:
        return [
            AdjacentLoopMerging,
            HorizontalExecutionMerging,
            OnTheFlyMerging,
            LocalTemporariesToScalars,
            WriteBeforeReadTemporariesToScalars,
            MaskStmtMerging,
            MaskInlining,
            UnreachableStmtPruning,
            NoFieldAccessPruning,
            IJCacheDetection,
            KCacheDetection,
            PruneKCacheFills,
            PruneKCacheFlushes,
        ]

    @property
    def steps(self) -> Sequence[PassT]:
        return [step for step in self.all_steps() if step not in self.skip] + self.add_steps

    def __hash__(self) -> int:
        return hash(repr(self))

    def __repr__(self) -> str:
        return str([step.__name__ for step in self.steps])

    def __eq__(self, other):
        return isinstance(other, DefaultPipeline) and self.skip == other.skip

    def run(self, oir: oir.Stencil) -> oir.Stencil:
        for step in self.steps:
            if isinstance(step, type) and issubclass(step, NodeVisitor):
                oir = step().visit(oir)
            else:
                oir = step(oir)
        return oir
