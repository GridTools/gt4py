from typing import Callable, Dict, Optional, Sequence, Tuple

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


PASS_T = Callable[[oir.Stencil], oir.Stencil]


class OirPipeline:
    """
    OIR passes pipeline runs passes in order and allows skipping.

    May only call existing passes and may not contain any pass logic itself.
    """

    def __init__(self, node: oir.Stencil):
        self.oir = node
        self._cache: Dict[Tuple[PASS_T, ...], oir.Stencil] = {}

    def steps(self) -> Sequence[PASS_T]:
        return [
            graph_merge_horizontal_executions,
            GreedyMerging().visit,
            AdjacentLoopMerging().visit,
            LocalTemporariesToScalars().visit,
            WriteBeforeReadTemporariesToScalars().visit,
            OnTheFlyMerging().visit,
            MaskStmtMerging().visit,
            MaskInlining().visit,
            NoFieldAccessPruning().visit,
            IJCacheDetection().visit,
            KCacheDetection().visit,
            PruneKCacheFills().visit,
            PruneKCacheFlushes().visit,
            FillFlushToLocalKCaches().visit,
        ]

    def apply(self, steps: Sequence[PASS_T]) -> oir.Stencil:
        result = self.oir
        for step in steps:
            result = step(result)
        return result

    def _get_cached(self, steps: Sequence[PASS_T]) -> Optional[oir.Stencil]:
        return self._cache.get(tuple(steps))

    def _set_cached(self, steps: Sequence[PASS_T], node: oir.Stencil) -> oir.Stencil:
        return self._cache.setdefault(tuple(steps), node)

    def full(self, skip: Sequence[PASS_T] = None) -> oir.Stencil:
        skip = skip or []
        pipeline = [step for step in self.steps() if step not in skip]
        return self._get_cached(pipeline) or self._set_cached(pipeline, self.apply(pipeline))
