# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable, ClassVar, Dict, Optional, Sequence, Tuple

from gt4py.cartesian.definitions import StencilID
from gt4py.cartesian.gtc import gtir
from gt4py.cartesian.gtc.passes.gtir_definitive_assignment_analysis import (
    check as check_assignments,
)
from gt4py.cartesian.gtc.passes.gtir_dtype_resolver import resolve_dtype
from gt4py.cartesian.gtc.passes.gtir_prune_unused_parameters import prune_unused_parameters
from gt4py.cartesian.gtc.passes.gtir_upcaster import upcast


PASS_T = Callable[[gtir.Stencil], gtir.Stencil]


class GtirPipeline:
    """
    GTIR passes pipeline runs passes in order and allows skipping.

    May only call existing passes and may not contain any pass logic itself.
    """

    # Cache pipelines across all instances
    _cache: ClassVar[Dict[Tuple[StencilID, Tuple[PASS_T, ...]], gtir.Stencil]] = {}

    def __init__(self, node: gtir.Stencil, stencil_id: StencilID):
        self.gtir = node
        self._stencil_id = stencil_id

    @property
    def stencil_id(self) -> StencilID:
        return self._stencil_id

    def steps(self) -> Sequence[PASS_T]:
        return [check_assignments, prune_unused_parameters, resolve_dtype, upcast]

    def apply(self, steps: Sequence[PASS_T]) -> gtir.Stencil:
        result = self.gtir
        for step in steps:
            result = step(result)
        return result

    def _get_cached(self, steps: Sequence[PASS_T]) -> Optional[gtir.Stencil]:
        return self._cache.get((self.stencil_id, tuple(steps)))

    def _set_cached(self, steps: Sequence[PASS_T], node: gtir.Stencil) -> gtir.Stencil:
        return self._cache.setdefault((self.stencil_id, tuple(steps)), node)

    def full(self, skip: Optional[Sequence[PASS_T]] = None) -> gtir.Stencil:
        skip = skip or []
        pipeline = [step for step in self.steps() if step not in skip]
        return self._get_cached(pipeline) or self._set_cached(pipeline, self.apply(pipeline))
