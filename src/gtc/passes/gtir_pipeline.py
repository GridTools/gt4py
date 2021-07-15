# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
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

from typing import Callable, Dict, Optional, Sequence, Tuple

from gtc import gtir
from gtc.passes.gtir_check_single_iteration import check_single_iteration
from gtc.passes.gtir_dtype_resolver import resolve_dtype
from gtc.passes.gtir_prune_unused_parameters import prune_unused_parameters
from gtc.passes.gtir_upcaster import upcast


PASS_T = Callable[[gtir.Stencil], gtir.Stencil]


class GtirPipeline:
    """
    GTIR passes pipeline runs passes in order and allows skipping.

    May only call existing passes and may not contain any pass logic itself.
    """

    def __init__(self, node: gtir.Stencil):
        self.gtir = node
        self._cache: Dict[Tuple[PASS_T, ...], gtir.Stencil] = {}

    def steps(self) -> Sequence[PASS_T]:
        return [prune_unused_parameters, resolve_dtype, upcast, check_single_iteration]

    def apply(self, steps: Sequence[PASS_T]) -> gtir.Stencil:
        result = self.gtir
        for step in steps:
            result = step(result)
        return result

    def _get_cached(self, steps: Sequence[PASS_T]) -> Optional[gtir.Stencil]:
        return self._cache.get(tuple(steps))

    def _set_cached(self, steps: Sequence[PASS_T], node: gtir.Stencil) -> gtir.Stencil:
        return self._cache.setdefault(tuple(steps), node)

    def full(self, skip: Sequence[PASS_T] = None) -> gtir.Stencil:
        skip = skip or []
        pipeline = [step for step in self.steps() if step not in skip]
        return self._get_cached(pipeline) or self._set_cached(pipeline, self.apply(pipeline))
