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

from typing import Any, List, Set

from eve import NodeTranslator
from gtc import common, oir

from .utils import AccessCollector


class IJCacheDetection(NodeTranslator):
    def visit_VerticalLoop(
        self, node: oir.VerticalLoop, *, temporaries: List[oir.Temporary], **kwargs: Any
    ) -> oir.VerticalLoop:
        if node.loop_order != common.LoopOrder.PARALLEL:
            return self.generic_visit(node, **kwargs)
        accesses = AccessCollector.apply(node).offsets()
        cacheable = {
            field
            for field, offsets in accesses.items()
            if field in {tmp.name for tmp in temporaries}
            and field not in {c.name for c in node.caches}
            and all(o[2] == 0 for o in offsets)
        }
        caches = self.visit(node.caches, **kwargs) + [
            oir.IJCache(name=field) for field in cacheable
        ]
        return oir.VerticalLoop(
            sections=node.sections,
            loop_order=node.loop_order,
            caches=caches,
        )

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        return self.generic_visit(node, temporaries=node.declarations, **kwargs)


class KCacheDetection(NodeTranslator):
    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> oir.VerticalLoop:
        if node.loop_order == common.LoopOrder.PARALLEL:
            return self.generic_visit(node, **kwargs)
        accesses = AccessCollector.apply(node).offsets()
        # TODO: k-caches with non-zero ij offsets?
        cacheable = {
            field
            for field, offsets in accesses.items()
            if field not in {c.name for c in node.caches}
            and len(offsets) > 1
            and all(o[:2] == (0, 0) for o in offsets)
        }
        caches = self.visit(node.caches, **kwargs) + [
            oir.KCache(name=field, fill=True, flush=True) for field in cacheable
        ]
        return oir.VerticalLoop(
            loop_order=node.loop_order,
            sections=node.sections,
            caches=caches,
        )


class PruneKCacheFills(NodeTranslator):
    def visit_KCache(self, node: oir.KCache, *, pruneable: Set[str], **kwargs: Any) -> oir.KCache:
        if node.name in pruneable:
            return oir.KCache(name=node.name, fill=False, flush=node.flush)
        return self.generic_visit(node, **kwargs)

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> oir.VerticalLoop:
        filling_fields = {c.name for c in node.caches if isinstance(c, oir.KCache) and c.fill}
        if not filling_fields:
            return self.generic_visit(node, **kwargs)
        assert node.loop_order != common.LoopOrder.PARALLEL

        accesses = AccessCollector.apply(node)
        offsets = accesses.offsets()
        center_accesses = [a for a in accesses.ordered_accesses() if a.offset == (0, 0, 0)]

        def requires_fill(field: str) -> bool:
            k_offsets = (o[2] for o in offsets[field])
            if node.loop_order == common.LoopOrder.FORWARD and max(k_offsets) > 0:
                return True
            if node.loop_order == common.LoopOrder.BACKWARD and min(k_offsets) < 0:
                return True
            return next(a.is_read for a in center_accesses if a.field == field)

        pruneable = {field for field in filling_fields if not requires_fill(field)}

        return self.generic_visit(node, pruneable=pruneable, **kwargs)


class PruneKCacheFlushes(NodeTranslator):
    def visit_KCache(self, node: oir.KCache, *, pruneable: Set[str], **kwargs: Any) -> oir.KCache:
        if node.name in pruneable:
            return oir.KCache(name=node.name, fill=node.fill, flush=False)
        return self.generic_visit(node, **kwargs)

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.KCache:
        accesses = [AccessCollector.apply(vertical_loop) for vertical_loop in node.vertical_loops]
        vertical_loops = []
        for i, vertical_loop in enumerate(node.vertical_loops):
            flushing_fields = {
                c.name for c in vertical_loop.caches if isinstance(c, oir.KCache) and c.flush
            }
            read_only_fields = flushing_fields & (
                accesses[i].read_fields() - accesses[i].write_fields()
            )
            tmps_without_reuse = (
                flushing_fields & {d.name for d in node.declarations}
            ) - set().union(
                *(acc.read_fields() for acc in accesses[i + 1 :])  # type: ignore
            )
            pruneable = read_only_fields | tmps_without_reuse
            vertical_loops.append(self.visit(vertical_loop, pruneable=pruneable, **kwargs))
        return oir.Stencil(
            name=node.name,
            params=self.visit(node.params, **kwargs),
            vertical_loops=vertical_loops,
            declarations=node.declarations,
        )
