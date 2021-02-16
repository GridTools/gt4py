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

from typing import Any, Callable, Dict, List, Set, Tuple

from eve import NodeTranslator
from gtc import common, oir

from .utils import AccessCollector, symbol_name_creator


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
            future_reads: Set[str] = set()
            future_reads = future_reads.union(*(acc.read_fields() for acc in accesses[i + 1 :]))
            tmps_without_reuse = (
                flushing_fields & {d.name for d in node.declarations}
            ) - future_reads
            pruneable = read_only_fields | tmps_without_reuse
            vertical_loops.append(self.visit(vertical_loop, pruneable=pruneable, **kwargs))
        return oir.Stencil(
            name=node.name,
            params=self.visit(node.params, **kwargs),
            vertical_loops=vertical_loops,
            declarations=node.declarations,
        )


class FillToLocalKCaches(NodeTranslator):
    def visit_FieldAccess(
        self, node: oir.FieldAccess, *, filling_fields: Dict[str, str], **kwargs: Any
    ) -> oir.FieldAccess:
        if node.name in filling_fields:
            return oir.FieldAccess(
                name=filling_fields[node.name], dtype=node.dtype, offset=node.offset
            )
        return self.generic_visit(node, **kwargs)

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        filling_fields: Dict[str, str],
        fills: List[oir.Stmt],
        **kwargs: Any,
    ) -> oir.HorizontalExecution:
        return oir.HorizontalExecution(
            body=fills + self.visit(node.body, filling_fields=filling_fields, **kwargs),
            mask=self.visit(node.mask, filling_fields=filling_fields, **kwargs),
            declarations=node.declarations,
        )

    def visit_VerticalLoopSection(
        self, node: oir.VerticalLoopSection, **kwargs: Any
    ) -> oir.VerticalLoopSection:
        if len(node.horizontal_executions) > 1:
            raise NotImplementedError("Multiple horizontal_executions are not supported")
        return self.generic_visit(node, **kwargs)

    def visit_VerticalLoop(
        self,
        node: oir.VerticalLoop,
        *,
        new_tmps: List[oir.Temporary],
        symtable: Dict[str, Any],
        new_symbol_name: Callable[[str], str],
        **kwargs: Any,
    ) -> oir.VerticalLoop:
        filling_fields = {
            c.name: new_symbol_name(c.name)
            for c in node.caches
            if isinstance(c, oir.KCache) and c.fill
        }

        for field_name, tmp_name in filling_fields.items():
            new_tmps.append(oir.Temporary(name=tmp_name, dtype=symtable[field_name].dtype))

        def directional_k_offset(offset: Tuple[int, int, int]) -> int:
            i, j, k = offset
            if not i == j == 0:
                raise NotImplementedError("Requires zero horizontal offsets")
            return k if node.loop_order == common.LoopOrder.FORWARD else -k

        def fill_limits(n: oir.VerticalLoopSection) -> Dict[str, Tuple[int, int]]:
            return {
                field: (min(k_offs := {directional_k_offset(o) for o in offsets}), max(k_offs))
                for field, offsets in AccessCollector.apply(n).read_offsets().items()
                if field in filling_fields
            }

        def requires_splitting(interval: oir.Interval) -> bool:
            if interval.start.level != interval.end.level:
                return True
            return abs(interval.end.offset - interval.start.offset) > 1

        def split_entry_level(section: oir.VerticalLoopSection) -> List[oir.VerticalLoopSection]:
            if node.loop_order == common.LoopOrder.FORWARD:
                bound = common.AxisBound(
                    level=section.interval.start.level, offset=section.interval.start.offset + 1
                )
                entry_interval = oir.Interval(start=section.interval.start, end=bound)
                rest_interval = oir.Interval(start=bound, end=section.interval.end)
            else:
                bound = common.AxisBound(
                    level=section.interval.stop.level, offset=section.interval.stop.offset - 1
                )
                entry_interval = oir.Interval(start=bound, end=section.interval.end)
                rest_interval = oir.Interval(start=section.interval.start, end=bound)
            return [
                oir.VerticalLoopSection(
                    interval=entry_interval, horizontal_executions=section.horizontal_executions
                ),
                oir.VerticalLoopSection(
                    interval=rest_interval, horizontal_executions=section.horizontal_executions
                ),
            ]

        previous_fill = {field: -100000 for field in filling_fields}
        split_sections = []
        for section in node.sections:
            limits = fill_limits(section)
            max_required_fills = 0
            for field in filling_fields:
                lmin, lmax = limits[field]
                required_fills = lmax - max(lmin, previous_fill[field] + 1) + 1
                max_required_fills = max(required_fills, max_required_fills)
                previous_fill[field] = lmax - 1
            if max_required_fills > 1 and requires_splitting(section.interval):
                split_sections += split_entry_level(section)
            else:
                split_sections.append(section)

        previous_fill = {field: -100000 for field in filling_fields}
        sections = []
        for section in split_sections:
            limits = fill_limits(section)
            fills = []
            for field in filling_fields:
                lmin, lmax = limits[field]
                for offset in range(max(lmin, previous_fill[field] + 1), lmax + 1):
                    fills.append(
                        oir.AssignStmt(
                            left=oir.FieldAccess(
                                name=filling_fields[field],
                                dtype=symtable[field].dtype,
                                offset=common.CartesianOffset.zero(),
                            ),
                            right=oir.FieldAccess(
                                name=field,
                                dtype=symtable[field].dtype,
                                offset=common.CartesianOffset(
                                    i=0,
                                    j=0,
                                    k=offset
                                    if node.loop_order == common.LoopOrder.FORWARD
                                    else -offset,
                                ),
                            ),
                        )
                    )
                sections.append(
                    self.visit(section, fills=fills, filling_fields=filling_fields, **kwargs)
                )
                previous_fill[field] = lmax - 1

        caches = (
            [c for c in node.caches if c.name not in filling_fields]
            + [oir.KCache(name=f, fill=False, flush=False) for f in filling_fields.values()]
            + [
                oir.KCache(name=c.name, fill=False, flush=True)
                for c in node.caches
                if c.name in filling_fields and c.flush
            ]
        )

        return oir.VerticalLoop(loop_order=node.loop_order, sections=sections, caches=caches)

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        new_tmps: List[oir.Temporary] = []
        return oir.Stencil(
            name=node.name,
            params=node.params,
            vertical_loops=self.visit(
                node.vertical_loops,
                new_tmps=new_tmps,
                symtable=node.symtable_,
                new_symbol_name=symbol_name_creator(set(node.symtable_)),
            ),
            declarations=node.declarations + new_tmps,
        )
