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

from typing import Any, Callable, Dict, Iterable, List, Set, Tuple

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


class _ToLocalKCachesBase(NodeTranslator):
    def visit_FieldAccess(
        self, node: oir.FieldAccess, *, name_map: Dict[str, str], **kwargs: Any
    ) -> oir.FieldAccess:
        if node.name in name_map:
            return oir.FieldAccess(name=name_map[node.name], dtype=node.dtype, offset=node.offset)
        return node

    def visit_VerticalLoopSection(
        self, node: oir.VerticalLoopSection, **kwargs: Any
    ) -> oir.VerticalLoopSection:
        if len(node.horizontal_executions) > 1:
            raise NotImplementedError("Multiple horizontal_executions are not supported")
        return self.generic_visit(node, **kwargs)

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


class FillToLocalKCaches(_ToLocalKCachesBase):
    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        name_map: Dict[str, str],
        fills: List[oir.Stmt],
        **kwargs: Any,
    ) -> oir.HorizontalExecution:
        return oir.HorizontalExecution(
            body=fills + self.visit(node.body, name_map=name_map, **kwargs),
            mask=self.visit(node.mask, name_map=name_map, **kwargs),
            declarations=node.declarations,
        )

    @staticmethod
    def _fill_limits(
        loop_order: common.LoopOrder, section: oir.VerticalLoopSection
    ) -> Dict[str, Tuple[int, int]]:
        """Direction-normalized min and max read accesses for each accessed field.

        Args:
            loop_order: forward or backward order.
            section: loop section to split.

        Returns:
            A dict, mapping field names to min and max read offsets relative to loop order (i.e., positive means in the direction of the loop order).

        """

        def directional_k_offset(offset: Tuple[int, int, int]) -> int:
            """Positive k-offset for forward loops, negative for backward."""
            return offset[2] if loop_order == common.LoopOrder.FORWARD else -offset[2]

        read_offsets = AccessCollector.apply(section).read_offsets()
        return {
            field: (
                min(directional_k_offset(o) for o in offsets),
                max(directional_k_offset(o) for o in offsets),
            )
            for field, offsets in read_offsets.items()
        }

    @staticmethod
    def _requires_splitting(interval: oir.Interval) -> bool:
        """Check if an interval is larger than one level and thus may require splitting."""
        if interval.start.level != interval.end.level:
            return True
        return abs(interval.end.offset - interval.start.offset) > 1

    @staticmethod
    def _split_entry_level(
        loop_order: common.LoopOrder, section: oir.VerticalLoopSection
    ) -> Tuple[oir.VerticalLoopSection, oir.VerticalLoopSection]:
        """Split the entry level of a loop section.

        Args:
            loop_order: forward or backward order.
            section: loop section to split.

        Returns:
            Two loop sections.
        """
        assert loop_order in (common.LoopOrder.FORWARD, common.LoopOrder.BACKWARD)
        if loop_order == common.LoopOrder.FORWARD:
            bound = common.AxisBound(
                level=section.interval.start.level, offset=section.interval.start.offset + 1
            )
            entry_interval = oir.Interval(start=section.interval.start, end=bound)
            rest_interval = oir.Interval(start=bound, end=section.interval.end)
        else:
            bound = common.AxisBound(
                level=section.interval.end.level, offset=section.interval.end.offset - 1
            )
            entry_interval = oir.Interval(start=bound, end=section.interval.end)
            rest_interval = oir.Interval(start=section.interval.start, end=bound)
        return (
            oir.VerticalLoopSection(
                interval=entry_interval, horizontal_executions=section.horizontal_executions
            ),
            oir.VerticalLoopSection(
                interval=rest_interval, horizontal_executions=section.horizontal_executions
            ),
        )

    @classmethod
    def _split_section_with_multiple_fills(
        cls,
        loop_order: common.LoopOrder,
        section: oir.VerticalLoopSection,
        filling_fields: Iterable[str],
        first_unfilled: Dict[str, int],
    ) -> Tuple[Tuple[oir.VerticalLoopSection, ...], Dict[str, int]]:
        """Split loop sections that require multiple fills.

        Args:
            loop_order: forward or backward order.
            section: loop section to split.
            filling_fields: fields that are using fill caches.
            first_unfilled: direction-normalized offset of the first unfilled cache entry for each field.

        Returns:
            A list of sections and an updated `first_unfilled` map.
        """
        fill_limits = cls._fill_limits(loop_order, section)
        max_required_fills = 0
        for field in filling_fields:
            lmin, lmax = fill_limits[field]
            required_fills = lmax + 1 - max(lmin, first_unfilled.get(field, lmin))
            max_required_fills = max(required_fills, max_required_fills)
            first_unfilled[field] = lmax
        if max_required_fills > 1 and cls._requires_splitting(section.interval):
            return cls._split_entry_level(loop_order, section), first_unfilled
        return (section,), first_unfilled

    @classmethod
    def _fill_stmts(
        cls,
        loop_order: common.LoopOrder,
        section: oir.VerticalLoopSection,
        filling_fields: Dict[str, str],
        first_unfilled: Dict[str, int],
        symtable: Dict[str, Any],
    ) -> Tuple[List[oir.AssignStmt], Dict[str, int]]:
        """Generate fill statements for the given loop section.

        Args:
            loop_order: forward or backward order.
            section: loop section to split.
            filling_fields: mapping from field names to cache names.
            first_unfilled: direction-normalized offset of the first unfilled cache entry for each field.

        Returns:
            A list of fill statements and an updated `first_unfilled` map.
        """
        fill_limits = cls._fill_limits(loop_order, section)
        fill_stmts = []
        for field, cache in filling_fields.items():
            lmin, lmax = fill_limits[field]
            lmin = max(lmin, first_unfilled.get(field, lmin))
            for offset in range(lmin, lmax + 1):
                fill_stmts.append(
                    oir.AssignStmt(
                        left=oir.FieldAccess(
                            name=cache,
                            dtype=symtable[field].dtype,
                            offset=common.CartesianOffset.zero(),
                        ),
                        right=oir.FieldAccess(
                            name=field,
                            dtype=symtable[field].dtype,
                            offset=common.CartesianOffset(
                                i=0,
                                j=0,
                                k=offset if loop_order == common.LoopOrder.FORWARD else -offset,
                            ),
                        ),
                    )
                )
            first_unfilled[field] = lmax
        return fill_stmts, first_unfilled

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
            str(c.name): new_symbol_name(c.name)
            for c in node.caches
            if isinstance(c, oir.KCache) and c.fill
        }

        if not filling_fields:
            return node

        # new temporaries used for caches, declarations are later added to stencil
        for field_name, tmp_name in filling_fields.items():
            new_tmps.append(oir.Temporary(name=tmp_name, dtype=symtable[field_name].dtype))

        # split sections where more than one fill operations are required at the entry level
        first_unfilled: Dict[str, int] = dict()
        split_sections: List[oir.VerticalLoopSection] = []
        for section in node.sections:
            split_section, previous_fills = self._split_section_with_multiple_fills(
                node.loop_order, section, filling_fields, first_unfilled
            )
            split_sections += split_section

        # generate cache filling statements
        first_unfilled = dict()
        sections = []
        for section in split_sections:
            fills, first_unfilled = self._fill_stmts(
                node.loop_order, section, filling_fields, first_unfilled, symtable
            )
            sections.append(self.visit(section, fills=fills, name_map=filling_fields, **kwargs))

        # replace fill cache declarations
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


class FlushToLocalKCaches(_ToLocalKCachesBase):
    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        name_map: Dict[str, str],
        flushes: List[oir.Stmt],
        **kwargs: Any,
    ) -> oir.HorizontalExecution:
        return oir.HorizontalExecution(
            body=self.visit(node.body, name_map=name_map, **kwargs) + flushes,
            mask=self.visit(node.mask, name_map=name_map, **kwargs),
            declarations=node.declarations,
        )

    @classmethod
    def _flush_stmts(
        cls,
        loop_order: common.LoopOrder,
        section: oir.VerticalLoopSection,
        flushing_fields: Dict[str, str],
        symtable: Dict[str, Any],
    ) -> List[oir.AssignStmt]:
        """Generate flush statements for the given loop section.

        Args:
            loop_order: forward or backward order.
            section: loop section to split.
            flushing_fields: mapping from field names to cache names.

        Returns:
            A list of flush statements.
        """
        write_fields = AccessCollector.apply(section).write_fields()
        flush_stmts = []
        for field, cache in flushing_fields.items():
            if field in write_fields:
                flush_stmts.append(
                    oir.AssignStmt(
                        left=oir.FieldAccess(
                            name=field,
                            dtype=symtable[field].dtype,
                            offset=common.CartesianOffset.zero(),
                        ),
                        right=oir.FieldAccess(
                            name=cache,
                            dtype=symtable[field].dtype,
                            offset=common.CartesianOffset.zero(),
                        ),
                    )
                )
        return flush_stmts

    def visit_VerticalLoop(
        self,
        node: oir.VerticalLoop,
        *,
        new_tmps: List[oir.Temporary],
        symtable: Dict[str, Any],
        new_symbol_name: Callable[[str], str],
        **kwargs: Any,
    ) -> oir.VerticalLoop:
        flushing_fields = {
            str(c.name): new_symbol_name(c.name)
            for c in node.caches
            if isinstance(c, oir.KCache) and c.flush
        }
        if not flushing_fields:
            return node

        for field_name, tmp_name in flushing_fields.items():
            new_tmps.append(oir.Temporary(name=tmp_name, dtype=symtable[field_name].dtype))

        sections = [
            self.visit(
                section,
                flushes=self._flush_stmts(node.loop_order, section, flushing_fields, symtable),
                name_map=flushing_fields,
                **kwargs,
            )
            for section in node.sections
        ]

        caches = (
            [c for c in node.caches if c.name not in flushing_fields]
            + [oir.KCache(name=f, fill=False, flush=False) for f in flushing_fields.values()]
            + [
                oir.KCache(name=c.name, fill=True, flush=False)
                for c in node.caches
                if c.name in flushing_fields and c.fill
            ]
        )

        return oir.VerticalLoop(loop_order=node.loop_order, sections=sections, caches=caches)
