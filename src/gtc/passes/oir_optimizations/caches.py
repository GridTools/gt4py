# GTC Toolchain - GT4Py Project - GridTools Framework
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

import collections
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple

import eve
from gtc import common, oir

from .utils import AccessCollector, symbol_name_creator


"""Utilities for cache detection and modifications on the OIR level.

Caches in GridTools terminology (with roots in STELLA) are local replacements
of temporary fields, which are usually stored as full fields in global memory,
by smaller buffers with more reuse (and faster memory on GPUs) or by registers.

IJ-caches are used in purely horizontal k-parallel stencils and replace the
fully 3D temporary field by 2D buffers (in block-local shared memory on GPUs
and thread-local memory on CPUs).

K-caches are used in vertical sweeps, most notably tridiagonal solvers. They
cache values from vertical accesses in a ring buffer, that is shifted on each
loop iteration. K-cache values can optionally be filled with values from global
memory or flushed to global memory, then acting like normal temporary fields,
but only requiring one read operation per loop iteration, independent of the
number of vertical accesses. In cases, where a field is always written before
read, the fill is not required. Similar, if a field is not read later, the
flushes can be omitted. When fills and flushes are omitted, all field accesses
are replaced by just a few registers.

Note that filling and flushing k-caches can always be replaced by a local
(non-filling or flushing) k-cache plus additional filling and flushing
statements.

"""


class IJCacheDetection(eve.NodeTranslator):
    def visit_VerticalLoop(
        self, node: oir.VerticalLoop, *, local_tmps: Set[str], **kwargs: Any
    ) -> oir.VerticalLoop:
        if node.loop_order != common.LoopOrder.PARALLEL or not local_tmps:
            return node

        def already_cached(field: str) -> bool:
            return any(c.name == field for c in node.caches)

        def has_vertical_offset(offsets: Set[Tuple[int, int, int]]) -> bool:
            return any(offset[2] != 0 for offset in offsets)

        accesses = AccessCollector.apply(node).cartesian_accesses().offsets()
        cacheable = {
            field
            for field, offsets in accesses.items()
            if field in local_tmps
            and not already_cached(field)
            and not has_vertical_offset(offsets)
        }
        caches = self.visit(node.caches, **kwargs) + [
            oir.IJCache(name=field) for field in cacheable
        ]
        return oir.VerticalLoop(
            sections=node.sections,
            loop_order=node.loop_order,
            caches=caches,
            loc=node.loc,
        )

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        vertical_loops = node.walk_values().if_isinstance(oir.VerticalLoop)
        counts: collections.Counter = sum(
            (
                collections.Counter(
                    vertical_loop.walk_values()
                    .if_isinstance(oir.FieldAccess)
                    .getattr("name")
                    .if_in({tmp.name for tmp in node.declarations})
                    .to_set()
                )
                for vertical_loop in vertical_loops
            ),
            collections.Counter(),
        )
        local_tmps = {tmp for tmp, count in counts.items() if count == 1}
        return self.generic_visit(node, local_tmps=local_tmps, **kwargs)


@dataclass
class KCacheDetection(eve.NodeTranslator):
    max_cacheable_offset: int = 5

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> oir.VerticalLoop:
        # k-caches are restricted to loops with a single horizontal region as all regions without
        # horizontal offsets should be merged before anyway and this restriction allows for easier
        # conversion of fill and flush caches to local caches later
        if node.loop_order == common.LoopOrder.PARALLEL or any(
            len(section.horizontal_executions) != 1 for section in node.sections
        ):
            return self.generic_visit(node, **kwargs)

        all_accesses = AccessCollector.apply(node)
        fields_with_variable_reads = {
            field
            for field, offsets in all_accesses.offsets().items()
            if any(off[2] is None for off in offsets)
        }

        def accessed_more_than_once(offsets: Set[Any]) -> bool:
            return len(offsets) > 1

        def already_cached(field: str) -> bool:
            return field in {c.name for c in node.caches}

        # TODO(fthaler): k-caches with non-zero ij offsets?
        def has_horizontal_offset(offsets: Set[Tuple[int, int, int]]) -> bool:
            return any(offset[:2] != (0, 0) for offset in offsets)

        def offsets_within_limits(offsets: Set[Tuple[int, int, int]]) -> bool:
            return all(abs(offset[2]) <= self.max_cacheable_offset for offset in offsets)

        def has_variable_offset_reads(field: str) -> bool:
            return field in fields_with_variable_reads

        accesses = all_accesses.cartesian_accesses().offsets()
        cacheable = {
            field
            for field, offsets in accesses.items()
            if not already_cached(field)
            and not has_variable_offset_reads(field)
            and accessed_more_than_once(offsets)
            and not has_horizontal_offset(offsets)
            and offsets_within_limits(offsets)
        }
        caches = self.visit(node.caches, **kwargs) + [
            oir.KCache(name=field, fill=True, flush=True) for field in cacheable
        ]
        return oir.VerticalLoop(
            loop_order=node.loop_order,
            sections=node.sections,
            caches=caches,
            loc=node.loc,
        )


class PruneKCacheFills(eve.NodeTranslator):
    """Prunes unneeded k-cache fills.

    A fill is classified as required if at least one of the two following conditions holds in any of the loop sections:
    * There is a read with offset in the direction of looping.
    * The first centered access is a read access.
    If none of the conditions holds for any loop section, the fill is considered as unneeded.
    """

    def visit_KCache(self, node: oir.KCache, *, pruneable: Set[str], **kwargs: Any) -> oir.KCache:
        if node.name in pruneable:
            return oir.KCache(name=node.name, fill=False, flush=node.flush)
        return self.generic_visit(node, **kwargs)

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> oir.VerticalLoop:
        filling_fields = {c.name for c in node.caches if isinstance(c, oir.KCache) and c.fill}
        if not filling_fields:
            return self.generic_visit(node, **kwargs)
        assert node.loop_order != common.LoopOrder.PARALLEL

        def pruneable_fields(section: oir.VerticalLoopSection) -> Set[str]:
            accesses = AccessCollector.apply(section).cartesian_accesses()
            offsets = accesses.offsets()
            center_accesses = [a for a in accesses.ordered_accesses() if a.offset == (0, 0, 0)]

            def requires_fill(field: str) -> bool:
                if field not in offsets:
                    return False
                k_offsets = (o[2] for o in offsets[field])
                if node.loop_order == common.LoopOrder.FORWARD and max(k_offsets) > 0:
                    return True
                if node.loop_order == common.LoopOrder.BACKWARD and min(k_offsets) < 0:
                    return True
                try:
                    return next(a.is_read for a in center_accesses if a.field == field)
                except StopIteration:
                    return False

            return {field for field in filling_fields if not requires_fill(field)}

        pruneable = set.intersection(*(pruneable_fields(section) for section in node.sections))

        # Consider accesses outside of loop interval
        # Note: those accesses would not require a fill in the whole loop interval in in theory
        # but restriction to the GridTools k-cache types makes this necessary
        # E.g. consider the following loop with k = 1, 2:
        # k | no fill required | initial fill required |
        # - | ---------------- | --------------------- |
        # 1 | a[k] = init      | a[k] = a[k - 1]       |
        # 2 | a[k] = a[k - 1]  | a[k] = a[k - 1]       |
        # In the first case, no fill is required. In the second case, a[0] has to be filled on
        # level k = 1. On level k = 2, no fill would be necessary, but GridTools C++ does not
        # support a k-cache type that only fills on the initial level, so fills are inserted on all
        # levels here.

        first_section_offsets = (
            AccessCollector.apply(node.sections[0]).cartesian_accesses().offsets()
        )
        last_section_offsets = (
            AccessCollector.apply(node.sections[-1]).cartesian_accesses().offsets()
        )
        for field in list(pruneable):
            first_k_offsets = (o[2] for o in first_section_offsets.get(field, {(0, 0, 0)}))
            last_k_offsets = (o[2] for o in last_section_offsets.get(field, {(0, 0, 0)}))
            if node.loop_order == common.LoopOrder.BACKWARD:
                first_k_offsets, last_k_offsets = last_k_offsets, first_k_offsets
            if min(first_k_offsets) < 0 or max(last_k_offsets) > 0:
                pruneable.remove(field)

        return self.generic_visit(node, pruneable=pruneable, **kwargs)


class PruneKCacheFlushes(eve.NodeTranslator):
    """Prunes unneeded k-cache flushes.

    A flush is classified as unneeded under the following conditions:
    * All accesses to the field are read-only in the current vertical loop.
    * There are no read accesses to the field in a following loop.
    """

    def visit_KCache(self, node: oir.KCache, *, pruneable: Set[str], **kwargs: Any) -> oir.KCache:
        if node.name in pruneable:
            return oir.KCache(name=node.name, fill=node.fill, flush=False, loc=node.loc)
        return self.generic_visit(node, **kwargs)

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        accesses = [AccessCollector.apply(vertical_loop) for vertical_loop in node.vertical_loops]
        vertical_loops = []
        for i, vertical_loop in enumerate(node.vertical_loops):
            flushing_fields = {
                str(c.name) for c in vertical_loop.caches if isinstance(c, oir.KCache) and c.flush
            }
            read_only_fields = flushing_fields & (
                accesses[i].read_fields() - accesses[i].write_fields()
            )
            future_reads: Set[str] = set()
            future_reads = future_reads.union(*(acc.read_fields() for acc in accesses[i + 1 :]))
            tmps_without_reuse = (
                flushing_fields & {str(d.name) for d in node.declarations}
            ) - future_reads
            pruneable = read_only_fields | tmps_without_reuse
            vertical_loops.append(self.visit(vertical_loop, pruneable=pruneable, **kwargs))
        return oir.Stencil(
            name=node.name,
            params=self.visit(node.params, **kwargs),
            vertical_loops=vertical_loops,
            declarations=node.declarations,
            loc=node.loc,
        )


class FillFlushToLocalKCaches(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    """Converts fill and flush k-caches to local k-caches.

    For each cached field, the following actions are performed:
    1. A new locally-k-cached temporary is introduced.
    2. All accesses to the original field are replaced by accesses to this temporary.
    3. Loop sections are split where necessary to allow single-level loads whereever possible.
    3. Fill statements from the original field to the temporary are introduced.
    4. Flush statements from the temporary to the original field are introduced.
    """

    def visit_FieldAccess(
        self, node: oir.FieldAccess, *, name_map: Dict[str, str], **kwargs: Any
    ) -> oir.FieldAccess:
        if node.name in name_map:
            return oir.FieldAccess(
                name=name_map[node.name],
                data_index=node.data_index,
                dtype=node.dtype,
                offset=node.offset,
                loc=node.loc,
            )
        return node

    def visit_VerticalLoopSection(
        self, node: oir.VerticalLoopSection, **kwargs: Any
    ) -> oir.VerticalLoopSection:
        if len(node.horizontal_executions) > 1:
            raise NotImplementedError("Multiple horizontal_executions are not supported")
        return self.generic_visit(node, **kwargs)

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        name_map: Dict[str, str],
        fills: List[oir.Stmt],
        flushes: List[oir.Stmt],
        **kwargs: Any,
    ) -> oir.HorizontalExecution:
        return oir.HorizontalExecution(
            body=fills + self.visit(node.body, name_map=name_map, **kwargs) + flushes,
            declarations=node.declarations,
            loc=node.loc,
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

        read_offsets = AccessCollector.apply(section).cartesian_accesses().read_offsets()
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
        loop_order: common.LoopOrder,
        section: oir.VerticalLoopSection,
        new_symbol_name: Callable[[str], str],
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
        decls = list(section.walk_values().if_isinstance(oir.Decl))
        decls_map = {decl.name: new_symbol_name(decl.name) for decl in decls}

        class FixSymbolNameClashes(eve.NodeTranslator):
            def visit_ScalarAccess(self, node: oir.ScalarAccess) -> oir.ScalarAccess:
                if node.name not in decls_map:
                    return node
                return oir.ScalarAccess(name=decls_map[node.name], dtype=node.dtype)

            def visit_LocalScalar(self, node: oir.LocalScalar) -> oir.LocalScalar:
                return oir.LocalScalar(name=decls_map[node.name], dtype=node.dtype)

        return (
            oir.VerticalLoopSection(
                interval=entry_interval,
                horizontal_executions=FixSymbolNameClashes().visit(section.horizontal_executions),
                loc=section.loc,
            ),
            oir.VerticalLoopSection(
                interval=rest_interval,
                horizontal_executions=section.horizontal_executions,
                loc=section.loc,
            ),
        )

    @classmethod
    def _split_section_with_multiple_fills(
        cls,
        loop_order: common.LoopOrder,
        section: oir.VerticalLoopSection,
        filling_fields: Iterable[str],
        first_unfilled: Dict[str, int],
        new_symbol_name: Callable[[str], str],
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
            lmin, lmax = fill_limits.get(field, (0, 0))
            required_fills = lmax + 1 - max(lmin, first_unfilled.get(field, lmin))
            max_required_fills = max(required_fills, max_required_fills)
            first_unfilled[field] = lmax
        if max_required_fills > 1 and cls._requires_splitting(section.interval):
            return cls._split_entry_level(loop_order, section, new_symbol_name), first_unfilled
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
            lmin, lmax = fill_limits.get(field, (0, 0))
            lmin = max(lmin, first_unfilled.get(field, lmin))
            for offset in range(lmin, lmax + 1):
                k_offset = common.CartesianOffset(
                    i=0,
                    j=0,
                    k=offset if loop_order == common.LoopOrder.FORWARD else -offset,
                )
                fill_stmts.append(
                    oir.AssignStmt(
                        left=oir.FieldAccess(
                            name=cache, dtype=symtable[field].dtype, offset=k_offset
                        ),
                        right=oir.FieldAccess(
                            name=field, dtype=symtable[field].dtype, offset=k_offset
                        ),
                    )
                )
            first_unfilled[field] = lmax
        return fill_stmts, first_unfilled

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
        filling_fields: Dict[str, str] = {
            c.name: new_symbol_name(c.name)
            for c in node.caches
            if isinstance(c, oir.KCache) and c.fill
        }
        flushing_fields: Dict[str, str] = {
            c.name: filling_fields[c.name] if c.name in filling_fields else new_symbol_name(c.name)
            for c in node.caches
            if isinstance(c, oir.KCache) and c.flush
        }

        filling_or_flushing_fields = dict(
            set(filling_fields.items()) | set(flushing_fields.items())
        )

        if not filling_or_flushing_fields:
            return node

        # new temporaries used for caches, declarations are later added to stencil
        for field_name, tmp_name in filling_or_flushing_fields.items():
            new_tmps.append(
                oir.Temporary(
                    name=tmp_name, dtype=symtable[field_name].dtype, dimensions=(True, True, True)
                )
            )

        if filling_fields:
            # split sections where more than one fill operations are required at the entry level
            first_unfilled: Dict[str, int] = dict()
            split_sections: List[oir.VerticalLoopSection] = []
            for section in node.sections:
                split_section, previous_fills = self._split_section_with_multiple_fills(
                    node.loop_order, section, filling_fields, first_unfilled, new_symbol_name
                )
                split_sections += split_section
        else:
            split_sections = node.sections

        # generate cache fill and flush statements
        first_unfilled = dict()
        sections = []
        for section in split_sections:
            fills, first_unfilled = self._fill_stmts(
                node.loop_order, section, filling_fields, first_unfilled, symtable
            )
            flushes = self._flush_stmts(node.loop_order, section, flushing_fields, symtable)
            sections.append(
                self.visit(
                    section,
                    fills=fills,
                    flushes=flushes,
                    name_map=filling_or_flushing_fields,
                    symtable=symtable,
                    **kwargs,
                )
            )

        # replace cache declarations
        caches = [c for c in node.caches if c.name not in filling_or_flushing_fields] + [
            oir.KCache(name=f, fill=False, flush=False) for f in filling_or_flushing_fields.values()
        ]

        return oir.VerticalLoop(
            loop_order=node.loop_order,
            sections=sections,
            caches=caches,
            loc=node.loc,
        )

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        new_tmps: List[oir.Temporary] = []
        return oir.Stencil(
            name=node.name,
            params=node.params,
            vertical_loops=self.visit(
                node.vertical_loops,
                new_tmps=new_tmps,
                new_symbol_name=symbol_name_creator(set(kwargs["symtable"])),
                **kwargs,
            ),
            declarations=node.declarations + new_tmps,
            loc=node.loc,
        )
