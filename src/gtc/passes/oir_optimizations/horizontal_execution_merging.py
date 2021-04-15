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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from eve import NodeTranslator
from gtc import common, oir
from gtc.common import GTCPostconditionError, GTCPreconditionError

from .utils import AccessCollector, symbol_name_creator


class GreedyMerging(NodeTranslator):
    """Merges consecutive horizontal executions if there are no write/read conflicts.

    Preconditions: All vertical loops are non-empty.
    Postcondition: The number of horizontal executions is equal or smaller than before.
    """

    def visit_VerticalLoopSection(
        self, node: oir.VerticalLoopSection, **kwargs: Any
    ) -> oir.VerticalLoopSection:
        if not node.horizontal_executions:
            raise GTCPreconditionError(expected="non-empty vertical loop")
        result = self.generic_visit(node, **kwargs)
        horizontal_executions = [result.horizontal_executions[0]]
        accesses = AccessCollector.apply(horizontal_executions[-1])

        def ij_offsets(
            offsets: Dict[str, Set[Tuple[int, int, int]]]
        ) -> Dict[str, Set[Tuple[int, int]]]:
            return {
                field: {o[:2] for o in field_offsets} for field, field_offsets in offsets.items()
            }

        previous_reads = ij_offsets(accesses.read_offsets())
        previous_writes = ij_offsets(accesses.write_offsets())
        for horizontal_execution in result.horizontal_executions[1:]:
            accesses = AccessCollector.apply(horizontal_execution)
            current_reads = ij_offsets(accesses.read_offsets())
            current_writes = ij_offsets(accesses.write_offsets())

            conflicting = {
                field
                for field, offsets in current_reads.items()
                if field in previous_writes and offsets ^ previous_writes[field]
            } | {
                field
                for field, offsets in current_writes.items()
                if field in previous_reads
                and any(o[:2] != (0, 0) for o in offsets ^ previous_reads[field])
            }
            if not conflicting and horizontal_execution.mask == horizontal_executions[-1].mask:
                horizontal_executions[-1].body += horizontal_execution.body
                for field, writes in current_writes.items():
                    previous_writes.setdefault(field, set()).update(writes)
                for field, reads in current_reads.items():
                    previous_reads.setdefault(field, set()).update(reads)
            else:
                horizontal_executions.append(horizontal_execution)
                previous_writes = current_writes
                previous_reads = current_reads
        result.horizontal_executions = horizontal_executions
        if len(result.horizontal_executions) > len(node.horizontal_executions):
            raise GTCPostconditionError(
                expected="the number of horizontal executions is equal or smaller than before"
            )
        return result


@dataclass
class OnTheFlyMerging(NodeTranslator):
    """Merges consecutive horizontal executions inside parallel vertical loops by introducing redundant computations.

    Limitations:
    * Works on the level of whole horizontal executions, no full dependency analysis is performed (common subexpression and dead code eliminitation at a later stage can work around this limitation).
    * The chosen default merge limits are totally arbitrary.
    """

    max_horizontal_execution_body_size: int = 100
    allow_expensive_function_duplication: bool = False

    def visit_CartesianOffset(
        self,
        node: common.CartesianOffset,
        *,
        shift: Optional[Tuple[int, int, int]] = None,
        **kwargs: Any,
    ) -> common.CartesianOffset:
        if shift:
            di, dj, dk = shift
            return common.CartesianOffset(i=node.i + di, j=node.j + dj, k=node.k + dk)
        return self.generic_visit(node, **kwargs)

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        offset_symbol_map: Dict[Tuple[str, Tuple[int, int, int]], str] = None,
        **kwargs: Any,
    ) -> Union[oir.FieldAccess, oir.ScalarAccess]:
        if offset_symbol_map:
            offset = self.visit(node.offset, **kwargs)
            key = node.name, (offset.i, offset.j, offset.k)
            if key in offset_symbol_map:
                return oir.ScalarAccess(name=offset_symbol_map[key], dtype=node.dtype)
        return self.generic_visit(node, **kwargs)

    def _merge(
        self,
        horizontal_executions: List[oir.HorizontalExecution],
        symtable: Dict[str, Any],
        new_symbol_name: Callable[[str], str],
    ) -> List[oir.HorizontalExecution]:
        """Recursively merge horizontal executions.

        Uses the following algorithm:
        1. Get output fields of the first horizontal execution.
        2. Check in which following h. execs. the outputs are read.
        3. Duplicate the body of the first h. exec. for each read access (with corresponding offset) and prepend it to the depending h. execs.
        4. Recurse on the resulting h. execs.
        """
        if len(horizontal_executions) <= 1:
            return horizontal_executions
        first, *others = horizontal_executions
        first_accesses = AccessCollector.apply(first)
        other_accesses = AccessCollector.apply(others)

        def first_fields_rewritten_later() -> bool:
            return bool(first_accesses.fields() & other_accesses.write_fields())

        def first_output_used_in_later_mask() -> bool:
            empty_str_set: Set[str] = set()
            return bool(
                first_accesses.write_fields()
                & empty_str_set.union(
                    *(AccessCollector.apply(o.mask, is_write=False).fields() for o in others)
                )
            )

        def first_has_large_body() -> bool:
            return len(first.body) > self.max_horizontal_execution_body_size

        def first_has_expensive_function_call() -> bool:
            if self.allow_expensive_function_duplication:
                return False
            nf = common.NativeFunction
            expensive_calls = {
                nf.SIN,
                nf.COS,
                nf.TAN,
                nf.ARCSIN,
                nf.ARCCOS,
                nf.ARCTAN,
                nf.SQRT,
                nf.EXP,
                nf.LOG,
            }
            calls = first.iter_tree().if_isinstance(oir.NativeFuncCall).getattr("func")
            return any(call in expensive_calls for call in calls)

        if (
            first.mask is not None
            or first_fields_rewritten_later()
            or first_output_used_in_later_mask()
            or first_has_large_body()
            or first_has_expensive_function_call()
        ):
            return [first] + self._merge(others, symtable, new_symbol_name)

        writes = first_accesses.write_fields()
        others_otf = []
        for horizontal_execution in others:
            read_offsets: Set[Tuple[int, int, int]] = set()
            read_offsets = read_offsets.union(
                *(
                    offsets
                    for field, offsets in AccessCollector.apply(horizontal_execution)
                    .read_offsets()
                    .items()
                    if field in writes
                )
            )

            if not read_offsets:
                others_otf.append(horizontal_execution)
                continue

            offset_symbol_map = {
                (name, o): new_symbol_name(name) for name in writes for o in read_offsets
            }

            merged = oir.HorizontalExecution(
                body=self.visit(horizontal_execution.body, offset_symbol_map=offset_symbol_map),
                mask=self.visit(horizontal_execution.mask, offset_symbol_map=offset_symbol_map),
                declarations=horizontal_execution.declarations
                + [
                    oir.LocalScalar(name=new_name, dtype=symtable[old_name].dtype)
                    for (old_name, _), new_name in offset_symbol_map.items()
                ]
                + [d for d in first.declarations if d not in horizontal_execution.declarations],
            )
            for offset in read_offsets:
                merged.body = (
                    self.visit(first.body, shift=offset, offset_symbol_map=offset_symbol_map)
                    + merged.body
                )
            others_otf.append(merged)

        return self._merge(others_otf, symtable, new_symbol_name)

    def visit_VerticalLoopSection(
        self, node: oir.VerticalLoopSection, **kwargs: Any
    ) -> oir.VerticalLoopSection:
        return oir.VerticalLoopSection(
            interval=node.interval,
            horizontal_executions=self._merge(node.horizontal_executions, **kwargs),
        )

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> oir.VerticalLoop:
        if node.loop_order != common.LoopOrder.PARALLEL:
            return node
        sections = self.visit(node.sections, **kwargs)
        accessed = AccessCollector.apply(sections).fields()
        return oir.VerticalLoop(
            loop_order=node.loop_order,
            sections=sections,
            caches=[c for c in node.caches if c.name in accessed],
        )

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        vertical_loops = self.visit(
            node.vertical_loops,
            symtable=node.symtable_,
            new_symbol_name=symbol_name_creator(set(node.symtable_)),
            **kwargs,
        )
        accessed = AccessCollector.apply(vertical_loops).fields()
        return oir.Stencil(
            name=node.name,
            params=node.params,
            vertical_loops=vertical_loops,
            declarations=[d for d in node.declarations if d.name in accessed],
        )
