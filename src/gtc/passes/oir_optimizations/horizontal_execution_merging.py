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

from eve import NodeTranslator, SymbolTableTrait
from gtc import common, oir

from .utils import AccessCollector, symbol_name_creator


@dataclass
class OnTheFlyMerging(NodeTranslator):
    """Merges consecutive horizontal executions inside parallel vertical loops by introducing redundant computations.

    Limitations:
    * Works on the level of whole horizontal executions, no full dependency analysis is performed (common subexpression and dead code eliminitation at a later stage can work around this limitation).
    * The chosen default merge limits are totally arbitrary.
    """

    max_horizontal_execution_body_size: int = 100
    allow_expensive_function_duplication: bool = False
    contexts = (SymbolTableTrait.add_symtable,)

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
        protected_fields: Set[str],
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

        def first_has_large_body() -> bool:
            return len(first.body) > self.max_horizontal_execution_body_size

        def first_writes_protected() -> bool:
            return bool(protected_fields & first_accesses.write_fields())

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
            first_fields_rewritten_later()
            or first_writes_protected()
            or first_has_large_body()
            or first_has_expensive_function_call()
        ):
            return [first] + self._merge(others, symtable, new_symbol_name, protected_fields)

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
                declarations=horizontal_execution.declarations
                + [
                    oir.LocalScalar(name=new_name, dtype=symtable[old_name].dtype)
                    for (old_name, _), new_name in offset_symbol_map.items()
                ]
                + [d for d in first.declarations if d not in horizontal_execution.declarations],
            )
            for offset in read_offsets:
                merged.body = (
                    self.visit(
                        first.body,
                        shift=offset,
                        offset_symbol_map=offset_symbol_map,
                        symtable=symtable,
                    )
                    + merged.body
                )
            others_otf.append(merged)

        return self._merge(others_otf, symtable, new_symbol_name, protected_fields)

    def visit_VerticalLoopSection(
        self, node: oir.VerticalLoopSection, **kwargs: Any
    ) -> oir.VerticalLoopSection:

        last_vls = None
        next_vls = node
        applied = True
        while applied:
            last_vls = next_vls
            next_vls = oir.VerticalLoopSection(
                interval=last_vls.interval,
                horizontal_executions=self._merge(last_vls.horizontal_executions, **kwargs),
            )
            applied = len(next_vls.horizontal_executions) < len(last_vls.horizontal_executions)

        return next_vls

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
        vertical_loops: List[oir.VerticalLoop] = []
        protected_fields = set(n.name for n in node.params)
        for vl in reversed(node.vertical_loops):
            vertical_loops.insert(
                0,
                self.visit(
                    vl,
                    new_symbol_name=symbol_name_creator(set(kwargs["symtable"])),
                    protected_fields=protected_fields,
                    **kwargs,
                ),
            )
            access_collection = AccessCollector.apply(vl)
            protected_fields |= access_collection.fields()
        accessed = AccessCollector.apply(vertical_loops).fields()
        return oir.Stencil(
            name=node.name,
            params=node.params,
            vertical_loops=vertical_loops,
            declarations=[d for d in node.declarations if d.name in accessed],
        )
