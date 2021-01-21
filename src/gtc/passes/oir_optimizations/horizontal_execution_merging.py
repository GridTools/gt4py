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

from collections import defaultdict
from typing import Any, Dict, Set, Tuple

from eve import NodeTranslator, NodeVisitor
from gtc import oir


class ZeroOffsetMerging(NodeTranslator):
    """Merges horizontal executions with zero offsets with previous horizontal exectutions within the same vertical loop."""

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> oir.VerticalLoop:
        assert node.horizontal_executions, "non-empty vertical loop expected"
        result = self.generic_visit(node, **kwargs)
        horizontal_executions = [result.horizontal_executions[0]]
        for horizontal_execution in result.horizontal_executions[1:]:
            has_zero_extents = (
                horizontal_execution.iter_tree()
                .if_isinstance(oir.FieldAccess)
                .map(lambda x: x.offset.i == x.offset.j == 0)
                .reduce(lambda a, b: a and b, init=True)
            )
            if has_zero_extents and horizontal_execution.mask == horizontal_executions[-1].mask:
                horizontal_executions[-1].body += horizontal_execution.body
            else:
                horizontal_executions.append(horizontal_execution)
        result.horizontal_executions = horizontal_executions
        return result


class GreedyMerging(NodeTranslator):
    class AccessCollector(NodeVisitor):
        def visit_FieldAccess(
            self,
            node: oir.FieldAccess,
            *,
            accesses: Dict[str, Set[Tuple[int, int, int]]],
            **kwargs: Any,
        ) -> None:
            accesses[node.name].add((node.offset.i, node.offset.j, node.offset.k))

        def visit_AssignStmt(
            self,
            node: oir.AssignStmt,
            *,
            reads: Dict[str, Set[Tuple[int, int, int]]],
            writes: Dict[str, Set[Tuple[int, int, int]]],
            **kwargs: Any,
        ) -> None:
            self.visit(node.left, accesses=writes, **kwargs)
            self.visit(node.right, accesses=reads, **kwargs)

        def visit_HorizontalExecution(
            self, node: oir.HorizontalExecution, **kwargs: Any
        ) -> Tuple[Dict[str, Set[Tuple[int, int, int]]], Dict[str, Set[Tuple[int, int, int]]]]:
            reads: Dict[str, Set[Tuple[int, int, int]]] = defaultdict(set)
            writes: Dict[str, Set[Tuple[int, int, int]]] = defaultdict(set)
            for stmt in node.body:
                self.visit(stmt, reads=reads, writes=writes)
            if node.mask:
                self.visit(node.mask, accesses=reads)
            return reads, writes

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> oir.VerticalLoop:
        assert node.horizontal_executions, "non-empty vertical loop expected"
        result = self.generic_visit(node, **kwargs)
        horizontal_executions = [result.horizontal_executions[0]]
        _, previous_writes = self.AccessCollector().visit(horizontal_executions[-1])
        for horizontal_execution in result.horizontal_executions[1:]:
            current_reads, current_writes = self.AccessCollector().visit(horizontal_execution)
            conflicting = {
                field
                for field, offsets in current_reads.items()
                if field in previous_writes and offsets ^ previous_writes[field]
            }
            if not conflicting and horizontal_execution.mask == horizontal_executions[-1].mask:
                horizontal_executions[-1].body += horizontal_execution.body
                for field, writes in current_writes.items():
                    previous_writes[field] |= writes
            else:
                horizontal_executions.append(horizontal_execution)
                previous_writes = current_writes
        result.horizontal_executions = horizontal_executions
        return result
