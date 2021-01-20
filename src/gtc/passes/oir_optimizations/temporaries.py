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

import collections
from typing import Any, Dict, List, Mapping, Set, Tuple

from eve import NodeTranslator, NodeVisitor
from gtc import oir


class TemporaryDisposal(NodeTranslator):
    class WritesAndReads(NodeVisitor):
        def visit_FieldAccess(
            self, node: oir.FieldAccess, *, other_accesses: Dict[str, int], **kwargs: Any
        ) -> None:
            other_accesses[node.name] += 1

        def visit_AssignStmt(
            self,
            node: oir.AssignStmt,
            *,
            basic_reads: Mapping[str, List[oir.AssignStmt]],
            basic_writes: Mapping[str, List[oir.AssignStmt]],
            other_accesses: Set[str],
            **kwargs: Any,
        ) -> None:
            assert isinstance(node.left, oir.FieldAccess)
            basic_writes[node.left.name].append(node)

            if (
                isinstance(node.right, oir.FieldAccess)
                and node.right.offset.i == node.right.offset.j == node.right.offset.k == 0
            ):
                basic_reads[node.right.name].append(node)
            else:
                self.visit(node.right, other_accesses=other_accesses)

        @classmethod
        def apply(cls, node: oir.Stmt) -> Dict[str, Tuple[oir.AssignStmt, oir.AssignStmt]]:
            basic_reads: Dict[str, List[oir.AssignStmt]] = collections.defaultdict(list)
            basic_writes: Dict[str, List[oir.AssignStmt]] = collections.defaultdict(list)
            other_accesses: Dict[str, int] = collections.Counter()
            cls().visit(
                node,
                basic_reads=basic_reads,
                basic_writes=basic_writes,
                other_accesses=other_accesses,
            )

            result = dict()
            for field, bwrites in basic_writes.items():
                breads = basic_reads[field]
                if len(bwrites) == len(breads) == 1 and not other_accesses[field]:
                    result[field] = breads[0], bwrites[0]

            return result

    def visit_VerticalLoop(
        self, node: oir.VerticalLoop, symtable: Mapping[str, Any], **kwargs: Any
    ) -> oir.VerticalLoop:
        result = self.generic_visit(node, **kwargs)
        global_candidates = self.WritesAndReads.apply(result)

        for horizontal_execution in result.horizontal_executions:
            local_candidates = self.WritesAndReads.apply(horizontal_execution)
            for field, (read, write) in local_candidates.items():
                if field in global_candidates and isinstance(symtable[field], oir.Temporary):
                    write.left = read.left
                    horizontal_execution.body.remove(read)
                    result.declarations.remove(symtable[field])

        return result

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        result = self.generic_visit(node, symtable=node.symtable_, **kwargs)
        result.collect_symbols()
        return result
