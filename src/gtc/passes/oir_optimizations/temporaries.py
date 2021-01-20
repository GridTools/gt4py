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
    """
    Removes unneeded temporary fields and accesses.

    Replaces temporaries with a single write and single read access within a single
    horizontal execution (and zero offsets) by direct assignment.

    E.g. replaces the following sequence of statements within a horizontal execution
        tmp[0, 0, 0] = expr()
        (other statements...)
        out[0, 0, 0] = tmp[0, 0, 0]
    by
        out[0, 0, 0] = expr()
        (other statements...)

    If there are any additional accesses to the output field between the two
    temporary accesses or if there are multiple accesses to the temporary anywhere
    inside the vertical loop, no optimization is performed.

    For example, this is kept as is:
        tmp[0, 0, 0] = expr()
        foo[0, 0, 0] = inout[0, 0, 0]
        inout[0, 0, 0] = tmp[0, 0, 0]
    """

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
                if not isinstance(symtable[field], oir.Temporary):
                    # can not discard non-temporary fields
                    continue

                if field not in global_candidates:
                    # temporary might be used in multiple horizontal executions
                    continue

                write_index = horizontal_execution.body.index(write)
                read_index = horizontal_execution.body.index(read)
                assert write_index < read_index, "unexpected write after read?"
                for stmt in horizontal_execution.body[write_index + 1 : read_index]:
                    # if there are any accesses to the destination field between the two to-be optimized accesses, we skip the optimization
                    if any(
                        name == read.left.name
                        for name in stmt.iter_tree().if_isinstance(oir.FieldAccess).getattr("name")
                    ):
                        break
                else:
                    write.left = read.left
                    horizontal_execution.body.remove(read)
                    result.declarations.remove(symtable[field])

        return result

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        result = self.generic_visit(node, symtable=node.symtable_, **kwargs)
        result.collect_symbols()
        return result
