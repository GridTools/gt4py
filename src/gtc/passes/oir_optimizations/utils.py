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
from typing import Any, Dict, NamedTuple, Set, Tuple

from eve import NodeVisitor
from gtc import oir


class AccessCollector(NodeVisitor):
    """Collects all field accesses and corresponding offsets."""

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
        self,
        node: oir.HorizontalExecution,
        reads: Dict[str, Set[Tuple[int, int, int]]],
        writes: Dict[str, Set[Tuple[int, int, int]]],
        **kwargs: Any,
    ) -> None:
        for stmt in node.body:
            self.visit(stmt, reads=reads, writes=writes)
        if node.mask:
            self.visit(node.mask, accesses=reads)

    class Result(NamedTuple):
        reads: Dict[str, Set[Tuple[int, int, int]]]
        writes: Dict[str, Set[Tuple[int, int, int]]]

    @classmethod
    def apply(cls, node: oir.LocNode) -> "Result":
        result = cls.Result(reads=defaultdict(set), writes=defaultdict(set))
        cls().visit(node, **result._asdict())
        return result
