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

from typing import List

from eve import NodeTranslator
from gtc import oir

from .utils import AccessCollector


class MaskStmtMerging(NodeTranslator):
    def _merge(self, stmts: List[oir.Stmt]) -> List[oir.Stmt]:
        merged = [self.visit(stmts[0])]
        for stmt in stmts[1:]:
            stmt = self.visit(stmt)
            if (
                isinstance(stmt, oir.MaskStmt)
                and isinstance(merged[-1], oir.MaskStmt)
                and stmt.mask == merged[-1].mask
                and stmt.is_loop == merged[-1].is_loop
                and not (
                    AccessCollector.apply(merged[-1].body).write_fields()
                    & AccessCollector.apply(stmt.mask, is_write=False).fields()
                )
            ):
                merged[-1] = oir.MaskStmt(
                    mask=merged[-1].mask,
                    body=merged[-1].body + stmt.body,
                    is_loop=merged[-1].is_loop,
                )
            else:
                merged.append(stmt)
        return merged

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution) -> oir.HorizontalExecution:
        return oir.HorizontalExecution(body=self._merge(node.body), declarations=node.declarations)

    def visit_MaskStmt(self, node: oir.MaskStmt) -> oir.MaskStmt:
        return oir.MaskStmt(mask=node.mask, body=self._merge(node.body), is_loop=node.is_loop)
