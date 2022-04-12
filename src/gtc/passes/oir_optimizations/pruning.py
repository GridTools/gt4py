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

from typing import Any, Dict

from eve import NOTHING, NodeTranslator, iter_tree
from gt4py.definitions import Extent
from gtc import oir
from gtc.passes.horizontal_masks import mask_overlap_with_extent
from gtc.passes.oir_optimizations.utils import compute_horizontal_block_extents


class NoFieldAccessPruning(NodeTranslator):
    def visit_HorizontalExecution(self, node: oir.HorizontalExecution) -> Any:
        try:
            next(
                iter(
                    acc
                    for left in node.iter_tree().if_isinstance(oir.AssignStmt).getattr("left")
                    for acc in left.iter_tree().if_isinstance(oir.FieldAccess)
                )
            )
        except StopIteration:
            return NOTHING

        return node

    def visit_VerticalLoopSection(self, node: oir.VerticalLoopSection) -> Any:
        horizontal_executions = self.visit(node.horizontal_executions)
        if not horizontal_executions:
            return NOTHING
        return oir.VerticalLoopSection(
            interval=node.interval,
            horizontal_executions=horizontal_executions,
            loc=node.loc,
        )

    def visit_VerticalLoop(self, node: oir.VerticalLoop) -> Any:
        sections = self.visit(node.sections)
        if not sections:
            return NOTHING
        return oir.VerticalLoop(
            loop_order=node.loop_order,
            sections=sections,
            caches=node.caches,
            loc=node.loc,
        )

    def visit_Stencil(self, node: oir.Stencil, **kwargs):
        vertical_loops = self.visit(node.vertical_loops, **kwargs)
        accessed_fields = (
            iter_tree(vertical_loops).if_isinstance(oir.FieldAccess).getattr("name").to_set()
        )
        declarations = [decl for decl in node.declarations if decl.name in accessed_fields]
        return oir.Stencil(
            name=node.name,
            vertical_loops=vertical_loops,
            params=node.params,
            declarations=declarations,
            loc=node.loc,
        )


class UnreachableStmtPruning(NodeTranslator):
    def visit_Stencil(self, node: oir.Stencil) -> oir.Stencil:
        block_extents = compute_horizontal_block_extents(node)
        return self.generic_visit(node, block_extents=block_extents)

    def visit_HorizontalExecution(
        self, node: oir.HorizontalExecution, *, block_extents: Dict[int, Extent]
    ) -> oir.HorizontalExecution:
        return self.generic_visit(node, block_extent=block_extents[id(node)])

    def visit_HorizontalRestriction(
        self, node: oir.HorizontalRestriction, *, block_extent: Extent, **kwargs: Any
    ) -> Any:
        overlap = mask_overlap_with_extent(node.mask, block_extent)
        return NOTHING if overlap is None else node
