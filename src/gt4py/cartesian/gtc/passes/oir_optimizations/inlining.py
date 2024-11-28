# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy as cp
from typing import Any, Dict, Optional, Set, Union, cast

from gt4py import eve
from gt4py.cartesian.gtc import oir


class MaskCollector(eve.NodeVisitor):
    """Collects the boolean expressions defining mask statements that are boolean fields."""

    def visit_AssignStmt(
        self, node: oir.AssignStmt, *, masks_to_inline: Dict[str, oir.Expr]
    ) -> None:
        if node.left.name in masks_to_inline:
            assert masks_to_inline[node.left.name] is None
            masks_to_inline[node.left.name] = node.right

    def visit_MaskStmt(
        self, node: oir.MaskStmt, *, masks_to_inline: Dict[str, oir.Expr], **kwargs: Any
    ) -> None:
        if isinstance(node.mask, oir.FieldAccess) and node.mask.name in masks_to_inline:
            # Find all reads in condition
            condition_reads: Set[str] = (
                masks_to_inline[node.mask.name]
                .walk_values()
                .if_isinstance(oir.FieldAccess, oir.ScalarAccess)
                .getattr("name")
                .to_set()
            )
            # Find all writes in body
            body_writes: Set[str] = {
                child.left.name for child in node.body if isinstance(child, oir.AssignStmt)
            }
            # Do not inline the mask if there is an intersection
            if condition_reads.intersection(body_writes):
                masks_to_inline.pop(node.mask.name)

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> Dict[str, oir.Expr]:
        masks_to_inline: Dict[str, Optional[oir.Expr]] = {
            mask_stmt.mask.name: None
            for mask_stmt in node.walk_values()
            .if_isinstance(oir.MaskStmt)
            .filter(lambda stmt: isinstance(stmt.mask, oir.FieldAccess))
        }
        self.visit(node.vertical_loops, masks_to_inline=masks_to_inline, **kwargs)
        assert all(value is not None for value in masks_to_inline.values())
        return cast(Dict[str, oir.Expr], masks_to_inline)


class MaskInlining(eve.NodeTranslator):
    """Inlines mask statements that are boolean mask fields with the expression that generates the field.

    Preconditions: Mask statements exist as boolean temporary field accesses.
    Postcondition: Mask statements exist as boolean expressions.
    """

    def visit_FieldAccess(
        self, node: oir.FieldAccess, *, masks_to_inline: Dict[str, oir.Expr], **kwargs: Any
    ) -> oir.Expr:
        if node.name in masks_to_inline:
            return cp.copy(masks_to_inline[node.name])
        return self.generic_visit(node, masks_to_inline=masks_to_inline, **kwargs)

    def visit_AssignStmt(
        self, node: oir.AssignStmt, *, masks_to_inline: Dict[str, oir.Expr], **kwargs: Any
    ) -> Union[oir.AssignStmt, eve.NothingType]:
        if node.left.name in masks_to_inline:
            return eve.NOTHING
        return self.generic_visit(node, masks_to_inline=masks_to_inline, **kwargs)

    def visit_Temporary(self, node: oir.Temporary, *, masks_to_inline, **kwargs):
        return node if node.name not in masks_to_inline else eve.NOTHING

    def visit_CacheDesc(self, node: oir.CacheDesc, *, masks_to_inline, **kwargs):
        return node if node.name not in masks_to_inline else eve.NOTHING

    def visit_Stencil(self, node: oir.Stencil, **kwargs):
        return self.generic_visit(node, masks_to_inline=MaskCollector().visit(node), **kwargs)
