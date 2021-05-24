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

import copy as cp
from typing import Any, Dict, Optional

from eve import NodeTranslator, NodeVisitor
from gtc import oir


class MaskDetection(NodeVisitor):
    def visit_AssignStmt(
        self,
        node: oir.AssignStmt,
        masks_to_inline: Dict[str, Optional[oir.Expr]],
    ) -> None:
        if node.left.name in masks_to_inline:
            masks_to_inline[node.left.name] = node.right

    def visit_VerticalLoop(
        self,
        node: oir.VerticalLoop,
        masks_to_inline: Dict[str, Optional[oir.Expr]],
        **kwargs: Any,
    ) -> oir.VerticalLoop:
        for mask_statement in node.iter_tree().if_isinstance(oir.MaskStmt):
            masks_to_inline[mask_statement.mask.name] = None
        self.visit(node.sections, masks_to_inline=masks_to_inline, **kwargs)

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> Dict[str, oir.Expr]:
        masks_to_inline: Dict[str, oir.Expr] = {}
        self.visit(node.vertical_loops, masks_to_inline=masks_to_inline, **kwargs)
        return masks_to_inline


class MaskInlining(NodeTranslator):
    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        masks_to_inline: Dict[str, Optional[oir.Expr]],
        **kwargs: Any,
    ) -> oir.Expr:
        if node.name in masks_to_inline:
            return cp.deepcopy(masks_to_inline[node.name])
        return self.generic_visit(node, masks_to_inline=masks_to_inline, **kwargs)

    def visit_AssignStmt(
        self,
        node: oir.AssignStmt,
        masks_to_inline: Dict[str, Optional[oir.Expr]],
        **kwargs: Any,
    ) -> oir.AssignStmt:
        if node.left.name in masks_to_inline:
            return None
        return self.generic_visit(node, masks_to_inline=masks_to_inline, **kwargs)

    def visit_MaskStmt(
        self,
        node: oir.MaskStmt,
        masks_to_inline: Dict[str, Optional[oir.Expr]],
        **kwargs: Any,
    ) -> oir.MaskStmt:
        return oir.MaskStmt(
            mask=self.visit(node.mask, masks_to_inline=masks_to_inline, **kwargs),
            body=self.visit(node.body, masks_to_inline=masks_to_inline, **kwargs),
        )

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        masks_to_inline: Dict[str, Optional[oir.Expr]],
        **kwargs: Any,
    ) -> oir.HorizontalExecution:
        return oir.HorizontalExecution(
            body=[
                stmt
                for stmt in self.visit(
                    node.body,
                    masks_to_inline=masks_to_inline,
                    **kwargs,
                )
                if stmt is not None
            ],
            declarations=node.declarations,
        )

    def visit_VerticalLoop(
        self,
        node: oir.VerticalLoop,
        masks_to_inline: Dict[str, Optional[oir.Expr]],
        **kwargs: Any,
    ) -> oir.VerticalLoop:
        return oir.VerticalLoop(
            loop_order=node.loop_order,
            sections=self.visit(
                node.sections,
                masks_to_inline=masks_to_inline,
                **kwargs,
            ),
            caches=[cache for cache in node.caches if cache.name not in masks_to_inline],
        )

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        masks_to_inline = MaskDetection().visit(node)
        return oir.Stencil(
            name=node.name,
            params=node.params,
            vertical_loops=self.visit(
                node.vertical_loops,
                masks_to_inline=masks_to_inline,
                **kwargs,
            ),
            declarations=[decl for decl in node.declarations if decl.name not in masks_to_inline],
        )
