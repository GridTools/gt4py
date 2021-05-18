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

from eve import NodeTranslator
from gtc import oir


class MaskInlining(NodeTranslator):
    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
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
        target_name = node.left.name
        if target_name in masks_to_inline:
            masks_to_inline[target_name] = node.right
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
        local_masks_to_inline = (
            node.iter_tree()
            .if_isinstance(oir.FieldAccess)
            .getattr("name")
            .filter(lambda name: name.startswith("mask_"))
            .to_set()
        )
        for mask_to_inline in local_masks_to_inline:
            masks_to_inline[mask_to_inline] = None
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
            caches=[c for c in node.caches if c.name not in masks_to_inline],
        )

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        masks_to_inline = dict(kwargs.get("masks_to_inline", {}))
        return oir.Stencil(
            name=node.name,
            params=node.params,
            vertical_loops=self.visit(
                node.vertical_loops,
                masks_to_inline=masks_to_inline,
            ),
            declarations=[
                d for d in node.declarations if d.name not in masks_to_inline
            ],
        )
