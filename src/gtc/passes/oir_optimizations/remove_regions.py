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

"""Removes horizontal executions that is never executed and computes the correct extents."""
from dataclasses import dataclass, field
from typing import Any, Dict

import eve
from gt4py.definitions import Extent
from gtc import oir
from gtc.passes import utils


class RemoveUnexecutedRegions(eve.NodeTranslator):
    @dataclass
    class Context:
        fields_extents: Dict[str, Extent] = field(default_factory=dict)

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        ctx = self.Context()
        vertical_loops = [self.visit(loop, ctx=ctx) for loop in reversed(node.vertical_loops)]
        return oir.Stencil(
            name=node.name,
            params=node.params,
            vertical_loops=vertical_loops,
            declarations=node.declarations,
        )

    def visit_VerticalLoopSection(
        self,
        node: oir.VerticalLoopSection,
        **kwargs: Any,
    ) -> oir.VerticalLoopSection:
        horizontal_executions = [
            self.visit(execution, **kwargs) for execution in reversed(node.horizontal_executions)
        ]
        return oir.VerticalLoopSection(
            interval=node.interval, horizontal_executions=horizontal_executions
        )

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        ctx: Context,
        **kwargs: Any,
    ) -> oir.HorizontalExecution:
        compute_extent = (
            node.iter_tree()
            .if_is(oir.AssignStmt)
            .getattr("left")
            .if_is(oir.FieldAccess)
            .reduce(
                lambda extent, node: extent.union(ctx.fields_extents.setdefault(node.name)),
                init=Extent.zeros(),
            )
        )

        filtered_body = self.visit(node.body, ctx=ctx, compute_extent=compute_extent, **kwargs)

        return oir.HorizontalExecution(body=filtered_body, declarations=node.declarations, **kwargs)

    def visit_MaskStmt(
        self,
        node: oir.MaskStmt,
        *,
        compute_extent: Extent,
        **kwargs: Any,
    ) -> oir.MaskStmt:
        if isinstance(node.mask, oir.HorizontalMask):
            dist_from_edge = utils.compute_extent_diff(compute_extent, node.mask)
            if dist_from_edge is None:
                return eve.NOTHING
        else:
            dist_from_edge = Extent.zeros()

        self.visit(
            node.body,
            compute_extent=(compute_extent - dist_from_edge),
            **kwargs,
        )

        return node

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        ctx: Context,
        compute_extent: Extent,
        **kwargs: Any,
    ) -> oir.FieldAccess:
        """Take the union of this access (converted to field extent) with all existing extents."""
        extent = utils.extent_from_offset(node.offset)
        accumulated_extent = compute_extent + extent
        ctx.fields_extents[node.name] = ctx.fields_extents.get(node.name, Extent.zeros()).union(
            accumulated_extent
        )

        return node
