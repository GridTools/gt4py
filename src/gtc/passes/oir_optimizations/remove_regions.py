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

from typing import Any, Dict, Optional, Tuple

import eve
from gtc import common, oir


FIELD_EXT_T = Dict[str, common.IJExtent]


def _overlap_in_axis(
    extent: Tuple[int, int],
    interval: common.HorizontalInterval,
) -> Optional[Tuple[int, int]]:
    """Return a tuple of the distances to the edge of the compute domain, if overlapping."""
    LARGE_NUM = 10000

    if interval.start.level == common.LevelMarker.START:
        start_diff = extent[0] - interval.start.offset
    else:
        start_diff = None

    if interval.end.level == common.LevelMarker.END:
        end_diff = extent[1] - interval.end.offset
    else:
        end_diff = None

    if start_diff is not None and start_diff > 0 and end_diff is None:
        if interval.end.offset <= extent[0]:
            return None
    elif end_diff is not None and end_diff < 0 and start_diff is None:
        if interval.start.offset > extent[1]:
            return None

    start_diff = min(start_diff, 0) if start_diff is not None else -LARGE_NUM
    end_diff = max(end_diff, 0) if end_diff is not None else LARGE_NUM
    return (start_diff, end_diff)


def _compute_extent_diff(
    extent: common.IJExtent, mask: common.HorizontalMask
) -> Optional[common.IJExtent]:
    diffs: Dict[str, Tuple[int, int]] = {}
    for axis in ("i", "j"):
        axis_diff = _overlap_in_axis(getattr(extent, axis), getattr(mask, axis))
        if not axis_diff:
            return None
        else:
            diffs[axis] = axis_diff

    return common.IJExtent(**diffs)


def _ijextent_from_offset(offset: common.CartesianOffset) -> common.IJExtent:
    return common.IJExtent(
        **{axis: (getattr(offset, axis), getattr(offset, axis)) for axis in ("i", "j")}
    )


class RemoveUnexecutedRegions(eve.NodeTranslator):
    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        fields_extents: FIELD_EXT_T = {}
        vertical_loops = [
            self.visit(loop, fields_extents=fields_extents)
            for loop in reversed(node.vertical_loops)
        ]
        return oir.Stencil(
            name=node.name,
            params=node.params,
            vertical_loops=vertical_loops,
            declarations=node.declarations,
        )

    def visit_VerticalLoopSection(
        self,
        node: oir.VerticalLoopSection,
        *,
        fields_extents: FIELD_EXT_T,
        **kwargs: Any,
    ) -> oir.VerticalLoopSection:
        horizontal_executions = [
            self.visit(execution, fields_extents=fields_extents, **kwargs)
            for execution in reversed(node.horizontal_executions)
        ]
        return oir.VerticalLoopSection(
            interval=node.interval, horizontal_executions=horizontal_executions
        )

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        fields_extents: FIELD_EXT_T,
        **kwargs: Any,
    ) -> oir.HorizontalExecution:
        compute_extent = (
            node.iter_tree()
            .if_is(oir.AssignStmt)
            .getattr("left")
            .if_is(oir.FieldAccess)
            .reduce(
                lambda extent, node: extent.union(fields_extents.setdefault(node.name)),
                init=common.IJExtent.zero(),
            )
        )

        filtered_body = self.visit(
            node.body, fields_extents=fields_extents, compute_extent=compute_extent, **kwargs
        )

        return oir.HorizontalExecution(body=filtered_body, declarations=node.declarations, **kwargs)

    def visit_MaskStmt(
        self,
        node: oir.MaskStmt,
        compute_extent: common.IJExtent,
        **kwargs: Any,
    ) -> oir.MaskStmt:
        if isinstance(node.mask, oir.HorizontalMask):
            dist_from_edge = _compute_extent_diff(compute_extent, node.mask)
            if dist_from_edge is None:
                return eve.NOTHING
        else:
            dist_from_edge = common.IJExtent.zero()

        self.visit(
            node.body,
            compute_extent=(compute_extent - dist_from_edge),
            **kwargs,
        )

        return node

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        fields_extents: FIELD_EXT_T,
        compute_extent: common.IJExtent,
        **kwargs: Any,
    ) -> oir.FieldAccess:
        """Take the union of this access (converted to field extent) with all existing extents."""
        extent = _ijextent_from_offset(node.offset)
        accumulated_extent = compute_extent + extent
        fields_extents[node.name] = fields_extents.get(node.name, common.IJExtent.zero()).union(
            accumulated_extent
        )

        return node
