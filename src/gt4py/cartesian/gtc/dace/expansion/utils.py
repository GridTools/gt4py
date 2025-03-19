# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, List

from gt4py import eve
from gt4py.cartesian.gtc import common, oir
from gt4py.cartesian.gtc.dace import daceir as dcir
from gt4py.cartesian.gtc.definitions import Extent


if TYPE_CHECKING:
    from gt4py.cartesian.gtc.dace.nodes import StencilComputation


class HorizontalIntervalRemover(eve.NodeTranslator):
    def visit_HorizontalMask(self, node: common.HorizontalMask, *, axis: dcir.Axis):
        mask_attrs = dict(i=node.i, j=node.j)
        mask_attrs[axis.lower()] = self.visit(getattr(node, axis.lower()))
        return common.HorizontalMask(**mask_attrs)

    def visit_HorizontalInterval(self, node: common.HorizontalInterval):
        return common.HorizontalInterval(start=None, end=None)


class HorizontalMaskRemover(eve.NodeTranslator):
    def visit_Tasklet(self, node: dcir.Tasklet):
        res_body = []
        for stmt in node.stmts:
            newstmt = self.visit(stmt)
            if isinstance(newstmt, list):
                res_body.extend(newstmt)
            else:
                res_body.append(newstmt)
        return dcir.Tasklet(
            label=f"he_remover_{id(node)}",
            stmts=res_body,
            read_memlets=node.read_memlets,
            write_memlets=node.write_memlets,
        )

    def visit_MaskStmt(self, node: oir.MaskStmt):
        if isinstance(node.mask, common.HorizontalMask):
            if (
                node.mask.i.start is None
                and node.mask.j.start is None
                and node.mask.i.end is None
                and node.mask.j.end is None
            ):
                return self.generic_visit(node.body)
        return self.generic_visit(node)


def remove_horizontal_region(node, axis):
    intervals_removed = HorizontalIntervalRemover().visit(node, axis=axis)
    return HorizontalMaskRemover().visit(intervals_removed)


def mask_includes_inner_domain(mask: common.HorizontalMask):
    for interval in mask.intervals:
        if interval.start is None and interval.end is None:
            return True
        elif (
            interval.start is None
            and interval.end is not None
            and interval.end.level == common.LevelMarker.END
        ):
            return True
        elif (
            interval.end is None
            and interval.start is not None
            and interval.start.level == common.LevelMarker.START
        ):
            return True
        elif (
            interval.start is not None
            and interval.end is not None
            and interval.start.level != interval.end.level
        ):
            return True
    return False


class HorizontalExecutionSplitter(eve.NodeTranslator):
    @staticmethod
    def is_horizontal_execution_splittable(he: oir.HorizontalExecution):
        for stmt in he.body:
            if isinstance(stmt, oir.HorizontalRestriction) and not mask_includes_inner_domain(
                stmt.mask
            ):
                continue
            elif isinstance(stmt, oir.AssignStmt) and isinstance(stmt.left, oir.ScalarAccess):
                continue
            return False

        # If the regions are not disjoint, then the horizontal executions are not splittable.
        regions: List[common.HorizontalMask] = []
        for stmt in he.walk_values().if_isinstance(oir.HorizontalRestriction):
            assert isinstance(stmt, oir.HorizontalRestriction)
            for region in regions:
                if region.i.overlaps(stmt.mask.i) and region.j.overlaps(stmt.mask.j):
                    return False
            regions.append(stmt.mask)

        return True

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution, *, extents, library_node):
        if not HorizontalExecutionSplitter.is_horizontal_execution_splittable(node):
            extents.append(library_node.get_extents(node))
            return node

        res_he_stmts = []
        scalar_writes = []
        for stmt in node.body:
            if isinstance(stmt, oir.AssignStmt):
                scalar_writes.append(stmt)
            else:
                assert isinstance(stmt, oir.HorizontalRestriction)
                new_he = oir.HorizontalRestriction(
                    mask=stmt.mask, body=[*scalar_writes, *stmt.body]
                )
                res_he_stmts.append([new_he])

        res_hes = []
        for stmts in res_he_stmts:
            accessed_scalars = (
                eve.walk_values(stmts).if_isinstance(oir.ScalarAccess).getattr("name").to_set()
            )
            declarations = [decl for decl in node.declarations if decl.name in accessed_scalars]
            res_he = oir.HorizontalExecution(declarations=declarations, body=stmts)
            res_hes.append(res_he)
            extents.append(library_node.get_extents(node))
        return res_hes

    def visit_VerticalLoopSection(self, node: oir.VerticalLoopSection, **kwargs):
        res_hes = []
        for he in node.horizontal_executions:
            new_he = self.visit(he, **kwargs)
            if isinstance(new_he, list):
                res_hes.extend(new_he)
            else:
                res_hes.append(new_he)
        return oir.VerticalLoopSection(interval=node.interval, horizontal_executions=res_hes)


def split_horizontal_executions_regions(node: StencilComputation):
    extents: List[Extent] = []

    node.oir_node = HorizontalExecutionSplitter().visit(
        node.oir_node, library_node=node, extents=extents
    )
    ctr = 0
    for i, section in enumerate(node.oir_node.sections):
        for j, _ in enumerate(section.horizontal_executions):
            node.extents[j * len(node.oir_node.sections) + i] = extents[ctr]
            ctr += 1
