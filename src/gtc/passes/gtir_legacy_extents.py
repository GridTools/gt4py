from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from eve import NodeVisitor
from eve.utils import XIterator
from gt4py.definitions import Extent
from gtc import common, gtir
from gtc.passes import utils


def _iter_field_names(node: Union[gtir.Stencil, gtir.ParAssignStmt]) -> XIterator[gtir.FieldAccess]:
    return node.iter_tree().if_isinstance(gtir.FieldDecl).getattr("name").unique()


def _iter_assigns(node: gtir.Stencil) -> XIterator[gtir.ParAssignStmt]:
    return node.iter_tree().if_isinstance(gtir.ParAssignStmt)


def _ext_from_off(offset: gtir.CartesianOffset) -> common.IJExtent:
    return common.IExtent(
        i=(min(offset.i, 0), max(offset.i, 0)), j=(min(offset.j, 0), max(offset.j, 0))
    )


def _to_gt4py_extent(extent: common.IJExtent) -> Extent:
    return Extent(extent.i, extent.j, (0, 0))


class LegacyExtentsVisitor(NodeVisitor):
    @dataclass
    class AssignContext:
        left_extent: common.IJExtent
        assign_extents: Dict[str, common.IJExtent] = field(default_factory=dict)

    @dataclass
    class StencilContext:
        assign_conditions: Dict[int, List[gtir.FieldAccess]] = field(default_factory=dict)

    def visit_Stencil(self, node: gtir.Stencil, **kwargs: Any) -> Dict[str, common.IJExtent]:
        field_extents = {name: common.IJExtent.zeros() for name in _iter_field_names(node)}
        ctx = self.StencilContext()
        for field_if in node.iter_tree().if_isinstance(gtir.FieldIfStmt):
            self.visit(field_if, ctx=ctx)
        for assign in reversed(_iter_assigns(node).to_list()):
            self.visit(assign, ctx=ctx, field_extents=field_extents)
        return field_extents

    def visit_HorizontalRegion(self, node: gtir.HorizontalRegion, **kwargs: Any) -> None:
        self.visit(node.body, horizontal_mask=node.mask, **kwargs)

    def visit_ParAssignStmt(
        self,
        node: gtir.ParAssignStmt,
        *,
        ctx: StencilContext,
        field_extents: Dict[str, common.IJExtent],
        horizontal_mask: Optional[common.HorizontalMask] = None,
        **kwargs: Any,
    ) -> None:
        left_extent = field_extents.setdefault(node.left.name, common.IJExtent.zeros())
        if horizontal_mask:
            dist_from_edge = utils.compute_extent_difference(left_extent, horizontal_mask)
            if dist_from_edge is None:
                return
        else:
            dist_from_edge = common.IJExtent.zero()
        pa_ctx = self.AssignContext(left_extent=left_extent - dist_from_edge)
        self.visit(
            ctx.assign_conditions.get(id(node), []),
            field_extents=field_extents,
            pa_ctx=pa_ctx,
            **kwargs,
        )
        self.visit(node.right, field_extents=field_extents, pa_ctx=pa_ctx, **kwargs)
        for key, value in pa_ctx.assign_extents.items():
            field_extents[key] |= value

    def visit_FieldIfStmt(
        self, node: gtir.FieldIfStmt, *, ctx: StencilContext, **kwargs: Any
    ) -> None:
        for assign_id in node.iter_tree().if_isinstance(gtir.ParAssignStmt).map(id):
            ctx.assign_conditions.setdefault(assign_id, []).extend(
                node.cond.iter_tree().if_isinstance(gtir.FieldAccess).to_list()
            )

    def visit_FieldAccess(
        self,
        node: gtir.FieldAccess,
        *,
        field_extents: Dict[str, common.IJExtent],
        pa_ctx: AssignContext,
        **kwargs: Any,
    ) -> None:
        pa_ctx.assign_extents.setdefault(
            node.name, field_extents.setdefault(node.name, common.IJExtent.zeros())
        )
        pa_ctx.assign_extents[node.name] |= pa_ctx.left_extent + _ext_from_off(node.offset)


def compute_legacy_extents(node: gtir.Stencil) -> Dict[str, Extent]:
    return {
        name: _to_gt4py_extent(extent)
        for name, extent in LegacyExtentsVisitor().visit(node).items()
    }
