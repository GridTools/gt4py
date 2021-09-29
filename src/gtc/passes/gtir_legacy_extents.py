from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from eve import NodeVisitor
from eve.utils import XIterable
from gt4py.definitions import Extent
from gtc import gtir


def _iter_field_names(node: Union[gtir.Stencil, gtir.ParAssignStmt]) -> XIterable[gtir.FieldAccess]:
    return node.iter_tree().if_isinstance(gtir.FieldDecl).getattr("name").unique()


def _iter_assigns(node: gtir.Stencil) -> XIterable[gtir.ParAssignStmt]:
    return node.iter_tree().if_isinstance(gtir.ParAssignStmt)


def _ext_from_off(offset: gtir.CartesianOffset) -> Extent:
    return Extent(
        ((min(offset.i, 0), max(offset.i, 0)), (min(offset.j, 0), max(offset.j, 0)), (0, 0))
    )


FIELD_EXT_T = Dict[str, Extent]


class LegacyExtentsVisitor(NodeVisitor):
    @dataclass
    class AssignContext:
        left_extent: Extent
        assign_extents: FIELD_EXT_T = field(default_factory=dict)

    @dataclass
    class StencilContext:
        assign_conditions: Dict[int, List[gtir.FieldAccess]] = field(default_factory=dict)

    def visit_Stencil(self, node: gtir.Stencil, **kwargs: Any) -> FIELD_EXT_T:
        field_extents = {name: Extent.zeros() for name in _iter_field_names(node)}
        ctx = self.StencilContext()
        for field_if in node.iter_tree().if_isinstance(gtir.FieldIfStmt):
            self.visit(field_if, ctx=ctx)
        for assign in reversed(_iter_assigns(node).to_list()):
            self.visit(assign, ctx=ctx, field_extents=field_extents)
        return field_extents

    def visit_ParAssignStmt(
        self,
        node: gtir.ParAssignStmt,
        *,
        ctx: StencilContext,
        field_extents: FIELD_EXT_T,
        **kwargs: Any,
    ) -> None:
        left_extent = field_extents.setdefault(node.left.name, Extent.zeros())
        pa_ctx = self.AssignContext(left_extent=left_extent)
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
        field_extents: FIELD_EXT_T,
        pa_ctx: AssignContext,
        **kwargs: Any,
    ) -> None:
        pa_ctx.assign_extents.setdefault(
            node.name, field_extents.setdefault(node.name, Extent.zeros())
        )
        pa_ctx.assign_extents[node.name] |= pa_ctx.left_extent + _ext_from_off(node.offset)


def compute_legacy_extents(node: gtir.Stencil) -> FIELD_EXT_T:
    return LegacyExtentsVisitor().visit(node)
