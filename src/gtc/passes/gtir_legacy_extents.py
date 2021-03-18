from dataclasses import dataclass, field
from typing import Any, Dict, Union

from eve import NodeVisitor
from eve.utils import XIterator
from gt4py.definitions import Extent
from gtc import gtir


def _iter_field_names(node: Union[gtir.Stencil, gtir.ParAssignStmt]) -> XIterator[gtir.FieldAccess]:
    return node.iter_tree().if_isinstance(gtir.FieldAccess).getattr("name").unique()


def _iter_assigns(node: gtir.Stencil) -> XIterator[gtir.ParAssignStmt]:
    return node.iter_tree().if_isinstance(gtir.ParAssignStmt)


def _ext_from_off(offset: gtir.CartesianOffset) -> Extent:
    return Extent(
        ((min(offset.i, 0), max(offset.i, 0)), (min(offset.j, 0), max(offset.j, 0)), (0, 0))
    )


class LegacyExtentsVisitor(NodeVisitor):
    @dataclass
    class AssignContext:
        left_extent: Extent
        assign_extents: Dict[str, Extent] = field(default_factory=dict)

    def visit_ParAssignStmt(
        self, node: gtir.ParAssignStmt, *, field_extents: Dict[str, Extent], **kwargs: Any
    ) -> None:
        left_extent = field_extents.setdefault(node.left.name, Extent.zeros())
        pa_ctx = self.AssignContext(left_extent=left_extent)
        self.visit(node.right, field_extents=field_extents, pa_ctx=pa_ctx, **kwargs)
        for key, value in pa_ctx.assign_extents.items():
            field_extents[key] |= value

    def visit_FieldAccess(
        self,
        node: gtir.FieldAccess,
        *,
        field_extents: Dict[str, Extent],
        pa_ctx: AssignContext,
        **kwargs: Any,
    ) -> None:
        pa_ctx.assign_extents.setdefault(
            node.name, field_extents.setdefault(node.name, Extent.zeros())
        )
        pa_ctx.assign_extents[node.name] |= pa_ctx.left_extent + _ext_from_off(node.offset)


def compute_legacy_extents(node: gtir.Stencil) -> Dict[str, Extent]:
    field_extents: Dict[str, Extent] = {}
    visitor = LegacyExtentsVisitor()
    for assign in _iter_assigns(node).to_list()[::-1]:
        visitor.visit(assign, field_extents=field_extents)
    return field_extents
