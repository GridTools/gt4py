from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from eve import NodeVisitor
from eve.utils import XIterable
from gt4py.definitions import Extent
from gtc import common, gtir
from gtc.passes import utils


def _iter_field_names(node: Union[gtir.Stencil, gtir.ParAssignStmt]) -> XIterable[gtir.FieldAccess]:
    return node.iter_tree().if_isinstance(gtir.FieldDecl).getattr("name").unique()


def _iter_assigns(node: gtir.Stencil) -> XIterable[gtir.ParAssignStmt]:
    return node.iter_tree().if_isinstance(gtir.ParAssignStmt)


def _ext_from_off(
    offset: gtir.CartesianOffset,
    region: Optional[gtir.HorizontalMask],
    left_extent: Extent,
    allow_negative: bool = False,
) -> Extent:

    res = Extent.from_offset((offset.i, offset.j, 0))

    if region is not None:
        res_ext = []
        for ext, interval, left_ext in zip(res, (region.i, region.j), left_extent):
            dim_ext = list(ext)
            if interval.start is not None:
                if interval.start.level == common.LevelMarker.START:
                    dim_ext[0] += interval.start.offset - left_ext[0]
                else:
                    dim_ext[0] = 0
            if interval.end is not None:
                if interval.end.level == common.LevelMarker.END:
                    dim_ext[1] += interval.end.offset - left_ext[1]
                else:
                    dim_ext[1] = 0
            res_ext.append(tuple(dim_ext))
        res_ext.append((0, 0))
        res = Extent(tuple(res_ext))

    if not allow_negative:
        res = Extent(
            ((min(res[0][0], 0), max(res[0][1], 0)), (min(res[1][0], 0), max(res[1][1], 0)), (0, 0))
        )
    return res


FIELD_EXT_T = Dict[str, Extent]


class LegacyExtentsVisitor(NodeVisitor):
    def __init__(self, allow_negative=False):
        self._allow_negative = allow_negative
        super().__init__()

    @dataclass
    class AssignContext:
        left_extent: Extent
        assign_extents: Dict[str, Extent] = field(default_factory=dict)

    @dataclass
    class StencilContext:
        assign_conditions: Dict[int, List[gtir.FieldAccess]] = field(default_factory=dict)
        assign_masks: Dict[int, List[gtir.HorizontalMask]] = field(default_factory=dict)

    def visit_Stencil(
        self, node: gtir.Stencil, *, mask_inwards: bool, **kwargs: Any
    ) -> FIELD_EXT_T:
        field_extents: FIELD_EXT_T = {}
        ctx = self.StencilContext()
        for field_if in node.iter_tree().if_isinstance(gtir.FieldIfStmt):
            self.visit(field_if, ctx=ctx)
        for region in node.iter_tree().if_isinstance(gtir.HorizontalRegion):
            self.visit(region, ctx=ctx)
        for assign in reversed(_iter_assigns(node).to_list()):
            self.visit(assign, ctx=ctx, field_extents=field_extents)
        for name in _iter_field_names(node):
            # ensure we have an extent for all fields. note that we do not initialize to zero in the beginning as this
            #  breaks inward pointing extends (i.e. negative boundaries).
            field_extents.setdefault(name, Extent.zeros())
            if mask_inwards:
                # set inward pointing extents to zero
                field_extents[name] = Extent(
                    *((min(0, e[0]), max(0, e[1])) for e in field_extents[name])
                )
        return field_extents

    def visit_HorizontalRegion(
        self,
        node: gtir.HorizontalRegion,
        *,
        ctx: StencilContext,
        **kwargs: Any,
    ) -> None:

        ctx.assign_masks.update({id(assign): node.mask for assign in _iter_assigns(node).to_list()})

    def visit_ParAssignStmt(
        self,
        node: gtir.ParAssignStmt,
        *,
        ctx: StencilContext,
        field_extents: Dict[str, Extent],
        **kwargs: Any,
    ) -> None:
        horizontal_mask = ctx.assign_masks.get(id(node), None)
        left_extent = field_extents.setdefault(node.left.name, Extent.zeros())
        if horizontal_mask:
            dist_from_edge = utils.compute_extent_difference(left_extent, horizontal_mask)
            if dist_from_edge is None:
                return
        else:
            dist_from_edge = Extent.zeros()
        pa_ctx = self.AssignContext(left_extent=left_extent - dist_from_edge)
        self.visit(
            ctx.assign_conditions.get(id(node), []),
            field_extents=field_extents,
            pa_ctx=pa_ctx,
            region=horizontal_mask,
            **kwargs,
        )
        self.visit(
            node.right, field_extents=field_extents, pa_ctx=pa_ctx, region=horizontal_mask, **kwargs
        )
        for key, value in pa_ctx.assign_extents.items():
            if key not in field_extents:
                field_extents[key] = value
            else:
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
        field_extents: Dict[str, Extent],
        pa_ctx: AssignContext,
        region: Optional[gtir.HorizontalRegion] = None,
        **kwargs: Any,
    ) -> None:
        if region is not None:
            res_extent = []
            for off, interval, left_extent in zip(
                (node.offset.i, node.offset.j), (region.i, region.j), pa_ctx.left_extent
            ):
                interval = common.HorizontalInterval(start=interval.start, end=interval.end)
                ext = [0, 0]

                if interval.start is None:
                    ext[0] = off
                elif interval.start.level == common.LevelMarker.START:
                    ext[0] = max(off, off + interval.start.offset + left_extent[0])
                else:
                    ext[0] = 0

                if interval.end is None:
                    ext[1] = off
                elif interval.end.level == common.LevelMarker.END:
                    ext[1] = min(off, off + interval.end.offset + left_extent[1])
                else:
                    ext[1] = 0

                res_extent.append(ext)
            res_extent.append([0, 0])
            default_extent = Extent(res_extent)
        else:
            default_extent = Extent.from_offset((node.offset.i, node.offset.j, 0))
        if not self._allow_negative:
            clipped_extent = tuple((min(0, de[0]), max(0, de[1])) for de in default_extent)
            default_extent = Extent(clipped_extent)

        pa_ctx.assign_extents.setdefault(
            node.name, field_extents.setdefault(node.name, default_extent)
        )

        pa_ctx.assign_extents[node.name] |= pa_ctx.left_extent + _ext_from_off(
            node.offset,
            region=region,
            allow_negative=self._allow_negative,
            left_extent=pa_ctx.left_extent,
        )


def compute_legacy_extents(
    node: gtir.Stencil, allow_negative=False, mask_inwards=False
) -> FIELD_EXT_T:
    return LegacyExtentsVisitor(allow_negative).visit(node, mask_inwards=mask_inwards)
