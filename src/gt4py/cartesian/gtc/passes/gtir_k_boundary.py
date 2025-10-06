# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math
import typing
from typing import Any, Dict, Tuple, Union

from gt4py import eve
from gt4py.cartesian.gtc import gtir
from gt4py.cartesian.gtc.common import LevelMarker


def _iter_field_names(
    node: Union[gtir.Stencil, gtir.ParAssignStmt],
) -> eve.utils.XIterable[gtir.FieldAccess]:
    return node.walk_values().if_isinstance(gtir.FieldDecl).getattr("name").unique()


class KBoundaryVisitor(eve.NodeVisitor):
    """For every field compute the boundary in k, e.g. (2, -1) if [k_origin-2, k_origin+k_domain-1] is accessed."""

    def visit_Stencil(self, node: gtir.Stencil, **kwargs: Any) -> Dict[str, Tuple[int, int]]:
        field_boundaries = {name: (-math.inf, -math.inf) for name in _iter_field_names(node)}
        for vloop in node.vertical_loops:
            self.generic_visit(vloop.body, vloop=vloop, field_boundaries=field_boundaries, **kwargs)
        # if there is no left or right boundary set to zero
        for name, b in field_boundaries.items():
            field_boundaries[name] = (
                b[0] if b[0] != -math.inf else 0,
                b[1] if b[1] != -math.inf else 0,
            )
        return typing.cast(Dict[str, Tuple[int, int]], field_boundaries)

    def visit_FieldAccess(
        self,
        node: gtir.FieldAccess,
        vloop: gtir.VerticalLoop,
        field_boundaries: Dict[str, Tuple[Union[float, int], Union[float, int]]],
        **_: Any,
    ):
        boundary = field_boundaries[node.name]
        interval = vloop.interval
        if not isinstance(node.offset, (gtir.VariableKOffset, gtir.AbsoluteKIndex)):
            if interval.start.level == LevelMarker.START:
                boundary = (
                    max(-interval.start.offset - node.offset.k, boundary[0]),
                    boundary[1],
                )
            if interval.end.level == LevelMarker.END:
                boundary = (
                    boundary[0],
                    max(interval.end.offset + node.offset.k, boundary[1]),
                )
        if node.name in [decl.name for decl in vloop.temporaries] and (
            boundary[0] > 0 or boundary[1] > 0
        ):
            raise TypeError(f"Invalid access with offset in k to temporary field {node.name}.")
        assert node.name in field_boundaries
        field_boundaries[node.name] = boundary


def compute_k_boundary(node: gtir.Stencil) -> Dict[str, Tuple[int, int]]:
    # loop from START to END is not considered as it might be empty. additional check possible in the future
    return KBoundaryVisitor().visit(node)


def compute_min_k_size(node: gtir.Stencil) -> int:
    """Compute the required number of k levels to run a stencil."""

    min_size_start = 0
    min_size_end = 0
    biggest_offset = 0
    for vloop in node.vertical_loops:
        if (
            vloop.interval.start.level == LevelMarker.START
            and vloop.interval.end.level == LevelMarker.END
        ):
            if not (vloop.interval.start.offset == 0 and vloop.interval.end.offset == 0):
                biggest_offset = max(
                    biggest_offset,
                    vloop.interval.start.offset - vloop.interval.end.offset + 1,
                )
        elif (
            vloop.interval.start.level == LevelMarker.START
            and vloop.interval.end.level == LevelMarker.START
        ):
            min_size_start = max(min_size_start, vloop.interval.end.offset)
            biggest_offset = max(biggest_offset, vloop.interval.end.offset)
        else:
            min_size_end = max(min_size_end, -vloop.interval.start.offset)
            biggest_offset = max(biggest_offset, -vloop.interval.start.offset)
    minimal_size = max(min_size_start + min_size_end, biggest_offset)
    return minimal_size
